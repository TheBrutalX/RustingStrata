# Conversation archive: phase 2 analysis

Phase 1 is merged (PR #12, `edc0204`): automatic checkpoint on divergence/reclaim, prefix restore in single-slot
and `--parallel`, manifest/LRU/lineage, routes and metrics. This document analyses what phase 2 should change, in
what order, and what it costs. **Status: analysis, no code.**

It is deliberately narrower than the phase-2 bullet list in [SLOT-ARCHIVE.md](SLOT-ARCHIVE.md): the goal is the
smallest set of changes that makes automatic archiving invisible in normal serving, then the lifecycle features
that operators need.

## 1. What phase 1 actually does

- `GpuModel::save_slot_file` streams the slot to disk in 64 MB chunks. The model thread performs every
  GPU→host copy **and** every disk write; a checkpoint blocks all slots and decoding.
- `restore_slot_file` reads the file and writes the KV/recurrent state on the model thread; the disk read is on
  the same thread as forward work.
- `Sched`/`Runner` trigger a checkpoint on divergence, before a free slot is reused, and before unified reclaim.
- `checkpoint_slot` holds the archive `Mutex` for the whole save, including the GPU copy and the disk write. The
  explicit slot-state routes are unchanged.
- The manifest is a JSON file rewritten atomically on commit, delete, GC and `touch`. `lookup` reads only the
  header/token list, but does so on the model thread (parallel admission) or the request thread (single slot).
- There is **no reader refcount**: a DELETE/GC between lookup and restore can make the restore fail; it falls
  back to prefill. There is no admin pin route; `pinned` is honored by GC but nothing sets it.
- No TTL, no per-key quota, no shutdown checkpoint.
- `--cache-ram` defaults to 2 GiB (and is clamped to the host's free RAM); the archive has its own disk budget.

## 2. Measured cost, compression and quantization

A direct measurement on the 35B-A3B test model (`--ctx 8192`, 1,380-token prompt) separates the file into its
fixed per-slot state and its per-token KV:

| `--kv` | file bytes | fixed slot state | KV bytes | KV bytes/token | 1M-token KV |
|---|---:|---:|---:|---:|---:|
| q8 | 80,884,336 | 65,863,680 | 15,014,400 | 10,880 | ~10.9 GB |
| q4 | 73,818,736 | 65,863,680 | 7,948,800 | 5,760 | ~5.8 GB |
| q2 | 71,169,136 | 65,863,680 | 5,299,200 | 3,840 | ~3.9 GB |

The fixed slot state (recurrent/DeltaNet, MTP draft, PLE/QSA rings) is ~66 MB **per slot and does not scale with
tokens**. The header and token list are 4 bytes/token (5.5 KB here). Consequences:

- The current docs' **10.9 GB per 1M tokens** is the **q8 KV** figure and is correct for this model; the total is
  ~10.9 GB + 66 MB per slot, and the model's KV layout can change the constant. The earlier conclusion that a 1M
  state was ~50 GB was wrong: it divided the whole file (fixed state included) by the token count.
- A **full host snapshot is still not viable** for 1M-token states: ~11 GB q8, ~6 GB q4, ~4 GB q2, against a
  default 2 GiB host cache. Phase 2 must use a bounded chunk ring.
- A 200 GB archive is about **18M tokens at q8**, ~34M at q4, ~52M at q2, before subtracting the fixed ~66 MB
  per conversation. Sizing should still be based on a measured file (`bytes / n_tokens` after subtracting the
  fixed state), not a universal constant.
- A synchronous 1M checkpoint can block the model for tens of seconds (disk) plus a few seconds (PCIe); the q4/q2
  files halve/third the KV but not the fixed state. Async I/O is the main phase-2 lever; quantization is the main
  storage lever.

### 2.1 Quantization

`--kv` already supports `f16`, `q8`, `q4`, `q2` (and turbo types); the archive stores whatever representation the
running model uses, so:

- q8 is the default storage/quality trade-off (half of f16);
- q4 halves the KV again (~5.8 GB/1M here) and q2 is ~3.9 GB/1M, with different numerics;
- the fixed ~66 MB state is not affected by `--kv`; shrinking it needs a new format that stores f32 state as
  f16/bf16 (lossy) or compresses it;
- an **archive-only** re-quantization (save q8 as q4 and dequantize on restore) is possible but lossy and not
  bit-exact: it changes the continuation. Do not do it silently. If approximate resume is acceptable, expose it
  as an explicit mode and document the divergence; otherwise the operator chooses `--kv q4/q2` for the whole
  server, and the archive inherits it.

Recommendation: no new quantization layer in phase 2. Document the q4/q2 sizing options; add a quality note.
Revisit an archive-only q4 mode only if users ask for it and accept non-identical output.

### 2.2 Lossless compression

On the same q8 file:

| codec | bytes | ratio | time |
|---|---:|---:|---:|
| zstd -3 | 75,707,073 | 93.6% | 0.18 s |
| zstd -19 | 75,654,119 | 93.5% | 11.9 s |
| gzip -9 | 75,826,442 | 93.8% | 3.5 s |
| xz -9 | 74,505,592 | 92.1% | 41.5 s |

Split by region: the fixed state compresses to 94.4%, the q8 KV to 89.9%. The KV is high-entropy quantized data;
the fixed state is mostly f32 but still not very compressible. A 6-8% saving is not worth a new codec dependency,
CPU cost on save/restore and a format flag in phase 2. Token history is compressible but is 4 bytes/token (a few
MB per 1M), negligible next to the KV.

Recommendation: **do not compress in phase 2**. If disk is tight, q4/q2 is a 2-3x lever with a known quality
trade-off; zstd at -3 could be an optional later flag, not a default. The real large win is cross-conversation
deduplication of shared prefixes (same system prompt/tools/RAG) via a content-addressed segment store, which is
phase 3 and changes the storage layout, not a codec.

## 3. Goals and non-goals

### Goals

1. A checkpoint stalls the model only for the GPU→host copies, not for disk writes; target a few hundred ms for
   a 1 GB state and no multi-second head-of-line block for large ones.
2. A restore stalls the model for H2D + state install, not for the disk read; a failed restore still falls back.
3. Host staging memory is bounded by a flag (e.g. 128–512 MB) regardless of slot size.
4. A slot that gets no new request can be archived while other slots keep decoding.
5. Operator lifecycle: TTL, per-key quota, admin pin/unpin, explicit delete; refcounts protect in-flight work.
6. Optional graceful-shutdown checkpoint with a deadline.
7. Manifest work stays O(1)-ish per hit for 10k+ entries; no full JSON rewrite on every LRU touch.
8. Clear metrics: queue depth, staged bytes, phase timings, GC reasons, failures.

### Non-goals (phase 2)

- Compression or encryption. KV is high-entropy; token history is small. Compression belongs to phase 3.
- Cross-model/KV restore, object-store/shared archive across servers, Web UI.
- Automatic branching semantics/UI. Keeping multiple versions and pinning a branch point is enough.
- Moving GPU work to a second thread. The backend context is not safe for concurrent forwards; all GPU calls
  stay on the model thread.

## 4. Workstreams and priority

| priority | stream | outcome |
|---|---|---|
| P0 | A. Snapshot/install split | Host-side chunks; no full-state RAM |
| P0 | B. Async save writer | Disk off the model thread |
| P0 | C. Async restore reader | Disk read off the model thread |
| P0 | D. Leases and lock scope | No DELETE/GC race; archive mutex not held over I/O |
| P1 | E. Shutdown checkpoint | Opt-in, deadline, bounded |
| P1 | F. TTL / per-key quota / pin | Lifecycle and fairness |
| P1 | G. Manifest batching/journal | 10k+ entries, fewer rewrites |
| P2 | H. Metrics/alerts and soak tests | Operability |
| P3 | I. Multi-version/branch retention | Rollback/branch workflow |

P0 is one coherent change: splitting the format I/O and adding a bounded pipeline. E/F can proceed in parallel
once the job model exists. G should be justified by a measured manifest size, not done pre-emptively.

## 5. Async save design

### 5.1 Why chunked, not a full snapshot

A full host snapshot is simple but costs one state in RAM (up to tens of GB). A bounded ring costs
`depth × chunk` (for example 4 × 64 MB = 256 MB) and still removes disk latency from the model thread.

### 5.2 Pipeline

```
model thread                         writer thread
------------                         -------------
snapshot header + tokens  ────────▶   write header/tokens
for each part/buffer chunk:
    D2H 64 MB into ring slot ──────▶  write chunk, release slot
    (return to scheduler between chunks)
...
footer                                  fsync, rename temp -> final
                                        commit manifest
```

- The slot is marked `saving`: not held by a request, not available for admission, not a unified victim. It is
  still visible in `GET /slots` with an `archiving: true` extension (or `is_processing: true`; prefer a separate
  field).
- The model thread copies one chunk per scheduler turn; other slots' forwards and decodes continue between
  chunks. The ring is a `sync_channel` or a mutex+condvar; the writer backpressures when full.
- The snapshot captures the token list and the exact buffer lengths once. The slot must not change until the last
  chunk is copied; a chat request that needs this slot waits or is routed to another free slot. If every slot is
  saving/held, the request queues as today.
- A queued chat waiting for a saving slot can raise the copy priority (copy the remaining chunks back-to-back)
  before admission. If the writer/disk is the bottleneck, the slot can be released once the host chunks are
  queued; the writer owns immutable data, so the slot can be reused.
- On failure, the model thread aborts, removes the temp file, releases the slot and logs; the chat path is
  unaffected.
- On success, the writer fsyncs, renames, and commits the manifest. The slot was already released after its last
  D2H copy; the file is complete before it is visible to lookup.
- Explicit admin `action=save` can stay synchronous (the caller expects the file), while automatic checkpoints
  use the pipeline. `action=checkpoint` can also stay synchronous for a deterministic response.

### 5.3 Ring and memory

- `--slot-archive-stage-mb` (default e.g. 256): total host staging bytes across in-flight saves/restores.
- One active automatic save at a time; a second request queues. This bounds memory and disk contention; decoding
  continues while one slot streams.
- Chunk size 64 MB (same as the current writer) or 16 MB for lower latency; benchmark.
- The ring holds raw bytes only; the format header/footer/token list are small (4 bytes/token).
- Host memory is checked against the existing `--cache-ram` budget; do not silently exceed it.

### 5.4 Slot state machine

```
empty/idle ──checkpoint──▶ saving ──done──▶ idle
   ▲                          │
   └────────abort─────────────┘
```

- `saving` is protected by `Sched.active()` so unified growth cannot reclaim it.
- `GET /slots` reports it; a chat that cannot use another slot waits.
- A timeout (`--slot-archive-save-timeout`, e.g. 60 s) aborts a stuck save and logs; the slot returns to idle.

## 6. Async restore design

```
request thread / scheduler       reader thread              model thread
----------------------------     -------------              ------------
lookup header + exact tokens ──▶ lease entry
                                 read/validate chunks ──▶  H2D chunk
                                                            install state
                                 footer check              restore_state
                                                            release lease
```

- `lookup` stays cheap: header + token list only, exact verification; it should move off the model thread where
  possible (the request already has the tokens/owner).
- The reader opens the file, validates the header/fingerprint/lengths/footer while reading, and feeds host chunks
  to the model thread. The model thread performs H2D and installs the recurrent state. A full file read into RAM
  is avoided; the ring bounds memory.
- The target slot is reserved/`restoring` and pinned during the operation. Holding it prevents a concurrent chat
  from using a half-installed state.
- If the file is deleted after lookup, the reader has the data it already read or fails cleanly; a lease prevents
  deletion while the read is active (§7).
- After install, `saved_ids`, `history`, `saved_owner`, `rope_offset` are set exactly as the synchronous restore
  does today; `cache_begin` / `cache_begin_unified` then reuse the prefix.
- On failure, reset the slot and fall back to prefill; never surface an error for an automatic restore.
- A restore that needs unified growth follows the phase-1 rule: only empty victims, never `force`. The
  `restoring` slot is active/pinned, so it is not a victim.

## 7. Leases, refcounts and lock scope

- Add an in-memory lease on `ArchiveEntry`: `lookup`/`checkpoint` returns a guard, GC skips entries with
  `leases > 0`, and the guard is released after install/commit. This closes the "DELETE between lookup and
  restore" race.
- `DELETE` can either wait for the lease or mark the entry deleted and unlink after the last lease. Waiting is
  simpler; an admin delete of an in-flight entry is rare.
- Never hold the archive `Mutex` across a GPU copy, disk write or fsync. The phase-1 `checkpoint_slot` does; phase
  2 takes the lock only to reserve an id/path, update the manifest and run GC.
- Manifest operations become short: `commit`, `touch`, `delete`, `gc`. A separate writer thread never needs the
  lock for the file bytes.
- `pinned` (admin) and `leases` (in-flight) are different: pinned persists and is a policy choice; leases are
  transient and protect correctness.

## 8. Shutdown checkpoint

Phase 1 has no shutdown trigger (the design review selected divergence and eviction only). Consequence: a
`SIGTERM` loses states that were never checkpointed; only earlier entries survive.

Phase 2 option `--slot-archive-on-shutdown` (default **off**; recommended on for the overnight workflow):

1. After SIGTERM, stop admitting new chats; let accepted requests drain.
2. Before dropping the model, checkpoint every non-empty slot, most recently used first.
3. Use the same pipeline; wait up to `--slot-archive-shutdown-timeout` (e.g. 30 s). At the deadline, log the
   skipped slots and exit.
4. The `systemd` `TimeoutStopSec` must exceed this deadline plus the existing expert-profile save.
5. `SIGKILL`/crash cannot be covered; the last triggered checkpoint is the recovery point.

Because GPU copies must run on the model thread and slots are processed sequentially, four 1 GB slots cost a few
seconds of H2D plus disk time; four 10 GB slots can exceed a short deadline. Prioritize and bound; do not hang a
restart.

## 9. TTL, per-key quota and pin

- `--slot-archive-ttl` (e.g. `7d`, default off/0): GC removes entries whose `last_used_ms` is older than TTL;
  pinned entries are exempt. TTL is checked on GC/commit, not on a timer.
- `--slot-archive-per-key-gb`: two-level budget. At commit, if the owner is over quota, evict that owner's LRU
  entries first; then enforce the global budget. Owner 0 (no API keys) is a single bucket.
- Admin pin: `POST /slots/archive/{id}?action=pin|unpin` (or `PATCH` with `{"pinned":true}`), admin-only,
  persisted in the manifest, returned by `GET /slots/archive`. `--slot-archive-pin-gb` caps pinned bytes so an
  admin cannot pin the whole archive.
- Pin is the tool for rollback/branch points and pre-compact snapshots; it must be visible in `GET`.
- Purge: `DELETE /slots/archive?owner=N` or a `--purge` CLI is a follow-up; per-entry delete is enough for phase 2.
- TTL and quotas need a per-owner byte index; compute at commit/GC, not on every request.

## 10. Manifest scale

Current costs:

- every commit/touch/delete/GC rewrites the whole JSON; `touch` is called for deduplicated checkpoints;
- `GET /slots/archive` clones all entries before applying `limit`;
- startup rebuild reads every file's token list when the manifest is missing/corrupt;
- `lookup` scans all entries for the owner/fingerprint and computes hashes for candidate lengths.

At 1k entries the current design is fine. At 10k (~3 MB JSON, ~10k files) it is borderline; at 100k it is not.

Options, in order of preference:

1. **Batch touches**: keep `last_used` in memory and flush it with the next structural write or every 60 s.
   Cheap, no format change. Do this first.
2. **Append-only journal** (`manifest.log` + periodic compaction): add/touch/delete/pin records, replay on
   startup, compact to JSON. Keeps crash safety and avoids full rewrites. Medium complexity.
3. **SQLite**: ACID and queries, but adds a dependency and cross-platform build weight; not justified unless the
   journal becomes complex.

Keep the file format (`STRATASLOT` v1) and the manifest schema version; indexes are an implementation detail.
Add an optional startup orphan scan (`--slot-archive-verify`) rather than scanning every startup.

## 11. Metrics and observability

Add to `/metrics` and `/api/stats`:

- `archive_jobs{kind=save|restore}` active, queued, aborted, timed out;
- `archive_stage_bytes` (current ring usage), `archive_stage_limit`;
- save/restore phase timings: `lookup_ms`, `gpu_copy_ms`, `disk_ms`, `install_ms`, `total_ms` (simple buckets or
  totals + count, no new dependency);
- `archive_gc_reason{ttl|quota|lru|lineage|foreign|manual}`;
- `archive_pinned_bytes`, `archive_leases`;
- disk free at the archive path (if available) and `archive_bytes / budget`;
- failures by phase (open/read/validate/write/rename/manifest).

Logs: one line per job with slot, entry id, n_tokens, bytes, phases, and outcome. Never log prompt text.

Alerts: archive bytes > 90% of budget, any failure rate, queue depth > 1 for minutes, disk free < reserve.

## 12. Testing and acceptance

### Correctness

- Chunked save output is byte-identical to the synchronous writer (or at least a valid file that restores to the
  same logits); concatenate chunks and compare.
- Abort/cancel at every chunk boundary leaves no visible entry and no temp file; the slot is usable.
- Async save while another slot decodes: the decoding tokens and timings are unchanged; only the saving slot is
  unavailable.
- Async restore while another slot decodes: same.
- A chat waiting for a saving slot either waits within the timeout or uses another slot; no deadlock.
- Lease/refcount: DELETE/GC during lookup, read, install and commit; the entry survives until release.
- Shutdown: graceful signal with N slots, deadline reached and not reached; restart restores the saved ones.
- TTL/quota/pin: deterministic eviction order, owner isolation, pin exemption, persisted pin across restart.
- Crash between write/rename/manifest: startup adopts or drops the orphan; old entry remains valid.
- Kill -9 during save: temp cleanup at next start; no manifest pointing at a partial file.

### Performance / soak

- 32k, 100k, 1M-token checkpoints and restores on the available model; record file bytes, GPU copy, disk, total
  stall with and without async.
- Host RSS with `--slot-archive-stage-mb` at 64/256/1024 MB; no growth over repeated jobs.
- Manifest: 1k/10k/100k entries, load time, commit/touch time, list/GC time.
- 24 h switching workload: no fd/memory leaks, LRU/TTL settle, queue never permanently stuck.
- Acceptance target: a checkpoint of a 1 GB state adds no more than the GPU copy + a small scheduling overhead to
  the model thread; a restore does not read the disk on the model thread; a chat is never failed by the archive.

## 13. Risks and mitigations

| risk | mitigation |
|---|---|
| Host RAM blow-up on 1M states | chunk ring + `--slot-archive-stage-mb`; no full snapshots |
| Slot starvation while saving/restoring | one job at a time, priority/cancel, timeout, other slots serve |
| Backend thread safety | all GPU calls on the model thread; disk only on workers |
| Snapshot changes under us | freeze the slot while copying; release after last D2H |
| GC/delete races | leases; manifest lock only for metadata |
| Manifest rewrite storms | batch touches now; journal if measured |
| Shutdown hangs | deadline, most-recent first, skip/log, systemd timeout |
| Disk wear/IOPS | sequential 64 MB writes, min-delta, one save at a time |
| TTL deletes a branch point | pin it; TTL exempts pinned |
| Privacy | opt-in, owner scope, 0600/0700, TTL/purge, no content in logs/API |
| Complexity | keep the synchronous path for explicit admin save and tests; async only for automatic checkpoints |

## 14. Parallel workstreams (if delegated)

After the chunk protocol is defined (one interface commit), four agents can work independently:

| agent | files | task |
|---|---|---|
| core-pipeline | `src/gpu_model/slot.rs`, `src/slot_archive.rs` | snapshot/install chunk API, ring, writer/reader |
| scheduler | `src/main.rs` | saving/restoring states, admission, priority, timeouts |
| lifecycle | `src/slot_archive.rs`, `src/main.rs` | leases, lock scope, TTL/quota/pin, shutdown |
| verification | `tools/`, `tests/` | chunk equivalence, fault injection, benchmarks, docs |

Dependencies: scheduler needs the chunk API; lifecycle needs leases; verification can write harnesses against the
documented states first. Keep the phase-1 E2E (`test_slot_archive.sh`, `test_slot_save.sh`) green throughout.

## 15. Open decisions

1. Ring size/default (`--slot-archive-stage-mb`): 128, 256 or 512 MB? One active job or two?
2. Chunk size: keep 64 MB or drop to 16 MB for latency?
3. Should `action=checkpoint` be synchronous (simple, deterministic) or return a job id (async)? Recommendation:
   keep synchronous; automatic triggers async.
4. Shutdown checkpoint default: off (phase 1 choice) or on when the archive is enabled? Recommendation: off by
   default, documented as recommended for overnight use.
5. TTL default: off or 30 days? Per-key quota default: off or a fraction of the global budget?
6. Manifest: batched JSON first, journal only if 10k+ entries are real. Confirm target entry count.
7. Pin API shape and `--slot-archive-pin-gb` default.
8. Should a chat waiting for a saving slot cancel the save after a timeout, losing that checkpoint? Recommendation:
   wait up to a short deadline, then use another slot or prefill; never fail the chat.
9. Do we need restore read-ahead depth > 1? Benchmark on NVMe.
10. Compression: not in phase 2; revisit only for token history, not KV.

### Proposed defaults (tuning)

These are the values I would ship unless a benchmark says otherwise:

| setting | default | rationale |
|---|---|---|
| `--slot-archive-gb` | 50 GiB | phase-1 default; 200 GiB for a large NVMe |
| `--slot-archive-min-tokens` | 256 | skip trivial states |
| `--slot-archive-min-delta` | 1024 | one checkpoint per meaningful growth, not per turn |
| `--slot-archive-versions` | 1 | latest per lineage; pin pre-compact/branch points |
| `--slot-archive-stage-mb` | 256 (4 × 64 MB) | bounded host RAM; enough to hide disk latency |
| in-flight jobs | 1 save + 1 restore | bound memory/IO; prioritize restore |
| save timeout | 60 s | abort a stuck save; slot returns to idle |
| restore timeout | 30 s | fall back to prefill rather than hang admission |
| admission wait for a saving slot | 2 s | then use another slot or prefill; never fail the chat |
| `--slot-archive-on-shutdown` | off | phase-1 choice; opt-in, recommended for overnight use |
| shutdown deadline | 30 s | most-recent slots first; skip/log the rest |
| `--slot-archive-ttl` | 0 (off) | do not delete a user's long-lived session silently; 30d recommended on shared servers |
| `--slot-archive-per-key-gb` | 0 (off) | single-tenant default; 25% of global or 20 GB on hosting |
| `--slot-archive-pin-gb` | 10% of budget (5 GiB at 50) | cap admin pins; pin route admin-only |
| periodic GC | every 5 min + on commit | TTL/quota settle without a request |
| alert thresholds | warn 80%, critical 90% of budget | leave room for a new checkpoint/temp |
| compression | off | 6-8% saving, CPU/dependency cost; q4/q2 is the real lever |
| explicit `action=checkpoint` | synchronous | deterministic response; automatic triggers async |

TTL and per-key quota are off by default to preserve phase-1 behavior; HOSTING should recommend 30 days and a
per-key fraction for multi-tenant servers. The pin cap is a safety valve, not a target.

## 16. Recommendation

Do phase 2 in this order:

1. **Chunked async save** for automatic checkpoints, with a bounded ring, slot `saving` state, timeouts and the
   phase-1 synchronous path kept for explicit save/checkpoint. This is the change that makes the archive usable on
   long contexts.
2. **Async restore read** + install split, leases and the manifest lock scope. This removes the remaining stall and
   the GC/delete race.
3. **Shutdown checkpoint**, TTL, per-key quota, pin/unpin. These are small once the job model exists.
4. **Manifest batching**; journal/SQLite only if a real 10k+ workload justifies it.
5. **Metrics, fault injection, soak**, then reconsider branching/version retention.

USAGE/HOSTING should add the measured decomposition: the 10.9 GB/1M figure is q8 **KV**, correct for this model,
and a checkpoint also carries ~66 MB of fixed slot state; q4/q2 reduce the KV to ~5.8/3.9 GB per 1M. A 200 GB
archive is about 18M/34M/52M tokens at q8/q4/q2 before the fixed state per conversation. Do not promise lossless
compression: measured zstd/gzip/xz savings are 6–8% on q8 KV and not worth the CPU/dependency in phase 2.
