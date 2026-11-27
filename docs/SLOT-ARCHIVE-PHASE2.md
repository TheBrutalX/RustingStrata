# Conversation archive: phase 2 analysis and implementation

Phase 1 is merged (PR #12, `edc0204`): automatic checkpoint on divergence/reclaim, prefix restore in single-slot
and `--parallel`, manifest/LRU/lineage, routes and metrics. This document analyses what phase 2 should change, in
what order, and what it costs. **Status: implemented (phase 2 subset).** The sections below keep the design
analysis, but the status notes and §15–§16 state what landed and what did not.

It is deliberately narrower than the phase-2 bullet list in [SLOT-ARCHIVE.md](SLOT-ARCHIVE.md): the goal is the
smallest set of changes that makes automatic archiving invisible in normal serving, then the lifecycle features
that operators need.

## Implementation status (phase 2 subset)

Landed on `feat/archive-phase2`:

- **v2 segmented entries.** New checkpoints are format 2: one `STRSTATE` state file holds the token list and the
  recurrent/DeltaNet/MTP/PLE/QSA snapshot (`STRRSEG` payload), and an ordered chain of content-addressed KV
  segments holds the KV rows (`STRKVSEG` payloads in `STRATSEG` files). The store is
  `<archive-dir>/segments.json` plus `<archive-dir>/segments/<2 hex>/<id>.seg`; ids chain the parent id, the
  fingerprint and the token list, and a segment is 1024 tokens. `Archive::save_segmented` /
  `restore_segmented`, `SegmentStore` refs/unref and GC, and shared prefixes are implemented. v1 `STRATASLOT`
  entries remain listed and restorable.
- **Leases and refcounts.** `lookup_leased` returns a `Lease`; `DELETE` answers `409` while an entry is leased and
  GC skips it. Deleting an entry releases its segment references; segments still referenced by another entry
  survive, and zero-ref segments are removed by the next GC (or at startup).
- **Lifecycle.** `--slot-archive-ttl`, `--slot-archive-per-key-gb`, `--slot-archive-pin-gb`,
  `POST /slots/archive/{id}?action=pin|unpin`, `pinned_bytes` in `GET` and metrics, and `expired` in the GC
  response and metrics are implemented.
- **I/O split in the GPU core.** `save_slot_to`/`restore_slot_from`, a bounded writer (`save_slot_file_async_as`,
  four 64 MiB chunks = 256 MiB) and the matching reader exist. v1-format entries restore through the async reader
  (disk read on a worker, H2D/install on the model thread). The v1 async writer API remains, but the server does
  not schedule background v1 saves for automatic checkpoints. **v2 `save_segmented` is still synchronous**, and
  the server's automatic checkpoints are v2.
- **Budget.** A v2 entry's `bytes` is its state file plus the segment bytes that entry created; a segment shared
  with an older entry is not charged again. The budget is logical, not a filesystem quota.

Not in this wave:

- asynchronous or chunked v2 save/restore, request-level `saving`/`restoring` slot states, and scheduler
  interleaving; an automatic v2 checkpoint still holds the archive mutex and the model thread for the segment
  writes, the state write and the GPU copies;
- a shutdown checkpoint (`--slot-archive-on-shutdown` does not exist);
- manifest batching/journal; `manifest.json` is still rewritten in full, and `segments.json` is a second full
  rewrite;
- v2 entry rebuild after manifest loss: the startup rebuild scans `.slot` v1 files only, while the segment store
  can rebuild its own manifest from `.seg` files;
- park/LRU eviction checkpoints, multi-version branching/branch ids, and compression or archive-only
  re-quantization;
- an owner-namespaced dedup mode; the default is global segment sharing with owner-scoped entries.

Measured results on the integration branch:

| suite | result |
|---|---|
| archive E2E (`tools/test_slot_archive.sh`) | 106/106: 28 single, 10 restart, 21 parallel, 15 parallel-restart, 30 lifecycle, 2 no-flag |
| dedup E2E (`tools/test_slot_archive_dedup.sh`) | 88/88: 63 full, 25 restart |
| explicit slot-save E2E (`tools/test_slot_save.sh`) | 43/43 |
| lib unit tests | 252/252 |
| GPU archive tests | `slot_archive_` 3/3, `slot_segment_` 2/2 |
| GPU slot/prefill regressions | `slot_` suite (async owner/equivalence included), `unified_`, `prefill_cached_reuses_history`, `prefill_matches_reference` |

The dedup run is the one that proves the storage claim: two conversations with a shared prefix produce segment
rows with `refs >= 2`, deleting one entry keeps the segments used by the other, a continuation restores from
those segments with `cached_tokens > 0`, and the surviving entry still restores after a process restart. The
lifecycle run covers TTL expiry, pin survival, unpin/expiry and the `404` on pinning a deleted id. No new timing
benchmark was run; the file decomposition in §2 remains the measured sizing evidence.

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
- There was **no reader refcount**: a DELETE/GC between lookup and restore could make the restore fail; it fell
  back to prefill. There was no admin pin route; `pinned` was honored by GC but nothing set it.
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
trade-off; zstd at -3 could be an optional later flag, not a default. The real large win was cross-conversation
deduplication of shared prefixes (same system prompt/tools/RAG) via a content-addressed segment store. It is no
longer a phase-3 idea: this wave implemented it as the format-2 layout below, and it changes storage layout rather
than compression.

### 2.3 Deduplication (chosen, implemented)

Deduplication was promoted from “phase 3 idea” to a chosen phase-2 workstream and is implemented as archive entry
format 2. It is the largest storage and write-amplification win, and it does not depend on quantization or
compression.

**Problem.** A phase-1 checkpoint wrote the whole slot again, including the KV rows of the system prompt, tools,
RAG documents and every earlier turn. A new turn wrote the full prefix; two conversations with the same system
prompt wrote it twice. Lineage pruning only removed older whole files; it did not share bytes.

**Unit.** A segment is a fixed 1024-token range of a history, aligned from position 0 (`SEGMENT_TOKENS`). It stores
the K/V rows for those positions in an opaque `STRKVSEG` payload inside a `STRATSEG` file. The id is a stable
FNV-1a hash over a domain tag, the fingerprint, the parent id (length-prefixed) and the token ids
(length-prefixed); `start` and `end` are implied by the parent chain and the token count. The design sketch below
was close, but the implementation does not hash `start`/`end` or a separate KV-layout field:

```
segment_id = FNV1a("STRATSEG-segment-v1", fingerprint, parent_id, tokens)
```

The parent chain is required: the KV of a token depends on the whole prefix, not only on the segment's own
tokens. A shared system prompt produces the same chain and therefore the same segments; a branch shares only
the prefix before its divergence. A v2 entry stores the ordered segment refs (`id`, `start`, `end`) and one
`STRSTATE` file with the token list and the final recurrent/MTP/PLE/QSA snapshot.

**Write path (as implemented).** `Archive::save_segmented` walks the history in 1024-token chunks and calls
`SegmentStore::put` for each. `put` reuses an existing content id and increments its refcount, so only segments
that do not exist yet are written; the new entry's state file is always written. The entry's `bytes` counts the
state file plus only the segment bytes this entry created. The GPU copy still covers every chunk in this wave:
`save_kv_segment` runs for existing segments too, then `put` compares the payload and skips the disk write. A
growing conversation therefore avoids rewriting old segment files, but not the whole-history D2H copy, until the
v2 save itself becomes chunked.

**Read path.** `restore_segmented` reads and validates the whole `STRSTATE` file, checks that the segment ranges
are a contiguous cover of the history, reads each segment (`get_checked` re-verifies its tokens), installs the KV
rows in order, then installs the recurrent snapshot and calls `finish_segmented_restore`. A missing/corrupt
segment rejects the entry and falls back to prefill. The v2 restore is synchronous and reads segments one at a
time; it does not use the bounded reader yet.

**GC and references.** Segments are content-addressed and refcounted by archive entries in `segments.json`.
Deleting one entry releases only its refs; GC removes a segment file when its refcount reaches zero, and a shared
segment disappears with its last entry. Archive entry TTL/pin/LRU applies to entries. A segment is shared by
entries but never exposed by the HTTP API. Segment files left by a crash, or unreferenced after a dropped entry,
are re-adopted or removed at startup, and the surviving v2 entries are the source of truth for the ref counts.

**Privacy.** Segments are global within the archive directory, so two entries with the exact same token prefix
share bytes. Entries remain owner-scoped and only an entry with the matching owner/fingerprint can be restored;
a shared segment does not give one key access to another key's entry. An owner-namespaced mode was proposed as
`--slot-archive-dedup owner` but **is not implemented**. The default is global within the server, matching the
fact that the same tokens produce the same KV.

**Trade-offs (with status).**

- Segment size is fixed at 1024 tokens; 512/2048 were not benchmarked. A smaller segment would share more and
  write less per change but add files and metadata.
- The fixed ~66 MB recurrent/MTP state is still stored per entry in `STRSTATE`. Deduplicating it at boundaries is
  possible (same chain → same state) but is not implemented.
- Directory layout is `segments/<2 hex>/<id>.seg` with refcounts in `segments.json`; orphan segments are
  re-adopted at startup. The archive entry's state file is `<2 hex>/<id>.state`.
- Explicit `--slot-save-path` files remain `STRATASLOT` v1. v1 archive entries remain readable and restorable;
  new archive entries use format 2.
- Crash safety: segments are written temp + fsync + rename before the entry manifest row; a crash leaves orphan
  segments, never a visible entry pointing at a missing segment.
- The win is largest for shared system prompts/tools/RAG and growing conversations. Unrelated one-shot
  conversations save the rewrite of their own prefix, which is the common case.

**Acceptance (verified by the dedup E2E).** Two conversations with a shared prefix reuse segments (`refs >= 2`);
a checkpoint writes only new segment files plus its state file; restore is exact; deleting one entry does not
delete segments still referenced by another, and the survivor still restores; the entry and its refs survive a
restart. The E2E does not construct a manifest-loss crash or an owner-isolation HTTP case beyond the unit tests.

Deduplication needs entry leases for safe refcounts; it landed with them and before the TTL/quota polish. It is
independent of the async writer, which is why the synchronous v2 path could land first.


## 3. Goals and non-goals

### Goals

1. A checkpoint stalls the model only for the GPU→host copies, not for disk writes; target a few hundred ms for
   a 1 GB state and no multi-second head-of-line block for large ones. **Not met for v2:** automatic checkpoints
   are synchronous and write segment/state files on the model thread.
2. A restore stalls the model for H2D + state install, not for the disk read; a failed restore still falls back.
   **Partial:** v1-format entries use the async reader; v2 restores are synchronous. Fallback is implemented.
3. Host staging memory is bounded by a flag (e.g. 128–512 MB) regardless of slot size. **Partial:** the async
   writer/reader use a fixed four-chunk × 64 MiB ring (256 MiB); there is no flag, and v2 does not use it.
4. A slot that gets no new request can be archived while other slots keep decoding. **Not met:** there is no
   scheduler interleaving yet.
5. Operator lifecycle: TTL, per-key quota, admin pin/unpin, explicit delete; refcounts protect in-flight work.
   **Met.**
6. Optional graceful-shutdown checkpoint with a deadline. **Not met.**
7. Manifest work stays O(1)-ish per hit for 10k+ entries; no full JSON rewrite on every LRU touch. **Not met:**
   the archive manifest and the segment manifest are still rewritten in full.
8. Clear metrics: queue depth, staged bytes, phase timings, GC reasons, failures. **Partial:** entries/bytes/
   budget/pinned/expired/hits/misses/restores/evictions/failures exist; job, staging and timing metrics do not.

### Non-goals (phase 2)

- Compression or encryption. KV is high-entropy; token history is small. Compression belongs to phase 3.
- Cross-model/KV restore, object-store/shared archive across servers, Web UI.
- Automatic branching semantics/UI. Keeping multiple versions and pinning a branch point is enough.
- Moving GPU work to a second thread. The backend context is not safe for concurrent forwards; all GPU calls
  stay on the model thread.

## 4. Workstreams and priority

| priority | stream | outcome | phase-2 status |
|---|---|---|---|
| P0 | A. Snapshot/install split | Host-side chunks; no full-state RAM | Core API landed (`save_slot_to`/`restore_slot_from`); v2 save/restore does not use the chunk ring |
| P0 | B. Async save writer | Disk off the model thread | API landed for v1; not used by v2 automatic checkpoints |
| P0 | C. Async restore reader | Disk read off the model thread | v1 entries only; v2 restore synchronous |
| P0 | D. Leases and lock scope | No DELETE/GC race; archive mutex not held over I/O | Leases landed; the archive mutex is still held across v2 I/O |
| P1 | E. Segment dedup | Share KV prefixes; write only new segments | **Implemented** (format 2) |
| P1 | F. Shutdown checkpoint | Opt-in, deadline, bounded | Not implemented |
| P1 | G. TTL / per-key quota / pin | Lifecycle and fairness | **Implemented** |
| P1 | H. Manifest batching/journal | 10k+ entries, fewer rewrites | Not implemented |
| P2 | I. Metrics/alerts and soak tests | Operability | Lifecycle metrics landed; no job/timing metrics or soak/benchmark |
| P3 | J. Multi-version/branch retention | Rollback/branch workflow | Not implemented |

P0 was one coherent change: splitting the format I/O and adding a bounded pipeline. Only the core split and the
v1 async path landed. G no longer needs to wait for a measured manifest size; it is already implemented.

## 5. Async save design

**Status:** the split writer exists in the GPU core, but the v2 archive does not use it. `save_slot_to` and
`save_slot_file_async_as` stream an immutable snapshot through a bounded four-chunk ring (64 MiB each), and
`AsyncSave` joins the fsync/rename; tests cover equivalence and the explicit-owner path. The archive's automatic
checkpoints are v2 `save_segmented` and run synchronously, with the archive mutex held. The rest of this section
is the target design for the remaining async work, not a description of the current v2 path.

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

- The target design marks the slot `saving`: not held by a request, not available for admission, not a unified
  victim. **Not implemented:** there is no request-level `saving` state.
- The target design copies one chunk per scheduler turn; other slots' forwards and decodes continue between
  chunks. **Not implemented:** there is no scheduler interleaving yet. In the v2 path the model thread runs the
  whole save, and the 256 MiB ring can only pace the model thread when the async writer API is used.
- The snapshot captures the token list and the exact buffer lengths once. The slot must not change until the last
  chunk is copied; a chat request that needs this slot waits or is routed to another free slot. If every slot is
  saving/held, the request queues as today.
- A queued chat waiting for a saving slot can raise the copy priority (copy the remaining chunks back-to-back)
  before admission. If the writer/disk is the bottleneck, the slot can be released once the host chunks are
  queued; the writer owns immutable data, so the slot can be reused. **Design only.**
- On failure, the model thread aborts, removes the temp file, releases the slot and logs; the chat path is
  unaffected. The async writer implements the temp removal/abort; v2 `save_segmented` releases its refs and state
  file on failure.
- On success, the writer fsyncs, renames, and commits the manifest. The slot was already released after its last
  D2H copy; the file is complete before it is visible to lookup.
- Explicit admin `action=save` and `action=checkpoint` stay synchronous, as they do today.

### 5.3 Ring and memory

- There is **no** `--slot-archive-stage-mb` flag. The async writer and reader use a compile-time ring of four
  64 MiB chunks (256 MiB).
- One active automatic save at a time is the design; the current server serializes v2 checkpoints on the archive
  mutex and model thread rather than through an explicit job queue.
- Chunk size is 64 MB.
- The ring holds raw bytes only; the format header/footer/token list are small (4 bytes/token).
- Host memory is checked against the existing `--cache-ram` budget; do not silently exceed it.

### 5.4 Slot state machine

```
empty/idle ──checkpoint──▶ saving ──done──▶ idle
   ▲                          │
   └────────abort─────────────┘
```

**Not implemented.** This is the target state machine. `GET /slots` has no `archiving` field, there is no
per-save timeout flag, and the v2 save leaves the slot marked saved only after the manifest commit.

## 6. Async restore design

**Status:** the v1-format path is implemented. `restore_slot_file_async` reads on a reader thread with the same
four-chunk ring and installs on the model thread; format-1 archive entries restore through it. Format-2 entries
use the synchronous `restore_segmented` and read segment files one at a time on the calling/model thread. There is
no request-level `restoring` slot state; `GET /slots/archive` exposes a lease-derived `restoring` boolean for an
entry, which can also be set by a lookup or save lease.

```
request thread / scheduler       reader thread              model thread
----------------------------     -------------              ------------
lookup header + exact tokens ──▶ lease entry
                                 read/validate chunks ──▶  H2D chunk
                                                            install state
                                 footer check              restore_state
                                                            release lease
```

- `lookup` stays cheap: header + token list only, exact verification; v2 reads the `STRSTATE` header/tokens, v1
  reads the `STRATASLOT` header/tokens. It still runs on the model thread for automatic admission.
- The v1 reader opens the file, validates the header/fingerprint/lengths/footer while reading, and feeds host
  chunks to the model thread. The model thread performs H2D and installs the recurrent state. A full file read
  into RAM is avoided; the ring bounds memory. The v2 reader validates the whole state file and each segment
  before install, but performs the reads synchronously.
- The target slot is reserved and pinned during the operation in the design. **Not implemented** for v2 as a
  distinct state; the unified-growth rule and the existing pin cover admission.
- If the file is deleted after lookup, the lease prevents deletion while the read is active (§7).
- After install, `saved_ids`, `history`, `saved_owner`, `rope_offset` are set exactly as the synchronous restore
  does today; `cache_begin` / `cache_begin_unified` then reuse the prefix.
- On failure, reset the slot and fall back to prefill; never surface an error for an automatic restore. This is
  implemented for both formats.
- A restore that needs unified growth follows the phase-1 rule: only empty victims, never `force`. The restored
  slot is active/pinned, so it is not a victim.

## 7. Leases, refcounts and lock scope

**Leases are implemented; the lock-scope goal is not.** `lookup_leased` returns an owned guard, `DELETE` returns
`409` while the entry is leased, and GC skips it. The v2 save and restore still hold the archive `Mutex` across
the GPU copies and disk I/O because `save_segmented`/`restore_segmented` borrow `&mut Archive`.

- An in-memory lease on `ArchiveEntry`: `lookup`/`checkpoint` returns a guard, GC skips entries with
  `leases > 0`, and the guard is released after install/commit. This closes the "DELETE between lookup and
  restore" race. **Implemented.**
- `DELETE` can either wait for the lease or mark the entry deleted and unlink after the last lease. The code takes
  the simpler route: it answers `409` and the caller retries. **Implemented.**
- Never hold the archive `Mutex` across a GPU copy, disk write or fsync. The phase-1 `checkpoint_slot` did; the
  v2 path still does. **Not implemented.**
- Manifest operations become short: `commit`, `touch`, `delete`, `gc`. A separate writer thread never needs the
  lock for the file bytes. **Not implemented.**
- `pinned` (admin) and `leases` (in-flight) are different: pinned persists and is a policy choice; leases are
  transient and protect correctness. **Implemented.**

## 8. Shutdown checkpoint

**Not implemented in the phase-2 subset.** Phase 1 had no shutdown trigger (the design review selected divergence
and eviction only), and neither does this wave. Consequence: a `SIGTERM` loses states that were never
checkpointed; only earlier entries survive.

The remaining phase-2 option `--slot-archive-on-shutdown` (default **off**; recommended on for the overnight
workflow) is designed as:

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

**Implemented** as `--slot-archive-ttl`, `--slot-archive-per-key-gb` and `--slot-archive-pin-gb`, with
`POST /slots/archive/{id}?action=pin|unpin`, `pinned_bytes` and `expired`.

- `--slot-archive-ttl` (`30d`, `12h`, `45s`; default off/0): GC removes entries whose `last_used_ms` is older
  than the TTL; pinned entries are exempt. TTL is also a preference during checkpoint eviction. It is checked by
  GC and under budget pressure, not on a timer.
- `--slot-archive-per-key-gb`: two-level budget. At commit, if the owner is over quota, evict that owner's LRU
  unpinned entries first; then enforce the global budget. Owner 0 (no API keys) is a single bucket. An entry
  larger than the quota is refused.
- Admin pin: `POST /slots/archive/{id}?action=pin|unpin`, admin-only, persisted in the manifest, returned by
  `GET /slots/archive`. `--slot-archive-pin-gb` caps pinned bytes so an admin cannot pin the whole archive; a pin
  that would exceed it answers `409`.
- Pin is the tool for rollback/branch points and pre-compact snapshots; it is visible in `GET`.
- Purge: `DELETE /slots/archive?owner=N` or a `--purge` CLI remains a follow-up; per-entry delete is the phase-2
  surface.
- TTL and quotas use a per-owner byte scan at commit/GC, not on every request.

## 10. Manifest scale

**Not implemented.** The phase-2 subset adds a second manifest (`segments.json`) but no batching or journal.

Current costs:

- every commit/touch/delete/GC rewrites the whole archive JSON; `touch` is called for deduplicated checkpoints;
- the segment store rewrites `segments.json` on `put`, `unref`, `sync_refs` and GC;
- `GET /slots/archive` clones all entries before applying `limit`;
- startup rebuild reads every `.slot` file's token list when `manifest.json` is missing/corrupt; it does not
  rebuild v2 entries from `STRSTATE`/segments;
- `lookup` scans all entries for the owner/fingerprint and computes hashes for candidate lengths.

At 1k entries the current design is fine. At 10k (~3 MB JSON, ~10k files) it is borderline; at 100k it is not.

Options, in order of preference:

1. **Batch touches**: keep `last_used` in memory and flush it with the next structural write or every 60 s.
   Cheap, no format change. Do this first.
2. **Append-only journal** (`manifest.log` + periodic compaction): add/touch/delete/pin records, replay on
   startup, compact to JSON. Keeps crash safety and avoids full rewrites. Medium complexity.
3. **SQLite**: ACID and queries, but adds a dependency and cross-platform build weight; not justified unless the
   journal becomes complex.

Keep the v1 file format for explicit files and read-only support, keep the manifest schema version, and add a v2
entry row to the rebuild before the archive manifest can be called fully recoverable. Indexes remain an
implementation detail. Add an optional startup orphan scan (`--slot-archive-verify`) rather than scanning every
startup.

## 11. Metrics and observability

**Landed:** `strata_slot_archive_entries`, `_bytes`, `_budget_bytes`, `_pinned_bytes`, `_hits_total`,
`_misses_total`, `_restores_total`, `_evictions_total`, `_expired_total`, `_failures_total`, and the same fields
under `archive` in `/api/stats`. **Not landed:** job/queue/staging/timing metrics and GC-reason labels.

Remaining target:

- `archive_jobs{kind=save|restore}` active, queued, aborted, timed out;
- `archive_stage_bytes` (current ring usage), `archive_stage_limit`;
- save/restore phase timings: `lookup_ms`, `gpu_copy_ms`, `disk_ms`, `install_ms`, `total_ms` (simple buckets or
  totals + count, no new dependency);
- `archive_gc_reason{ttl|quota|lru|lineage|foreign|manual}`;
- `archive_leases`;
- disk free at the archive path (if available) and `archive_bytes / budget`;
- failures by phase (open/read/validate/write/rename/manifest).

Logs: one line per job with slot, entry id, n_tokens, bytes, phases, and outcome. Never log prompt text.

Alerts: archive bytes > 90% of budget, any failure rate, queue depth > 1 for minutes, disk free < reserve.

## 12. Testing and acceptance

### Verified in this wave

- `tools/test_slot_archive.sh`: 106/106 checks across single-slot, restart, `--parallel 2`, parallel restart,
  lifecycle and the no-flag server.
- `tools/test_slot_archive_dedup.sh`: 88/88 checks. It proves shared segment rows with `refs >= 2`, contiguous
  per-entry ranges, delete keeping the survivor's segments, a restored continuation with `cached_tokens > 0`, and
  a restore after a process restart.
- `tools/test_slot_save.sh`: 43/43, unchanged explicit save/restore.
- `cargo test --lib`: 252/252, including `slot_archive` (40 tests), `slot_segments` (12) and the GPU segment/async
  helper tests.
- GPU: `slot_archive_` 3/3, `slot_segment_` 2/2, `slot_` suite (async owner/equivalence included), `unified_`,
  `prefill_cached_reuses_history`, `prefill_matches_reference`.
- The lifecycle E2E covers TTL expiry, pin survival, unpin/expiry, `pinned_bytes`, and `404` on a deleted id.

### Correctness still to add

- chunked v2 save output equivalent to the synchronous writer, aborted at every chunk boundary with no visible
  entry and no temp file;
- async v2 save/restore while another slot decodes, including admission while a slot is saving;
- lease/refcount during a v2 read/install and during commit (the current E2E covers lookup-to-restore);
- shutdown with N slots, deadline reached and not reached;
- v2 manifest-loss rebuild and kill -9 during a segment/state write;
- owner-isolation over HTTP for shared segments (unit tests cover the owner gate, the E2E uses owner 0).

### Performance / soak (not run)

- 32k, 100k, 1M-token checkpoints and restores on the available model; record file bytes, GPU copy, disk, total
  stall with and without async.
- Host RSS with the fixed 256 MiB ring (and any future stage-size flag); no growth over repeated jobs.
- Manifest: 1k/10k/100k entries, load time, commit/touch time, list/GC time.
- 24 h switching workload: no fd/memory leaks, LRU/TTL settle, queue never permanently stuck.
- Acceptance target: a checkpoint of a 1 GB state adds no more than the GPU copy + a small scheduling overhead to
  the model thread; a restore does not read the disk on the model thread; a chat is never failed by the archive.
  The v2 path does not meet the first two yet.

## 13. Risks and mitigations

| risk | mitigation | phase-2 status |
|---|---|---|
| Host RAM blow-up on 1M states | per-segment payloads; the bounded ring exists only in the async API; no full snapshot in v2 | v2 avoids a full KV snapshot; the recurrent snapshot is read whole |
| Slot starvation while saving/restoring | one job at a time, priority/cancel, timeout, other slots serve | v2 serializes on the model thread; no job queue or timeout |
| Backend thread safety | all GPU calls on the model thread; disk only on workers | v1 async reader only; v2 also writes on the model thread |
| Snapshot changes under us | freeze the slot while copying; release after last D2H | automatic save only runs on an idle slot; no chunked freeze |
| GC/delete races | leases; manifest lock only for metadata | leases implemented; v2 I/O still under the lock |
| Manifest rewrite storms | batch touches now; journal if measured | not implemented |
| Shutdown hangs | deadline, most-recent first, skip/log, systemd timeout | no shutdown checkpoint |
| Disk wear/IOPS | per-segment writes, min-delta, one save at a time | implemented; no 64 MB sequential writes for v2 |
| TTL deletes a branch point | pin it; TTL exempts pinned | implemented |
| Privacy | opt-in, owner scope, 0600/0700, TTL/purge, no content in logs/API | implemented for entries; segments are global and never exposed |
| Complexity | keep the synchronous path for explicit admin save and tests; async only for automatic checkpoints | v2 is synchronous; the async path is the v1 API |

## 14. Parallel workstreams (delegation plan)

This was the plan after the chunk protocol was defined; the work landed mostly in sequence, and only the core
split, leases/lifecycle, dedup and the two server integrations shipped. The remaining items still map cleanly to
the same split:

| agent | files | task |
|---|---|---|
| core-pipeline | `src/gpu_model/slot.rs`, `src/slot_archive.rs`, `src/slot_segments.rs` | chunked v2 save/restore, ring, writer/reader |
| scheduler | `src/main.rs` | saving/restoring states, admission, priority, timeouts |
| lifecycle | `src/slot_archive.rs`, `src/main.rs` | lock scope, shutdown, manifest batching |
| verification | `tools/`, `tests/` | v2 async equivalence, fault injection, benchmarks, docs |

Dependencies: the scheduler needs a chunked v2 API; verification can write harnesses against the documented
states first. Keep `test_slot_archive.sh`, `test_slot_archive_dedup.sh` and `test_slot_save.sh` green throughout.

## 15. Chosen decisions and remaining questions

### Chosen (approved) with status

- **Deduplication: yes — implemented.** Content-addressed KV segments with parent-chain ids, global within the
  archive directory, owner-scoped entries, 1024-token segments, refcounted and GC'd. New entries use format 2
  (`STRSTATE` + segments); v1 entries remain readable. It landed with leases and before the TTL/quota polish.
- **Async pipeline: partially implemented.** A bounded ring, a split save/restore API and a v1 async reader
  exist. The v2 automatic checkpoint is synchronous; there is no explicit job queue, priority or timeout.
- **Explicit `action=checkpoint` stays synchronous — implemented.** Automatic checkpoints are also synchronous
  in this wave.
- **Leases/refcounts — implemented.** They protect in-flight entries. The archive mutex is still held over v2
  GPU copies and disk I/O, so the second half of the decision is not met.
- **Shutdown checkpoint is opt-in (off) with a 30 s deadline — not implemented.**
- **TTL and per-key quota are off by default — implemented.** Recommendations remain 30 days and 25% of the global
  budget (or 20 GB) on shared servers.
- **Pin is an admin-only route, capped at 10% of the budget — implemented.**
- **Manifest: batched JSON touches first; journal/SQLite only if a real 10k+ workload justifies it — not
  implemented.**
- **Compression: off — implemented.** q4/q2 is the storage lever; there is no archive-only re-quantization.
- **Owner-scoped restore, global segment sharing — implemented.** A shared segment never exposes another owner's
  entry.

### Still open (benchmark or operator input)

1. Segment size is fixed at 1024; benchmarking 512/1024/2048 remains open.
2. Ring size is fixed at four 64 MiB chunks (256 MiB); there is no flag and it has not been validated on NVMe.
3. Chunk size is fixed at 64 MiB; a smaller chunk was not measured.
4. Restore read-ahead depth > 1 is not implemented or benchmarked.
5. An optional conversation-id hint for listing/deletion UX (matching remains token-based) is not implemented.
6. The admission behavior while a slot is saving is moot until the `saving` state exists; the design target is a
   2 s wait, then another slot or prefill.
7. An optional zstd mode (6–8% saving) is not implemented.
8. Rebuilding v2 entry rows after `manifest.json` loss needs a design; today only v1 `.slot` files are scanned.
9. An owner-namespaced dedup mode (`--slot-archive-dedup owner`) is not implemented; the default is global.

### Implemented defaults

| setting | default | note |
|---|---|---|
| `--slot-archive-gb` | 50 GiB | 200 GiB for a large NVMe |
| `--slot-archive-min-tokens` | 256 | skip trivial states |
| `--slot-archive-min-delta` | 1024 | one checkpoint per meaningful growth, not per turn |
| `--slot-archive-versions` | 1 | latest per lineage; pin pre-compact/branch points |
| `--slot-archive-ttl` | 0 (off) | 30d recommended on shared servers |
| `--slot-archive-per-key-gb` | 0 (off) | 25% of global or 20 GB on hosting |
| `--slot-archive-pin-gb` | 10% of budget (5 GiB at 50) | cap admin pins; pin route admin-only |
| compression | off | q4/q2 is the real lever |
| explicit `action=checkpoint` | synchronous | deterministic response |

### Target-only defaults (not flags yet)

The following are design targets, not current settings: `--slot-archive-stage-mb` 256 MiB, one save + one
restore in flight, 60 s save timeout, 30 s restore timeout, 2 s admission wait for a saving slot,
`--slot-archive-on-shutdown` off with a 30 s deadline, periodic GC every 5 min, and warn/critical alerts at
80%/90%. There is no background GC timer; `POST /slots/archive/gc` is manual.

TTL and per-key quota are off by default to preserve phase-1 behavior; HOSTING recommends 30 days and a
per-key fraction for multi-tenant servers. The pin cap is a safety valve, not a target.

## 16. Recommendation and remaining order

What this wave delivered, in the originally recommended order:

1. ~~Chunked async save for automatic checkpoints~~ — **not done.** The split/async writer exists for the v1
   format, but v2 `save_segmented` is synchronous and holds the model thread.
2. ~~Async restore read + install split and the manifest lock scope~~ — **partly done.** v1 entries use the async
   reader; v2 restores are synchronous. Leases closed the GC/delete race; v2 still holds the archive mutex over
   I/O.
3. **Segment deduplication (v2 entries)** — **done.** Parent-chain KV segments, only new segments written,
   refcounted GC, owner-scoped entries with global segment sharing.
4. **TTL, per-key quota, pin/unpin** — **done.** The shutdown checkpoint from the same item is **not done**.
5. **Manifest batching/journal** — **not done.**
6. **Metrics, fault injection, soak, branching/version retention** — **partly done:** lifecycle metrics and the
   normal test suites, but no v2 fault injection, no timing benchmark, no soak, no branching.

The next phase-2 work should therefore start with the v2 async save/restore and scheduler interleaving, because
that is the remaining multi-second model-thread stall. Then add the shutdown checkpoint and the v2 manifest
rebuild/journal.

USAGE/HOSTING document the measured decomposition: the 10.9 GB/1M figure is q8 **KV**, correct for this model,
and a checkpoint also carries ~66 MB of fixed slot state; q4/q2 reduce the KV to ~5.8/3.9 GB per 1M. A 200 GB
archive is about 18M/34M/52M tokens at q8/q4/q2 before the fixed state per conversation. Do not promise lossless
compression: measured zstd/gzip/xz savings are 6–8% on q8 KV and not worth the CPU/dependency.
