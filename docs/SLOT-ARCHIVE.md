# Conversation archive

Disk-backed, automatic archiving of idle slot state, with transparent restore when a later request continues an
archived conversation. This is the next layer above [SLOT-SAVE.md](SLOT-SAVE.md): that document defines the
`STRATASLOT` v1 file and the explicit save/restore API; this one defines the **policy, index, budget and matching**
that make those files a cache the server manages on its own.

**Status: design proposal, not implemented.** The decisions below come from the design review:

- restore is automatic, by **exact token-prefix match** (no conversation id, no client change);
- checkpoint triggers are **before a slot is freed** and **before a prompt diverges from the slot history**
  (which covers switching conversation and compacting);
- a checkpoint on server shutdown is **not** in the first version; it is an opt-in follow-up (see §9.6);
- this branch adds the document only. No code yet.

The existing explicit `--slot-save-path` files remain unchanged. The archive is a second, opt-in directory.

## 1. Goal

A server with 2-8 slots in VRAM should behave as if it had hundreds of conversations available:

- when a slot is about to be reused, its state is written to disk instead of being dropped;
- when a request continues that conversation, the state is restored into a free slot and only the new suffix is
  prefilled;
- disk usage is capped by a configurable budget (for example 200 GB), with LRU eviction and no unbounded growth;
- a corrupt, foreign or missing archive file never fails a request: the server falls back to a normal prefill.

The target use cases are long agent sessions and long builds: switching away from a conversation, compacting its
context, or stopping work for the night should not throw away a prefill that took minutes.

## 2. Non-goals

- Binary compatibility with llama.cpp `.slot` files. The format stays `STRATASLOT` v1, as in SLOT-SAVE.
- Encryption or compression in v1. Files are plaintext, like the explicit slot files.
- Saving a slot in the middle of a generation. A checkpoint requires an idle slot whose `pos == history.len()`.
- Moving a live slot to disk asynchronously on the first version. The MVP can block the model thread for the copy;
  the host-staging split is phase 2 (§10.3).
- Cross-model or cross-KV restore. The fingerprint still decides compatibility.
- A UI for browsing or editing archive entries. The HTTP metadata/delete API is enough for v1.
- Sharing entries between API keys by default. Entries are owner-scoped (§15).
- Guaranteeing a restore when every slot would have to evict a live conversation. Automatic restore uses free slots
  only; explicit admin restore keeps the existing `force=1` semantics.

## 3. Current baseline

Already in `main` from SLOT-SAVE:

- `src/slot_file.rs`: `STRATASLOT` v1 header, token list, framed buffers, footer, filename validation;
- `src/gpu_model/slot.rs`: streamed `save_slot_file`, `restore_slot_file`, `erase_slot`, stable fingerprint;
- `GpuModel::history` / `Seq::history`: full prompt + generated-token history;
- `cache_begin` / `cache_begin_unified`: prefix reuse from `saved_ids`, including after a restore;
- `Sched` control jobs, pinning, `--slot-save-path`, `GET /slots` and the explicit POST routes;
- tests: format, history, unified reuse, save/restore round-trip, 43-check E2E.

What is missing for the archive:

- no index of saved states, so a request cannot find one;
- no automatic trigger: save happens only on an explicit HTTP request with a filename;
- no budget or eviction policy;
- save and restore run on the request path with no background writer;
- no owner-safe bookkeeping distinct from the current request's owner;
- no metrics or listing for the cache.

## 4. Terms

- **Working set**: the slots currently resident in VRAM (`GpuModel::slots()`), with their KV and recurrent state.
- **Archive entry**: one immutable `STRATASLOT` v1 file plus one manifest row.
- **Checkpoint**: writing the state of an idle slot to the archive.
- **Prefix match**: an entry's token history is an exact prefix of the request's prompt tokens.
- **Lineage**: entry B extends entry A when A's history is a prefix of B's history. B's `parent` is A.
- **Hit**: a request whose prompt matches an entry and is restored from it.
- **Fallback**: the normal path, prefill from position 0 (or from whatever RAM prefix exists).

## 5. Why token-prefix matching

The KV rows and the recurrent state are a pure function of the token prefix, the model configuration and the KV
layout. They are not a function of a conversation id, a session cookie or a chat name. Two consequences:

1. Any request whose prompt starts with a saved history can reuse that state, even if it comes from another client
   that happens to share a long prefix (same system prompt, same tools, same document).
2. No client change is needed. OpenAI and Anthropic clients keep sending the full message list; the server
   tokenizes it and looks for the longest saved prefix.

The match is **exact on token ids**, never on text similarity, embeddings or message hashes:

- the entry must be from the same model/KV/YaRN/split fingerprint;
- the entry must belong to the same owner (API key id) unless the server has no keys at all;
- the entry's token list must equal the request's first `n` tokens;
- the entry's `n` must not exceed the request's `keep` or the current `--ctx`.

**Compact does not reuse the old prefix.** A compacted prompt is a summary plus recent messages: it is generally not
a prefix of the old conversation, so the old KV cannot accelerate it. The pre-compact entry is still valuable for
rollback and audit, and the post-compact prompt is prefilled once and then checkpointed as a new lineage. This
document does not pretend otherwise.

## 6. Architecture

```
             request
                |
          tokenize prompt
                |
      archive.lookup(owner, prompt[..keep], fingerprint)
          | hit                         | miss
          v                             v
  restore into a free slot       normal slot selection
          |                             |
  prefill only the suffix        prefill (cache_begin)
          \_____________________________/
                        |
                 generation starts

  slot idle and about to be reused / prompt diverges / unified reclaim
                        |
              archive.checkpoint(slot, reason)
                        |
          STRATASLOT file + manifest row
                        |
                 budget / GC check
```

Components:

- **`Archive`** (server side): directory, logical budget, manifest and prefix index, owner/refcount handling, GC,
  metrics. Owned by `Hosting` and shared with `Sched` (parallel) or `Runner::Gpu` (single slot).
- **`Sched` / `Runner`**: decide when a slot is about to be lost, call `checkpoint`, and look up an entry before
  admission.
- **`GpuModel`**: exposes `slot_tokens`, `saved_owner`, `slot_fingerprint`; performs the GPU copies in
  `save_slot_file` / `restore_slot_file`.
- **Writer thread** (phase 2): receives a host copy and streams it to disk while the model thread continues.

The archive must not live inside `GpuModel`: the GPU code knows how to copy a slot, not where the cache directory
is, how much disk it may use or which slot to pick. Triggers and matching belong to the scheduler.

## 7. Archive layout and manifest

### 7.1 Files

```
<archive-dir>/manifest.json
<archive-dir>/manifest.json.tmp.<pid>.<nanos>
<archive-dir>/<2 hex>/<id>.slot
<archive-dir>/<2 hex>/.<id>.<pid>.<nanos>.tmp
```

- The two-hex shard is the first byte of the entry id, so one directory does not hold tens of thousands of files.
- `id` is server-generated: `hex64(fingerprint) + "-" + hex64(owner) + "-" + n_tokens + "-" + hex64(prefix_hash)`.
  It is stable for the same state, which makes deduplication trivial.
- Entry files are ordinary `STRATASLOT` v1 files. An operator can copy one into `--slot-save-path` and restore it
  explicitly, subject to the filename validation of that route.
- Temp files are created `0600`; the directory is created `0700` if missing.

### 7.2 Manifest

`manifest.json` is a small JSON object rewritten atomically on change:

```json
{
  "version": 1,
  "entries": [
    {
      "id": "a1b2c3d4e5f60718-0000000000000007-1058-9f3c2a1b4d5e6f70",
      "file": "a1/a1b2c3d4e5f60718-....slot",
      "bytes": 77379688,
      "n_tokens": 1058,
      "owner": 7,
      "fingerprint": "a1b2c3d4e5f60718",
      "prefix_hash": "9f3c2a1b4d5e6f70",
      "created_ms": 1795766400000,
      "last_used_ms": 1795766400000,
      "parent": null,
      "pinned": false,
      "reason": "divergence",
      "format": 1
    }
  ]
}
```

Fields:

- `id`: stable identity; also the dedupe key.
- `file`: path relative to the archive directory; never accepted from a client.
- `bytes`: apparent file size, used for the budget. `st_blocks` could be used for allocated space; v1 uses `bytes`.
- `n_tokens`: the entry's history length, i.e. the number of KV rows it restores.
- `owner`: the **conversation owner**, not the owner of the request that triggered the checkpoint (§9.5).
- `fingerprint`: the stable `slot_fingerprint` from SLOT-SAVE.
- `prefix_hash`: a 64-bit hash of the full token history. It is only a candidate filter; the restore verifies the
  exact tokens, so a collision wastes a read but cannot select a wrong state.
- `created_ms`, `last_used_ms`: wall-clock milliseconds; LRU uses `last_used_ms`.
- `parent`: the longest entry whose history is a prefix of this one, if known; used for lineage pruning.
- `pinned`: admin protection from GC (a future `POST /slots/archive/{id}?action=pin`; v1 can reserve it).
- `reason`: `divergence`, `evict`, `checkpoint`, `shutdown` (phase 2).
- `format`: `1`; a future format bump can coexist.

`used_bytes` is derived by summing `bytes`; it is not stored, to avoid a second source of truth.

### 7.3 Atomicity and startup

- File: the existing save writes a temp file in the destination directory, fsyncs it, then renames it over the
  target. The archive uses the same path but with a server-generated name.
- Manifest: write `manifest.json.tmp.<pid>.<nanos>`, fsync, rename over `manifest.json`, fsync the directory on Unix
  (best effort elsewhere).
- Crash windows:
  - file renamed, manifest not updated: the file is an orphan. At startup it is adopted by reading its header and
    token list, or deleted if invalid;
  - manifest updated, file missing: the entry is dropped at startup;
  - temp file left: deleted at startup (no writer survives a restart).
- Startup validation is cheap: parse the manifest, stat each file, read the `STRATASLOT` header (magic, version,
  fingerprint, `n_tokens`, `n_parts`) and check `n_tokens <= ctx`. Full buffer/footer validation stays on restore.
- A corrupt `manifest.json` is moved aside (`manifest.json.bad.<ts>`) and rebuilt from the files in the shards.
- Entries with a foreign fingerprint are kept (they may be valid after switching back to that model) but are marked
  unusable for the current model and are the first GC candidates.
- `manifest.json` rewriting is O(entries). At 10k entries it is about 3 MB; rewriting on checkpoints and GC is
  acceptable for v1. If it becomes a problem, phase 2 can switch to an append-only journal or SQLite without
  changing the file format.

## 8. Matching and automatic restore

### 8.1 Lookup

```
lookup(owner, prompt, keep, fingerprint, ctx):
    n_max = min(keep, ctx)
    candidates = entries where
        owner == owner and fingerprint == fingerprint and n_tokens <= n_max
    for each distinct n_tokens, compute hash(prompt[..n_tokens])
    sort candidates by n_tokens descending
    for entry in candidates:
        if entry.prefix_hash == hash(entry.n_tokens)
           and entry file is readable
           and read_tokens(entry.file) == prompt[..entry.n_tokens]:
               return entry
    return none
```

- Hashing the request prefixes is linear in the prompt length, which tokenization already paid for.
- Candidates are usually few (tens to hundreds). A bucket by `(fingerprint, owner)` and sorted `n_tokens` keeps the
  scan small; a trie is unnecessary for v1.
- Exact verification reads only the token list at the start of the slot file (`4 * n_tokens` bytes), not the KV.
- The file is opened read-only. Multiple simultaneous restores of the same entry are safe; a refcount keeps GC from
  deleting it while a reader is active.

### 8.2 Slot selection and prefill

1. Prefer an in-RAM state when it is at least as long as the archive entry: no disk read.
2. Otherwise pick a free slot. Prefer an empty slot; among non-empty free slots, the least recently used.
3. Restore the entry into that slot. The existing restore validates fingerprint, counts, lengths and footer before
   writing GPU rows; on failure it resets the slot.
4. Mark the slot pinned (existing behavior), admit the request, and let `cache_begin` / `cache_begin_unified`
   reuse the restored prefix. The request prefills only `prompt[reused..]`.
5. Report the restored length in `usage.prompt_tokens_details.cached_tokens` through the existing `reused` value.

If no slot is free, the request queues as it does today. Automatic restore does **not** evict a non-empty slot in
v1: evicting a live conversation to load another is surprising and can cascade into more archives. The admin can
still restore explicitly with `force=1` where that behavior is wanted.

### 8.3 Unified KV

- A restored slot may need to grow. Reuse `GpuModel::kv_reclaim_plan(slot, n)` (new, §9.3) to see which slots would
  be reclaimed.
- Automatic restore proceeds only if every victim is **empty** (or there are no victims). If a non-empty victim
  would be needed, skip the archive hit and prefill normally. Automatic restore never uses `force`.
- The restored slot is pinned until a request uses it or `action=erase`, so a later growth cannot take it.
- `--ctx` changes: entries with `n_tokens > ctx` are skipped, never truncated; the KV layout fingerprint does not
  include the current capacity by design.

### 8.4 Failures

Every failure is best-effort:

- unreadable/corrupt/truncated entry: log once, mark it bad, delete it in the next GC, fall back;
- fingerprint mismatch: skip, never attempt;
- owner mismatch: skip, never attempt;
- manifest inconsistent: skip the entry, rebuild later;
- restore I/O error after GPU writes started: the existing restore resets the target slot, and the chat falls back
  to a normal prefill;
- archive directory unavailable (permissions, unmounted): disable the archive for the process, log once, keep
  serving.

The HTTP request must never receive a `4xx`/`5xx` because the cache missed or broke.

## 9. Checkpoint triggers

### 9.1 A prompt diverges from the slot history

This is the "switch conversation" and "compact" trigger. The scheduler knows the slot that will receive a chat and
the request's tokens/`keep`. Before the prefill:

```
let h = g.slot_tokens(slot);
let extends = !h.is_empty() && keep >= h.len() && ids[..keep].starts_with(h);
if !extends { archive.checkpoint(slot, "divergence"); }
```

- `cache_begin` currently calls `park()`/`reset()` in this case; `cache_begin_unified` calls `reset()`. The archive
  runs before those calls so the state is not lost.
- The trigger archives only if `h.len() >= --slot-archive-min-tokens`, and skips if the same history was already
  checkpointed (same `id`).
- A compacted prompt is just a diverging prompt: the pre-compact state is saved, then the compacted context is
  prefilled and, after the turn, checkpointed as a new lineage.
- If the agent wants a checkpoint *before* it builds the compacted prompt, `POST /slots/{id}?action=checkpoint`
  gives it that point explicitly. The automatic trigger covers the server-side view (the prompt arrives diverging).

### 9.2 A slot is about to be freed or reclaimed

- **Parallel admission**: the scheduler picks a free slot. If that slot holds another conversation and will be
  reset, checkpoint it first (same test as §9.1).
- **Unified growth**: before `kv_need`/`kv_grow_unified` drops the victims' caches, checkpoint each non-empty
  victim. The model already has the victim order in `kv_grow_unified`; expose a pure
  `kv_reclaim_plan(slot, new) -> Vec<usize>` that follows the same order and `fits` rule, and call it on the model
  thread before growth.
- **Park/LRU eviction**: when `park()` would drop an old parked state to make room in `--cache-ram`, that state can
  be checkpointed too. It is host-resident, so it does not touch the GPU; phase 2.
- **`erase`**: an explicit admin erase means delete, not archive. Do not checkpoint.

The hook must stay **outside** `drop_slot_cache`/`kv_grow_unified`: those run deep inside the GPU code, sometimes
mid-forward, and adding file I/O there would be fragile and recursive. The scheduler computes the plan, archives,
then lets the model grow.

### 9.3 Reclaim plan

Add to `GpuModel`:

```rust
/// The slots `kv_grow_unified(s, new)` would reclaim, in the order it would drop them.
/// Pure: reads capacities and `active`, changes nothing.
pub fn kv_reclaim_plan(&self, s: usize, new: usize) -> Vec<usize>;
```

It duplicates the `order`/`fits` loop of `kv_grow_unified` (or, better, both call one helper). The scheduler uses it
for both prompt growth and restore growth. The plan is deterministic on the single model thread; no other job can
change capacities between planning and growing.

### 9.4 Minimum work

Avoid archiving noise:

- `--slot-archive-min-tokens 256`: do not archive tiny states;
- `--slot-archive-min-delta 1024`: do not rewrite a lineage unless it grew by at least this many tokens since the
  last checkpoint;
- skip if the current history is byte-identical to the newest entry for that lineage (same `id`);
- never archive a busy slot (`held[slot] || prefill[slot]`), a slot with `pos != history.len()`, or an empty slot.

### 9.5 Owner of a checkpoint

A checkpoint must record the owner of the **conversation**, not the owner of the incoming request that triggered
the eviction. Otherwise owner B could archive owner A's slot and later match it, leaking A's state.

`GpuModel::saved_owner` is the owner of the saved state. Add:

```rust
pub fn slot_owner(&self, s: usize) -> u64; // saved_owner for slot s
```

and extend the save call with an explicit owner:

```rust
fn save_slot_file_as(&mut self, slot: usize, path: &Path, owner: u64) -> Result<u64>;
```

`save_slot_file` keeps its current signature and calls the new one with `self.owner`. The archive always uses
`slot_owner(slot)`, even when the triggering request belongs to another key. Matching still compares the entry's
owner with the **request's** owner; a mismatch is a miss.

If the server runs without API keys, every request has owner 0 and the archive is effectively server-wide; document
this in HOSTING.

### 9.6 Shutdown (not in v1)

The design review selected divergence and eviction, not shutdown. Consequence: if the **server process** stops,
states that were not checkpointed earlier are lost; only the last checkpoint per conversation survives. The
overnight case works without this trigger if the server stays up and the slot is later reused or diverges.

A phase-2 option can cover it:

- `--slot-archive-on-shutdown` (default off in v1, recommended on for the overnight workflow);
- on graceful `SIGTERM`, after the queue drains and before the model is dropped, checkpoint every non-empty slot;
- `SIGKILL`/crash cannot be covered; the last periodic/triggered checkpoint is the recovery point.

## 10. Save pipeline

### 10.1 v1: synchronous

1. The scheduler/model thread verifies the slot is idle and `n_tokens >= min`.
2. Compute the entry id and check for an existing identical entry; if present, touch `last_used_ms` and stop.
3. Ensure budget: GC if `used + estimated_bytes > budget - reserve`; if the entry cannot fit, skip.
4. Create the shard and a temp file, and call the existing streaming writer to it.
5. fsync and rename; insert/update the manifest atomically; update lineage and GC ancestors if configured.
6. Log `archive: saved slot N reason=... n=... bytes=... ms=...`.

The model thread is blocked for the disk copy, as with the explicit save. At 0.5-1 GB/s a 1 GB state costs 1-2 s;
a 10.9 GB 1M-token state costs 10-20 s. That is acceptable for an admin-style trigger but noticeable on every
conversation switch. Phase 2 removes the disk part from the model thread.

### 10.2 Dirty tracking and dedupe

- If the history is unchanged since the last checkpoint, do nothing.
- If the new history is an extension of an existing entry for the same lineage and the growth is below
  `--slot-archive-min-delta`, do nothing.
- Content-addressed ids make a repeated save of the same state a no-op (touch LRU, no disk write).
- A save that fails leaves the previous entry intact; the temp file is removed.

### 10.3 Phase 2: host staging

The GPU→host copy must stay on the model thread because the backend context is not thread-safe and the slot's
buffers are owned by `GpuModel`. The disk write does not:

1. On the model thread, copy the slot's recurrent and KV bytes into an owned host `Snapshot` (pinned memory if the
   backend supports it). This is PCIe-bound: about 0.1 s/GB at 10 GB/s.
2. Hand the `Snapshot` to a writer thread. The slot can immediately be reused; the snapshot is immutable.
3. The writer streams to disk, fsyncs, renames and updates the manifest.
4. On restore, read the file on a request/writer thread, then hand the bytes to the model thread for the H2D copy
   and state install.

This keeps the model stall to the GPU copy. It needs a `Snapshot` representation separate from `save_slot_file`'s
inline GPU reads; the format and the validation code stay shared.

### 10.4 Failure and disk-full

- ENOSPC: delete the temp, run GC once, retry once; if it still fails, log and skip. Never fail the chat.
- Manifest write failure: keep the file as an orphan; the next startup adopts or deletes it.
- A single entry larger than the budget is not archived (log once); it would evict everything and still not fit.
- Reserve headroom (1 GB or 2% of the budget) so a save does not fill the last byte and starve the manifest/temp.

## 11. Budget, GC and lineage

### 11.1 Budget

- `--slot-archive-gb N` (default 50 when the archive is enabled; 200 is a reasonable large-server value).
- `used_bytes = sum(entry.bytes)`. The budget is logical; the filesystem's free space is a second limit. If a
  `statvfs`-style check is available, require `free > new_bytes + reserve`; otherwise rely on ENOSPC handling.
- The budget includes only `*.slot` files; the manifest and temps are negligible but count in the reserve.

### 11.2 Eviction order

Before a write that would exceed the budget, evict candidates in this order:

1. entries with a foreign fingerprint (unusable for the current model);
2. entries flagged bad (missing file, failed restore);
3. unpinned ancestors when `--slot-archive-versions 1` (the child contains a longer history, but see the caveat);
4. least recently used, oldest first;
5. never a pinned entry, an entry with `refcount > 0`, or the entry currently being restored.

Deleting a file is best-effort; if unlink fails, keep the entry and try the next candidate. Update the manifest
after the batch.

### 11.3 Lineage caveat

A child is not a superset of the parent for **restore** purposes: the parent's KV/recurrent state is a snapshot at
an earlier position; the child cannot be rewound to the parent without a separate snapshot. Keeping only the child
is a disk/rollback trade-off:

- default `--slot-archive-versions 1`: keep the latest, delete ancestors; smallest disk, no rollback to earlier
  points;
- `--slot-archive-versions N`: keep up to N per lineage;
- admin `pin` protects a specific entry (a pre-compact checkpoint, a branch point);
- for the branching use case, pin the pre-branch entry and keep the children.

If rollback matters more than disk, set `N` or pin explicitly; the doc should not imply that a longer state can
replace a shorter one for time travel.

### 11.4 Sizing

With q8 KV on the 35B-A3B layout, SLOT-SAVE measures about 10.9 GB per 1M tokens, so:

| state | approximate file |
|---|---:|
| 8k tokens | ~90 MB |
| 32k tokens | ~350 MB |
| 100k tokens | ~1.1 GB |
| 256k tokens | ~2.8 GB |
| 1M tokens | ~10.9 GB |

- 200 GB ≈ 18M archived tokens, about 180 conversations of 100k tokens or 18 of 1M.
- The token history is 4 bytes/token (400 KB for 100k) and the recurrent state is a few MB per slot; both are small
  next to the KV.
- Lineage pruning and `min-delta` keep the archive close to "one latest snapshot per conversation" rather than one
  per switch.

## 12. CLI and HTTP API

### 12.1 Flags

```
--slot-archive-dir DIR        enable the archive; created 0700 if missing
--slot-archive-gb N           logical budget in GB (default 50)
--slot-archive-min-tokens N   do not archive states shorter than N (default 256)
--slot-archive-min-delta N    minimum growth since the last checkpoint (default 1024)
--slot-archive-versions N     keep N entries per lineage (default 1)
```

Phase 2: `--slot-archive-ttl`, `--slot-archive-per-key-gb`, `--slot-archive-on-shutdown`.

The archive and `--slot-save-path` should be different directories; if they are the same, the archive uses a
reserved subdirectory or refuses to start. Shard names are server-controlled; no client-supplied filename reaches
the archive.

### 12.2 Routes

All are admin-only, like `/api/load` and the explicit slot routes. Customer keys get `403`.

- `GET /slots/archive` → metadata:

  ```json
  {
    "dir": "/var/lib/strata/archive",
    "budget_bytes": 53687091200,
    "used_bytes": 12345678901,
    "entries": [
      {"id": "...", "owner": 7, "n_tokens": 1058, "bytes": 77379688,
       "created_ms": 1795766400000, "last_used_ms": 1795766400000,
       "pinned": false, "reason": "divergence", "fingerprint": "..."}
    ]
  }
  ```

  Optional `?limit=`, `?owner=`, `?fingerprint=`; no token contents and no file paths in the response.

- `DELETE /slots/archive/{id}` → `{"id": "...", "deleted": true, "freed_bytes": 77379688}`.
- `POST /slots/archive/gc` → `{"evicted": 3, "freed_bytes": 123}` (manual GC; same policy as automatic).
- `POST /slots/{id}?action=checkpoint` → `{"id_slot": 0, "id": "...", "n_saved": 1058, "n_written": 77379688,
  "timings": {"save_ms": 83.7}}`. This is the explicit pre-compact/branch checkpoint; it uses the archive budget,
  not a client filename.
- The existing `POST /slots/{id}?action=save|restore|erase` keep their `--slot-save-path` semantics unchanged.

Route ordering matters: `GET /slots/archive` and `DELETE /slots/archive/...` must be matched before the generic
`POST /slots/{id}` parser, otherwise `archive` fails to parse as an id.

### 12.3 Metrics and logs

Prometheus text at `/metrics`, JSON at `/api/stats`:

- `strata_slot_archive_bytes`, `strata_slot_archive_entries`;
- `strata_slot_archive_hits_total`, `_misses_total`, `_restores_total`, `_evictions_total`, `_failures_total`;
- `strata_slot_archive_save_seconds`, `_restore_seconds` (histograms or totals + count).

Logs: one line per save/restore/GC, with `slot`, `entry id`, `n_tokens`, `bytes` and `ms`; failure logs include the
reason but not the prompt text.

## 13. Single-slot and parallel

- **`--parallel N`**: `Sched` holds the archive handle. For every queued chat, before selecting a slot, look up an
  entry. Prefer an in-RAM match; else a free slot; else queue. The model thread performs the restore and prefill.
- **`--parallel 1` / direct `Runner::Gpu`**: one slot. `Runner::chat` checks the archive before
  `prefill_cached`; if the slot holds another conversation, checkpoint it, then restore the best entry if it fits.
  The server's model lock serializes this with other requests.
- **CPU / `Runner::Own`**: no archive; the routes behave as today. `GET /slots/archive` can still list metadata if
  the archive is configured, but there is no GPU state to save.
- **Image prompts**: v1 skips automatic archive because `rope_offset`/M-RoPE state is not a plain text prefix.
  The explicit save already supports image slots; archive support can come later with an image key.

## 14. Security and privacy

- The archive is **opt-in**. Enabling it means every checkpointed conversation's tokens and KV are written to disk.
- Files are plaintext, `0600`, in a `0700` directory. Use full-disk encryption if the host is not trusted.
- Owner-scoped by default: an entry is only matched by a request with the same API key id. The admin can list
  metadata and delete entries, but HTTP never serves file contents.
- A checkpoint records `saved_owner`, not the triggering request's owner (§9.5), so a cross-key eviction cannot
  re-label another key's conversation.
- Without API keys, owner 0 is shared by all requests; the archive is server-wide. Document this in HOSTING.
- `--slot-archive-gb`, `--slot-archive-min-tokens`, TTL and per-key quotas bound disk use and the damage of a
  misbehaving client.
- Manifest and filenames are server-generated; no traversal surface. Deletion takes an entry id, never a path.
- Fingerprint checks keep a model/KV/YaRN/split change from restoring incompatible state.
- Backups: an archive is a copy of user conversations. Include it in retention/purge policy or exclude it
  deliberately, exactly as HOSTING says for explicit slot files.

## 15. Invariants

1. Never restore without an exact token-prefix match, the same fingerprint and the same owner.
2. Never touch a slot before the file's header, lengths and footer pass validation.
3. Never fail an HTTP request because the archive missed, broke, is full or is unmounted.
4. Never checkpoint a busy slot, a partial state (`pos != history.len()`) or an empty slot.
5. Never evict a pinned entry, an entry with an active reader, or the entry being restored.
6. File and manifest updates are atomic (temp + fsync + rename); a crash leaves the old entry usable.
7. A failed restore resets the target slot (existing behavior) and falls back to prefill.
8. Automatic restore never uses `force` and never evicts a non-empty slot in v1.
9. The archive is content-addressed; saving the same state twice does not duplicate the file.
10. The archive never changes the token stream or the greedy output; it only changes how much is prefilled.

## 16. Test plan

### Unit (no GPU)

- manifest round-trip, atomic replace, corrupt manifest rebuild, orphan adoption;
- id stability/dedupe; lineage parent selection; LRU order; budget eviction; pinned/refcount protection;
- prefix-hash candidate filter and exact-token verification (including a deliberate hash collision);
- owner-scoped lookup; foreign fingerprint; `n_tokens > ctx`; path/shard validation;
- disk-full/ENOSPC behavior with a tiny budget; temp cleanup.

### GPU

- checkpoint a slot, erase it, restore through the archive, continue: `|dlogit|` identical to an uninterrupted run;
- divergence triggers a checkpoint before `cache_begin` resets;
- unified growth checkpoints the victims before `drop_slot_cache`;
- longest-prefix wins over a shorter in-RAM or archive entry;
- owner B with the same prompt does not restore owner A's entry;
- corrupt/truncated entry falls back to prefill and the request still succeeds;
- automatic restore skips when unified growth needs a non-empty victim;
- restored slot is pinned and the next request reuses it (`cached_tokens == n`).

### E2E (`tools/test_slot_archive.sh`)

- start with a small archive budget; chat in slot 0, switch to another conversation, return: the second visit
  restores and `cached_tokens` reflects the archived prefix;
- compact-like divergence: the pre-compact entry exists; the compacted prompt prefills and creates a new lineage;
- `GET /slots/archive`, `DELETE`, `POST action=checkpoint`, `POST /slots/archive/gc`;
- budget: fill with tiny entries, check LRU eviction and `used_bytes <= budget`;
- restart: manifest and entries survive; auto-restore still works;
- no `--slot-archive-dir`: behavior identical to today;
- customer key: `403` on archive routes; owner isolation.

### Performance

- 32k/100k/1M checkpoint and restore timings; model-thread stall; phase-2 host staging comparison;
- lookup cost with 1k/10k entries; manifest load time; GC time;
- ensure the normal prefill path is unchanged when the archive is off.

## 17. Rollout

### Phase 0 — this document

Agree on the matching rule, triggers, budget and owner semantics. No code.

### Phase 1 — MVP

- `--slot-archive-dir`, `--slot-archive-gb`, `--slot-archive-min-tokens`, `--slot-archive-min-delta`;
- manifest, startup load/rebuild, orphan adoption, LRU/budget GC;
- checkpoint on divergence and on unified reclaim (via `kv_reclaim_plan`);
- automatic longest-prefix restore into a free slot, owner/fingerprint gates, fallback;
- `GET /slots/archive`, `DELETE`, `POST action=checkpoint`, metrics;
- synchronous save/restore, full snapshot, no compression;
- unit/GPU/E2E tests, docs in USAGE/HOSTING, no behavior change without the flag.

### Phase 2 — operational polish

- host staging + writer thread; restore read on a worker;
- `--slot-archive-versions`, TTL, per-key quota, admin pin;
- park/LRU eviction checkpoints; optional shutdown checkpoint;
- prompt-cache interaction on restart; manifest journal/SQLite if needed;
- dashboards/alerts for bytes, hit rate, failures.

### Phase 3 — branching and scale

- multiple checkpoints per conversation, explicit branch ids, pinning;
- content-addressed segment store to avoid rewriting unchanged KV rows;
- optional compression of the token history (KV is high-entropy; expect little);
- shared archive across servers with the same fingerprint (network filesystem or object store);
- UI for browsing/deleting entries.

## 18. Open decisions

1. Default budget: 50 GB (conservative) or 200 GB (the review example)? The flag is explicit either way.
2. Should automatic restore ever evict a non-empty slot after checkpointing it, or always queue/fallback? The doc
   chooses fallback for v1.
3. Keep the latest entry per lineage by default (disk-efficient, no rollback) or keep N? The doc defaults to 1 and
   recommends pinning pre-compact/branch points.
4. Manifest as rewritten JSON or an append-only journal/SQLite? JSON for v1; revisit at 10k+ entries.
5. Is a conversation id worth adding as an optional hint, even though matching is by tokens? It could improve
   listing and deletion UX, but is not needed for correctness.
6. Shutdown checkpoint default: off in v1 (selected), recommended on for the overnight workflow? Decide before
   phase 2.
7. Image/M-RoPE archive support: v1 skips; the explicit save path already handles image slots.
8. Shared archive across API keys: keep owner-scoped by default; an admin-only "shared prefix" mode could improve
   hit rate for a common system prompt, at a privacy cost.

## 19. Acceptance criteria for phase 1

- With `--slot-archive-dir` and no client changes, a conversation that diverges (switch/compact) is checkpointed,
  and a later prompt that continues it is restored with only the suffix prefilled; `cached_tokens` reports the
  archived prefix.
- With `--kv-unified`, a restore grows the slot only when the victims are empty; the restored slot is pinned and
  reused; a non-empty victim means the archive hit is skipped, not a `409`.
- The archive never exceeds `--slot-archive-gb` (within one entry/headroom); LRU and lineage pruning are tested.
- A missing/corrupt/foreign/owner-mismatched entry falls back to a normal prefill and the request succeeds.
- A checkpoint records the conversation owner, even when triggered by another key.
- `GET /slots/archive` and `DELETE` work; customer keys get `403`; metrics expose bytes, entries, hits, failures.
- Restart rebuilds the manifest and entries remain usable.
- Without `--slot-archive-dir`, the server behaves exactly as today.
- The existing 179 lib tests, slot GPU tests, explicit slot E2E and `prefill_matches_reference` stay green.

## 20. Compatibility with SLOT-SAVE

- Archive entries are ordinary `STRATASLOT` v1 files. No format change is required for the MVP.
- The explicit `--slot-save-path` routes, filename validation and response fields are untouched.
- An operator can copy an archive entry into the explicit save directory and restore it by filename, or use the
  archive API; the fingerprint rules are the same.
- A future `STRATASLOT` v2 (checksum, compression) can be introduced behind the same manifest/version field.
