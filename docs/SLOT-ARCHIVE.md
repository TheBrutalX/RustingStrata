# Conversation archive

Disk-backed, automatic archiving of idle slot state, with transparent restore when a later request continues an
archived conversation. This is the next layer above [SLOT-SAVE.md](SLOT-SAVE.md): that document defines the
`STRATASLOT` v1 file and the explicit save/restore API; this one defines the **policy, index, budget and matching**
that make those files a cache the server manages on its own.

**Status: implemented (phase 1) on `feat/slot-archive`.** The decisions below are the design that the phase-1
code follows:

- restore is automatic, by **exact token-prefix match** (no conversation id, no client change);
- checkpoint triggers are **before a slot is freed** and **before a prompt diverges from the slot history**
  (which covers switching conversation and compacting);
- a checkpoint on server shutdown is **not** in phase 1; it is a phase-2 follow-up (see §9.6).

The existing explicit `--slot-save-path` files remain unchanged. The archive is a second, opt-in directory.

## Implementation status (phase 1)

Code:

- `src/slot_archive.rs`: directory and JSON manifest, startup load/rebuild and orphan adoption, prefix index,
  budget and GC, lazy lineage pruning, owner/fingerprint gates.
- `src/gpu_model/slot.rs`: `save_slot_file_as` with an explicit conversation owner, `slot_owner`,
  `slot_fingerprint`, and the existing validated restore.
- `src/gpu_model.rs`: slot accessors used by the triggers, `kv_reclaim_plan`, and its pure tests.
- `src/main.rs`: `--slot-archive-*` flags, the checkpoint triggers in single-slot and `--parallel` serving,
  `GET /slots/archive`, `DELETE /slots/archive/{id}`, `POST /slots/archive/gc`,
  `POST /slots/{id}?action=checkpoint`, and the archive metrics.
- `tools/test_slot_archive.sh`: single-slot end-to-end test (divergence/restore, explicit checkpoint, delete, GC,
  restart, 501 without the flag).

Verified on `feat/slot-archive`:

- `cargo check --locked` is clean.
- `src/slot_archive.rs`: 17 unit tests.
- GPU helpers: 9 pure `reclaim_plan` tests and the `slot_save_as_records_explicit_owner` GPU test.
- GPU regression: `slot_` 4/4, `unified_` 2/2, `prefill_cached_reuses_history`, `prefill_matches_reference`.
- `tools/test_slot_archive.sh`: 40/40 checks. A parallel-slot E2E is being added separately and is not part of
  those 40 checks.

Not in phase 1 (see §17): asynchronous host staging and a writer thread, shutdown checkpoint, TTL and per-key
quotas, explicit multi-version branching (the `--slot-archive-versions` flag exists and prunes lazily), an admin
pin route, and checkpointing parked host state on LRU eviction.

## 1. Goal

A server with 2-8 slots in VRAM should behave as if it had hundreds of conversations available:

- when a slot is about to be reused, its state is written to disk instead of being dropped;
- when a request continues that conversation, the state is restored into a free slot and only the new suffix is
  prefilled;
- disk usage is capped by a configurable budget (for example 200 GiB), with LRU eviction and no unbounded growth;
- a corrupt, foreign or missing archive file never fails a request: the server falls back to a normal prefill.

The target use cases are long agent sessions and long builds: switching away from a conversation, compacting its
context, or stopping work for the night should not throw away a prefill that took minutes.

## 2. Non-goals

- Binary compatibility with llama.cpp `.slot` files. The format stays `STRATASLOT` v1, as in SLOT-SAVE.
- Encryption or compression in v1. Files are plaintext, like the explicit slot files.
- Saving a slot in the middle of a generation. A checkpoint requires an idle slot whose `pos == history.len()`.
- Moving a live slot to disk asynchronously in phase 1. A checkpoint blocks the model thread for the copy; the
  host-staging split is phase 2 (§10.3).
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

What phase 1 added:

- an index of saved states (`manifest.json` plus the token-prefix lookup) so a request can find one;
- automatic triggers: a checkpoint on divergence and on unified reclaim, and an explicit admin checkpoint;
- a logical budget with LRU eviction and lazy lineage pruning;
- owner-safe bookkeeping: a checkpoint records `slot_owner`, not the requesting key;
- `GET /slots/archive`, `DELETE`, manual GC, `POST action=checkpoint` and archive metrics;
- best-effort restore: a missing, corrupt, foreign or owner-mismatched entry falls back to normal prefill.

Still phase 2 (or later): a background writer and asynchronous host staging, a shutdown checkpoint, TTL and
per-key quotas, an admin pin route, and multi-version branching.

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

- **`Archive`** (server side): directory, logical budget, manifest and prefix index, owner handling, GC, metrics.
  Owned by `Hosting` and shared with `Sched` (parallel) or `Runner::Gpu` (single slot).
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
- `pinned`: admin protection from GC. GC honors it, but phase 1 has no route that sets it; the pin route is
  phase 2.
- `reason`: `divergence`, `evict` or `checkpoint`; a manifest rebuild records `recovered`. `shutdown` is phase 2.
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
  fingerprint, owner, `n_tokens`) and compare it with the manifest row. Full buffer/footer validation stays on
  restore.
- A corrupt `manifest.json` is moved aside (`manifest.json.bad.<ts>`) and rebuilt from the files in the shards.
- Entries with a foreign fingerprint are kept (they may be valid after switching back to that model). A save
  under budget pressure evicts them before the least-recently-used entries; manual GC has no current-fingerprint
  argument, so it treats them as ordinary LRU candidates.
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
- The file is opened read-only; a lookup reads only its header and token list. Phase 1 has no refcount: an entry
  deleted between lookup and restore is simply a failed restore and the request falls back to prefill. Refcounting
  a reader is phase 2.

### 8.2 Slot selection and prefill

1. Prefer an in-RAM state when it is at least as long as the archive entry: no disk read.
2. Otherwise pick a free slot. Prefer an empty slot; among non-empty free slots, the least recently used.
3. If that free slot holds another conversation, checkpoint it first; then restore the entry. The existing restore
   validates fingerprint, counts, lengths and footer before writing GPU rows; on failure it resets the slot.
4. Mark the slot pinned (existing behavior), admit the request, and let `cache_begin` / `cache_begin_unified`
   reuse the restored prefix. The request prefills only `prompt[reused..]`. The pin is cleared once the request
   holds the slot, so a later unified reclaim is not blocked by a restore that has already been admitted.
5. Report the restored length in `usage.prompt_tokens_details.cached_tokens` through the existing `reused` value.

If no slot is free, the request queues as it does today. Automatic restore never takes a slot that a request
holds, and never uses the explicit `force=1`; it may replace a free slot's conversation only after checkpointing
it. The admin can still restore explicitly with `force=1` where that behavior is wanted.

### 8.3 Unified KV

- A restored slot may need to grow. `GpuModel::kv_reclaim_plan(slot, n)` (§9.3) reports which slots would be
  reclaimed.
- Automatic restore proceeds only if every victim is **empty** (or there are no victims). If a non-empty victim
  would be needed, skip the archive hit and prefill normally. Automatic restore never uses `force`.
- The restored slot is pinned until the request holds it or `action=erase` clears it, so a later growth cannot take
  it between the restore and admission.
- `--ctx` changes: entries with `n_tokens > ctx` are skipped, never truncated; the KV layout fingerprint does not
  include the current capacity by design.

### 8.4 Failures

Every failure is best-effort:

- unreadable/corrupt/truncated entry: log, skip it, and fall back; phase 1 does not mark it bad, so an admin can
  delete it or a manifest rebuild can drop it;
- fingerprint mismatch: skip, never attempt;
- owner mismatch: skip, never attempt;
- manifest row inconsistent with its file at startup: drop the row and rewrite the manifest;
- restore I/O error after GPU writes started: the existing restore resets the target slot, and the chat falls back
  to a normal prefill;
- archive directory or a single read/write failing at run time: log and skip the operation (a lookup is a miss);
  the chat keeps serving. A directory that cannot be opened at startup stops the server with the path in the error.

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
  victim. `GpuModel::kv_reclaim_plan(slot, new) -> Vec<usize>` follows the same order and `fits` rule; the
  scheduler calls it on the model thread before growth.
- **Park/LRU eviction**: when `park()` would drop an old parked state to make room in `--cache-ram`, that state can
  be checkpointed too. It is host-resident, so it does not touch the GPU; phase 2.
- **`erase`**: an explicit admin erase means delete, not archive. Do not checkpoint.

The hook must stay **outside** `drop_slot_cache`/`kv_grow_unified`: those run deep inside the GPU code, sometimes
mid-forward, and adding file I/O there would be fragile and recursive. The scheduler computes the plan, archives,
then lets the model grow.

### 9.3 Reclaim plan

`GpuModel` has:

```rust
/// The slots `kv_grow_unified(s, new)` would reclaim, in the order it would drop them.
/// Pure: reads capacities and `active`, changes nothing.
pub fn kv_reclaim_plan(&self, s: usize, new: usize) -> Vec<usize>;
```

It shares the `reclaim_plan` helper with `kv_grow_unified`. The scheduler uses it for both prompt growth and
restore growth. The plan is deterministic on the single model thread; no other job can change capacities between
planning and growing.

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

`GpuModel::saved_owner` is the owner of the saved state. `GpuModel` exposes:

```rust
pub fn slot_owner(&self, s: usize) -> u64; // saved_owner for slot s
```

and the save call takes an explicit owner:

```rust
pub fn save_slot_file_as(&mut self, slot: usize, path: &Path, owner: u64) -> Result<u64>;
```

`save_slot_file` keeps its current signature and calls the new one with `self.owner`. `save_slot_file_as`
restores the caller's request owner on success, error and unwind. The archive always uses `slot_owner(slot)`, even
when the triggering request belongs to another key. Matching still compares the entry's owner with the
**request's** owner; a mismatch is a miss.

If the server runs without API keys, every request has owner 0 and the archive is effectively server-wide; document
this in HOSTING.

### 9.6 Shutdown (phase 2)

Phase 1 selected divergence and eviction, not shutdown. Consequence: if the **server process** stops, states
that were not checkpointed earlier are lost; only the last checkpoint per conversation survives. The overnight
case works without this trigger if the server stays up and the slot is later reused or diverges.

A phase-2 option can cover it:

- `--slot-archive-on-shutdown` (off in phase 1, recommended on for the overnight workflow when added);
- on graceful `SIGTERM`, after the queue drains and before the model is dropped, checkpoint every non-empty slot;
- `SIGKILL`/crash cannot be covered; the last periodic/triggered checkpoint is the recovery point.

## 10. Save pipeline

### 10.1 Phase 1: synchronous

1. The scheduler/model thread verifies the slot is idle, not partial and `n_tokens >= min`; an image prompt
   (`rope_offset != 0`) is skipped.
2. Compute the entry id and check for an existing identical entry; if present, touch `last_used_ms` and stop. If
   the history extends an archived entry by less than `--slot-archive-min-delta`, stop.
3. Reserve the shard path and stream the slot into a temp file there with the existing writer; fsync it and rename
   it over the final name.
4. `commit_save` validates the file's header, owner and token list, then enforces the budget: under pressure it
   prunes ancestors per `--slot-archive-versions`, evicts foreign-fingerprint entries before LRU entries, and
   removes the new file if it still cannot fit.
5. Insert the manifest row and rewrite `manifest.json` atomically; log
   `archive: saved slot N reason=... n=... bytes=... ms=...`.

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

- ENOSPC or another write failure: remove the temp/new file, log and skip. Phase 1 does not retry after GC; the
  next checkpoint can try again. Never fail the chat.
- Manifest write failure: the file stays on disk as an orphan; the next startup adopts or deletes it.
- A single entry larger than `budget - 1 GiB` is not archived (log once); it would evict everything and still not
  fit.
- The 1 GiB reserve (`RESERVE_BYTES`) keeps room for the temp file and the manifest below the logical budget.

## 11. Budget, GC and lineage

### 11.1 Budget

- `--slot-archive-gb N` is GiB (1 GiB = 2^30 bytes). The default is 50 GiB when the archive is enabled; 200 GiB
  is a reasonable large-server value.
- `used_bytes = sum(entry.bytes)`. The budget is logical, with 1 GiB (`RESERVE_BYTES`) kept free for the temp file
  and the manifest. The filesystem's free space is a second limit; phase 1 relies on write/ENOSPC handling rather
  than a `statvfs` check.
- The budget includes only `*.slot` files; the manifest and temps are negligible but count in the reserve.

### 11.2 Eviction order

A checkpoint that would exceed `budget - 1 GiB` (`commit_save`) evicts in this order:

1. unpinned ancestors above `--slot-archive-versions`, for every lineage whose parent chain is not shared with
   another branch;
2. entries with a foreign fingerprint;
3. least recently used, oldest first.

It never evicts a pinned entry or the entry being written. Deleting a file is best-effort; if unlink fails, keep
the entry and try the next candidate, then rewrite the manifest after the batch.

`POST /slots/archive/gc` first drops rows whose file is missing, then, when the archive is still over the target,
applies the same lineage pruning and LRU eviction. GC has no current-fingerprint argument, so foreign-fingerprint
entries are ordinary LRU candidates there. Phase 1 has no refcount for an entry being restored; a file deleted
between lookup and restore makes that restore fall back to prefill.

### 11.3 Lineage caveat

A child is not a superset of the parent for **restore** purposes: the parent's KV/recurrent state is a snapshot at
an earlier position; the child cannot be rewound to the parent without a separate snapshot. Keeping only the child
is a disk/rollback trade-off:

- with `--slot-archive-versions 1`, pruning keeps the leaf and removes ancestors; smallest disk, no rollback to
  earlier points. Pruning is lazy: it runs only under budget pressure or in GC, so a parent and its continuation
  coexist while there is room;
- `--slot-archive-versions N`: keep up to N entries per lineage, pruned the same way;
- the `pinned` field protects a specific entry from GC, but phase 1 has no route that sets it; an admin pin route
  (for a pre-compact checkpoint or a branch point) is phase 2;
- for the branching use case, phase 2 adds explicit branch ids and multi-version management.

If rollback matters more than disk, raise `N`; do not imply that a longer state can replace a shorter one for time
travel.

### 11.4 Sizing

With q8 KV on the 35B-A3B layout, SLOT-SAVE measures about 10.9 GB per 1M tokens, so:

| state | approximate file |
|---|---:|
| 8k tokens | ~90 MB |
| 32k tokens | ~350 MB |
| 100k tokens | ~1.1 GB |
| 256k tokens | ~2.8 GB |
| 1M tokens | ~10.9 GB |

- 200 GiB ≈ 18M archived tokens, about 180 conversations of 100k tokens or 18 of 1M.
- The token history is 4 bytes/token (400 KB for 100k) and the recurrent state is a few MB per slot; both are small
  next to the KV.
- Lazy lineage pruning, `min-delta` and content-addressed dedupe keep the archive close to "one latest snapshot per
  conversation" once it is under pressure, rather than one entry per switch.

## 12. CLI and HTTP API

### 12.1 Flags

```
--slot-archive-dir DIR        enable the archive; created 0700 if missing
--slot-archive-gb N           logical budget in GiB (default 50; 1 GiB = 2^30 bytes)
--slot-archive-min-tokens N   do not archive states shorter than N (default 256)
--slot-archive-min-delta N    minimum growth since the last checkpoint (default 1024)
--slot-archive-versions N     keep N entries per lineage (default 1)
```

Phase 2: `--slot-archive-ttl`, `--slot-archive-per-key-gb`, `--slot-archive-on-shutdown`.

The archive and `--slot-save-path` must be different directories; the server refuses to start when they resolve
to the same directory. Shard names are server-controlled; no client-supplied filename reaches the archive.

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

  Optional `?limit=`, `?owner=` and `?fingerprint=` (hex) filter the list; no token contents and no file paths in
  the response.

- `DELETE /slots/archive/{id}` → `{"id": "...", "deleted": true, "freed_bytes": 77379688}`.
- `POST /slots/archive/gc` → `{"evicted": 3, "freed_bytes": 123}` (manual GC; see §11.2 for how it differs from
  the pressure path of a save).
- `POST /slots/{id}?action=checkpoint` → `{"id_slot": 0, "id": "...", "n_saved": 1058, "n_written": 77379688,
  "timings": {"save_ms": 83.7}}`. This is the explicit pre-compact/branch checkpoint; it uses the archive budget,
  not a client filename.
- The existing `POST /slots/{id}?action=save|restore|erase` keep their `--slot-save-path` semantics unchanged.

Route ordering matters: `GET /slots/archive` and `DELETE /slots/archive/...` must be matched before the generic
`POST /slots/{id}` parser, otherwise `archive` fails to parse as an id.

### 12.3 Metrics and logs

Prometheus text at `/metrics`, JSON at `/api/stats` (under `archive`):

- `strata_slot_archive_entries`, `strata_slot_archive_bytes`, `strata_slot_archive_budget_bytes`;
- `strata_slot_archive_hits_total`, `strata_slot_archive_misses_total`, `strata_slot_archive_restores_total`,
  `strata_slot_archive_evictions_total`, `strata_slot_archive_failures_total`.

Phase 1 has no save/restore duration histogram; the explicit checkpoint response reports `timings.save_ms`, and the
save/restore log lines report their own milliseconds.

Logs: a save line has `slot`, `reason`, `n`, `bytes` and `ms`; a restore line has `slot`, `n` and `ms`. Failures log
the reason but never the prompt text.

## 13. Single-slot and parallel

- **`--parallel N`**: `Sched` holds the archive handle. For every queued chat, before selecting a slot, look up an
  entry. Prefer an in-RAM match; else a free slot; else queue. The model thread performs the restore and prefill.
- **`--parallel 1` / direct `Runner::Gpu`**: one slot. `Runner::chat` checks the archive before
  `prefill_cached`; if the slot holds another conversation, checkpoint it, then restore the best entry if it fits.
  The server's model lock serializes this with other requests.
- **CPU / `Runner::Own`**: no GPU state to archive. The archive can still be configured and `GET /slots/archive`
  lists its metadata, but `POST /slots/{id}?action=checkpoint` answers `400` ("slot actions need a GPU model").
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
- `--slot-archive-gb` and `--slot-archive-min-tokens` bound disk use and the damage of a misbehaving client.
  TTL and per-key quotas are phase 2.
- Manifest and filenames are server-generated; no traversal surface. Deletion takes an entry id, never a path.
- Fingerprint checks keep a model/KV/YaRN/split change from restoring incompatible state.
- Backups: an archive is a copy of user conversations. Include it in retention/purge policy or exclude it
  deliberately, exactly as HOSTING says for explicit slot files.

## 15. Invariants

1. Never restore without an exact token-prefix match, the same fingerprint and the same owner.
2. Never touch a slot before the file's header, lengths and footer pass validation.
3. Never fail an HTTP request because the archive missed, broke, is full or is unmounted.
4. Never checkpoint a busy slot, a partial state (`pos != history.len()`) or an empty slot.
5. Never evict a pinned entry. Phase 1 has no reader refcount; if an entry disappears between lookup and restore,
   that restore falls back to prefill.
6. File and manifest updates are atomic (temp + fsync + rename); a crash leaves the old entry usable.
7. A failed restore resets the target slot (existing behavior) and falls back to prefill.
8. Automatic restore never uses `force` and never takes a slot that a request holds. A non-empty free slot is
   checkpointed before the restore overwrites it.
9. The archive is content-addressed; saving the same state twice does not duplicate the file.
10. The archive never changes the token stream or the greedy output; it only changes how much is prefilled.

## 16. Tests

### Unit (`src/slot_archive.rs`, 17 tests)

- manifest round-trip and relative paths; rebuild from orphan files; corrupt manifest moved aside and rebuilt;
- id stability/dedupe; `prepare_save` path and `min_tokens`; newest-used-first listing with a limit;
- longest exact-prefix lookup and a hash-only false candidate rejected;
- owner/fingerprint isolation; path escapes rejected; missing files dropped at open;
- budget eviction and an entry that cannot fit; lazy `versions` lineage pruning; delete; GC of missing files then
  LRU.

### GPU helpers and integration

- 9 pure `reclaim_plan` tests in `src/gpu_model.rs` (already fits, one victim, largest first, active excluded,
  cap-one excluded, target excluded, budget exhausted, exact boundary, no unified budget);
- `slot_save_as_records_explicit_owner`: `save_slot_file_as` writes the explicit owner to the header and saved
  state and restores the caller's request owner on success and failure.

### GPU regression

- `slot_` 4/4 and `unified_` 2/2, plus `prefill_cached_reuses_history` and `prefill_matches_reference`.

### E2E (`tools/test_slot_archive.sh`, 40/40 checks)

- single-slot divergence: chat A, switch to B, return to A; the divergence entry is listed and the continuation
  restores the archived prefix (`cached_tokens > 0`);
- explicit `POST /slots/0?action=checkpoint` response fields; `DELETE` removes an entry; manual GC;
- restart on the same archive directory keeps the entry and restores from disk;
- no `--slot-archive-dir`: `GET /slots/archive` and `action=checkpoint` answer `501`.

A parallel-slot E2E is being added separately and was not part of the 40 checks. The 40 checks do not set API keys,
so `403` for customer keys and cross-owner isolation are covered by the key guard and the unit tests, not by this
script. The budget/LRU behavior is covered by the unit tests, not by filling a small archive over HTTP.

### Performance

No new benchmark numbers are claimed here. The phase-2 evaluation still needs 32k/100k/1M checkpoint and restore
timings, model-thread stall, lookup cost at 1k/10k entries, manifest load, GC time, and confirmation that the normal
prefill path is unchanged when the archive is off.

## 17. Rollout

### Phase 0 — the design document

Done: matching rule, triggers, budget and owner semantics agreed.

### Phase 1 — MVP (done on `feat/slot-archive`)

- `--slot-archive-dir`, `--slot-archive-gb` (GiB), `--slot-archive-min-tokens`, `--slot-archive-min-delta`,
  `--slot-archive-versions`;
- manifest, startup load/rebuild, orphan adoption, LRU/budget GC;
- checkpoint on divergence, on automatic-restore eviction and on unified reclaim (via `kv_reclaim_plan`);
- automatic longest-prefix restore into a free slot, owner/fingerprint gates, fallback;
- `GET /slots/archive` (with `?limit=`, `?owner=`, `?fingerprint=`), `DELETE`, `POST action=checkpoint`, metrics;
- synchronous save/restore, full snapshot, no compression;
- unit/GPU/E2E tests, docs in USAGE/HOSTING, no behavior change without the flag.

### Phase 2 — operational polish

- host staging + writer thread; restore read on a worker;
- TTL, per-key quota, admin pin route;
- multi-version branching and explicit branch ids;
- park/LRU eviction checkpoints; optional shutdown checkpoint;
- prompt-cache interaction on restart; manifest journal/SQLite if needed;
- dashboards/alerts for bytes, hit rate, failures.

### Phase 3 — scale

- content-addressed segment store to avoid rewriting unchanged KV rows;
- optional compression of the token history (KV is high-entropy; expect little);
- shared archive across servers with the same fingerprint (network filesystem or object store);
- UI for browsing/deleting entries.

## 18. Open decisions

1. **Chosen: 50 GiB default.** 200 GiB is the large-server value; the flag is explicit either way.
2. **Chosen: never take a held slot and never use `force`.** A free slot's old conversation is checkpointed before
   the automatic restore overwrites it; under `--kv-unified`, a restore that would need a non-empty victim is
   skipped and the prompt prefills normally.
3. **Chosen: keep 1 entry per lineage by default, pruned lazily.** Raise `--slot-archive-versions` for more
   rollback points. An admin pin route is phase 2.
4. **Chosen: rewritten JSON manifest for phase 1.** Revisit an append-only journal or SQLite at 10k+ entries.
5. Still open: is a conversation id worth adding as an optional hint, even though matching is by tokens? It could
   improve listing and deletion UX, but is not needed for correctness.
6. **Chosen: no shutdown checkpoint in phase 1.** `--slot-archive-on-shutdown` is a phase-2 option, recommended on
   for the overnight workflow when it exists.
7. **Chosen: phase 1 skips image/M-RoPE archive.** The explicit save path already handles image slots; archive
   support needs an image key.
8. **Chosen: owner-scoped by default.** An admin-only "shared prefix" mode could improve hit rate for a common
   system prompt, at a privacy cost; it is still open.

## 19. Phase 1 acceptance

- With `--slot-archive-dir` and no client changes, a conversation that diverges (switch/compact) is checkpointed,
  and a later prompt that continues it is restored with only the suffix prefilled; `cached_tokens` reports the
  archived prefix.
- With `--kv-unified`, a restore grows the slot only when the victims are empty; the restored slot is pinned until
  admission and reused; a non-empty victim means the archive hit is skipped, not a `409`.
- The archive never exceeds `--slot-archive-gb` (within one entry and the 1 GiB reserve); LRU and lineage pruning
  are tested.
- A missing/corrupt/foreign/owner-mismatched entry falls back to a normal prefill and the request succeeds.
- A checkpoint records the conversation owner, even when triggered by another key.
- `GET /slots/archive` and `DELETE` work; customer keys get `403`; metrics expose bytes, entries, hits and
  failures.
- Restart keeps the manifest and entries usable; a missing or corrupt manifest is rebuilt from the shard files.
- Without `--slot-archive-dir`, the server behaves as before (the archive routes answer `501`).
- The slot and prefill regressions listed in §16 stay green, and the server behaves as before without
  `--slot-archive-dir`.

## 20. Compatibility with SLOT-SAVE

- Archive entries are ordinary `STRATASLOT` v1 files. No format change is required for the MVP.
- The explicit `--slot-save-path` routes, filename validation and response fields are untouched.
- An operator can copy an archive entry into the explicit save directory and restore it by filename, or use the
  archive API; the fingerprint rules are the same.
- A future `STRATASLOT` v2 (checksum, compression) can be introduced behind the same manifest/version field.
