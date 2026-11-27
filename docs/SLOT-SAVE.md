# Slot save/restore

Explicit, per-slot persistence of the full inference state to disk, with a llama.cpp-compatible HTTP
surface. A slot file can be written while the server is idle, moved, and restored into any compatible slot
of a fresh process, so a long conversation does not have to be prefilled again.

This document is the implementation contract for the `feat/slot-save` branch (issue #8). It covers the
state that is saved, the `STRATASLOT` v1 file format, the save/restore algorithm, the HTTP API, the
scheduler behaviour, unified-KV safety, security, limits, and the test matrix.

**Status: implemented.** The branch carries the format module (`src/slot_file.rs`), the per-slot token
history, GPU streaming save/restore/erase (`src/gpu_model/slot.rs`), unified prefix reuse, the HTTP
control plane and `--slot-save-path` in `src/main.rs`, the unit/GPU tests and the E2E script
`tools/test_slot_save.sh`. The sections below describe the code as implemented on the branch, not a plan;
where an earlier design assumed a different shape, the text follows the code.

The automatic, budgeted layer above these explicit files is designed in
[SLOT-ARCHIVE.md](SLOT-ARCHIVE.md).

## 1. Goals and non-goals

### Goals

- Save a slot's **complete** state: prompt and generated tokens, KV rows, DeltaNet conv/scan state, MTP
  draft state, PLE/QSA rings, owner, and RoPE offset.
- Restore it into a slot in the same process or in a fresh process, and continue with the same greedy
  output as an uninterrupted run.
- Stream to and from disk in bounded chunks: a 1M-token q8 slot is about 10.9 GB and must not be held in
  host RAM.
- Match llama.cpp's slot HTTP API (`--slot-save-path`, `GET /slots`,
  `POST /slots/{id}?action=save|restore|erase`) and its response fields.
- Work with `--parallel N` and with `--kv-unified`, including safe growth and reuse of a restored slot.
- Reject incompatible files (model, KV type, YaRN, split) before changing a slot.

### Non-goals

- Binary compatibility with llama.cpp `.slot` files. The KV layout, quantization, and model identity
  differ. Only the HTTP API is compatible.
- Automatic/background snapshots. Save and restore are explicit admin actions.
- Encryption at rest. Slot files contain plaintext KV, i.e. prompt and reply content.
- Saving a slot mid-generation. A save/restore/erase for a busy slot is deferred until the slot is idle,
  as llama.cpp does.
- Image prompts, DFlash, and external draft models in the first version. They are rejected or documented
  as unsupported; text MTP, PLE, QSA, and DeltaNet are covered.

## 2. State model

A slot is the `GpuModel` view behind one `--parallel` id. The persistent state is split in two groups.

### Recurrent/auxiliary buffers (`GpuModel::slot()`)

- per recurrent layer: conv state and DeltaNet scan state (`Mixer::Delta { conv, s }`);
- the MTP draft head's hidden row (`saved_x`; `cur_g` for EAGLE3);
- the PLE ring and the last two PLE token ids (`saved_ple_prev`);
- the QSA ring (`qsa.saved`).

These are copied by `save_state()` into the per-slot `saved` buffers and restored by `restore_state()`.

### Attention KV (`GpuModel::kv_bufs()`)

Per attention layer: K and V rows for the current slot, plus the QSA pool and any DFlash layer KV. The
buffer length divided by its row count is the bytes per token; rows are positions from 0 to `pos`.

### Token history

The full history lives in `GpuModel::history` for the current slot and in `Seq::history` for the other
slots; `use_slot` swaps the two. It is one token per KV row in use, prompt and generated tokens
included. `GpuModel::ids` remains the backend's small token-id buffer and is not the history.

Update rules:

| path | update |
|---|---|
| `prefill_cached`/`prefill_chunk` | appends the processed chunk tokens, then sets the history to the reused prefix plus that chunk; at the end `history.len() == pos` |
| `prefill_slots_run` | sets each job slot's history to its full prompt |
| `step`, `step_embed`, `step_rows` | appends the fed token after a successful forward |
| `verify_run` | appends all fed tokens; `rollback` truncates to `start + keep` |
| `prefill_image` | appends `p.ids[i]`; keeps `rope_offset` |
| `unpark`, `unstash`, `restore_slot_file` | sets the history from the saved/stashed token list |
| `reset`, `drop_slot_cache` | clears the history |
| `save_slot_file` | requires `history.len() == pos`; otherwise refuses with a clear error |

`slot_saved(s)` still returns `saved_ids` for prompt-cache admission. `slot_tokens(s)` returns the full
history and backs `GET /slots` and `n_saved`/`n_restored`/`n_erased`.

### What is not stored

- The n-gram/lookup drafter cache. It is rebuilt from the restored token list on demand; it does not change
  greedy output.
- The shortlist/adaptation counters. They are performance hints and reset to their defaults.
- Expert placement. It is stored in the existing expert profile, not in the slot file; a restored slot may
  swap experts on its first tokens.
- Sampling state (temperature, RNG). Sampling is per request, not per slot.

## 3. `STRATASLOT` v1 file format

All integers are little-endian. The format is designed for streaming: every buffer is framed by its length,
so a writer never needs the whole state in RAM.

### Header

| offset | size | field |
|---:|---:|---|
| 0 | 8 | magic `STRATASL` |
| 8 | 4 | version `1` |
| 12 | 4 | flags |
| 16 | 8 | fingerprint |
| 24 | 8 | `n_tokens` |
| 32 | 4 | `rope_offset` (i32) |
| 36 | 4 | reserved (0) |
| 40 | 8 | `owner` (API key id) |
| 48 | 4 | `n_parts` |
| 52 | 4 | reserved (0) |
| 56 | `4 * n_tokens` | token ids |

Flags:

- bit 0: image/M-RoPE slot (`rope_offset != 0`);
- bit 1: saved with `--kv-unified` (informational);
- bit 2: MTP/PLE/QSA buffers present (informational; the counts are authoritative);
- bits 3..31: reserved, must be 0 in v1.

### Parts

For each part, in model order (the root `GpuModel`, then `next`):

```
i64 ple_prev[2]          // the PLE n-gram state before the saved position
u32 n_slot
for each slot buffer:
    u64 len
    u8  len bytes
u32 n_kv
for each KV buffer:
    u64 row_bytes
    u8  row_bytes * n_tokens bytes
```

`n_slot` must equal the current `GpuModel::slot().len()`. Each length must equal the current buffer length.
For KV, `row_bytes` must equal the current buffer length divided by its row count, and `n_tokens` must not
exceed the row count. Any mismatch is a format/config error and is rejected before writing a slot.

### Footer

| size | field |
|---:|---|
| 8 | magic `STRATEND` |
| 8 | payload length: bytes from offset 0 to the footer |

A restore checks that the payload length matches the bytes consumed; this catches truncation and trailing
garbage. A checksum is not required by v1; the framing and the exact buffer lengths are enough to detect
the common failure modes. A future version can add a checksum without breaking v1 readers.

### Fingerprint

A 64-bit hash of everything that makes two states incompatible:

- model config: arch, layers, embedding, vocab, heads, head dim, RoPE fields, expert fields, SSM fields,
  PLE/QSA/hyper-connection fields, DFlash presence;
- the full GGUF tensor table: name, dims, type, offset, shard (sorted; `BTreeMap` order);
- the effective KV layout per part: number of parts, per-part layer range, `kv_words`/`kv_words_v`, and
  each KV buffer's bytes per token;
- the auxiliary buffer layout: `slot()` lengths per part;
- YaRN/linear parameters from `RopeParams` (`base`, `freq_scale`, `ext_factor`, `corr`, `n_rot`);
- the M-RoPE sections, when present.

The current `prompt_cache_fingerprint` hashes only part of this and includes the current KV capacity, which
changes under `--kv-unified`. Slot files must use the new stable fingerprint. A file with a different
fingerprint is rejected with `400` and the slot is not touched.

## 4. Save algorithm

```
save_slot_file(path):
    require pos == history.len()
    require the slot is not held/prefilling (enforced by the scheduler)
    save_state()                       # copies recurrent state into `saved`
    write header (fingerprint, n_tokens=history.len(), rope_offset, owner, n_parts)
    write history
    for each part:
        write slot buffer count and each saved buffer in 64 MB chunks
        write KV buffer count, row_bytes and rows 0..n_tokens in 64 MB chunks
    write footer
    flush, fsync, rename temp -> path
```

- The temp file is created in the destination directory with `create_new` and a suffix made of the PID and
  a nanosecond timestamp; the target is replaced atomically by `rename`.
- On Unix the file mode is `0600`.
- A failed write removes the temp file and leaves the previous target intact.
- The model thread is blocked for the duration. A 1M q8 slot is ~10.9 GB: roughly 1 s of PCIe transfer
  and 10-20 s of disk at 0.5-1 GB/s. There is no progress log; the admin request returns when the stream
  and the rename are done.

After a successful save the slot's `saved_ids` is set to the full history, so a later request that
continues the same conversation can reuse the whole prefix, not only the original prompt prefix.

## 5. Restore algorithm

```
restore_slot_file(path, allow_evict):
    read and validate header, tokens, fingerprint
    require n_tokens <= ctx
    require the target slot is idle
    if unified and n_tokens > kv_cap[slot]:
        compute the victims kv_grow_unified would reclaim
        if any victim is non-empty and !allow_evict: error 409
    pin the target slot (scheduler)
    kv_need(slot, n_tokens)
    save_state()                       # allocate `saved` if needed
    for each part:
        read slot buffers into `saved`
        read KV rows into the slot's KV buffers
    saved_pos = n_tokens
    saved_ids = tokens
    history = tokens
    saved_owner = owner
    rope_offset = header.rope_offset
    restore_state()                    # recurrent buffers -> current, pos = n_tokens
    write footer check
```

- The header and all lengths are validated before any GPU write. A read error after the writes start
  clears the target slot (`reset`/`drop_slot_cache`) and returns `400`; a partially restored slot is never
  left usable.
- Non-unified KV grows normally up to `--ctx`; a too-large file is rejected by `n_tokens > ctx`.
- Unified KV: see section 7.

## 6. HTTP API

The routes and fields mirror llama.cpp's server. They are admin-only, like `/api/load`.

### `GET /slots`

Returns a JSON object with a `slots` array:

```json
{
  "slots": [
    {"id": 0, "is_processing": false, "n_ctx": 262144, "n_past": 1234, "pinned": false},
    {"id": 1, "is_processing": true,  "n_ctx": 262144, "n_past": 42,   "pinned": false}
  ]
}
```

`n_past` is `history.len()`; `n_ctx` is the slot's current KV capacity (under `--kv-unified` it grows as
the slot is used); `pinned` is true for a slot whose restored state must not be reclaimed. On a server
without `--parallel` there is one slot and `is_processing` is always `false`.

### `POST /slots/{id}?action=save`

Body: `{"filename": "slot1.bin"}`

```json
{
  "id_slot": 1,
  "filename": "slot1.bin",
  "n_saved": 84,
  "n_written": 12345678,
  "timings": {"save_ms": 123.4}
}
```

### `POST /slots/{id}?action=restore`

Body: `{"filename": "slot1.bin"}`

```json
{
  "id_slot": 0,
  "filename": "slot1.bin",
  "n_restored": 84,
  "n_read": 12345678,
  "timings": {"restore_ms": 123.4}
}
```

### `POST /slots/{id}?action=erase`

```json
{"id_slot": 1, "n_erased": 84}
```

### Errors

- no `--slot-save-path`: `501` with
  `This server does not support slots action. Start it with \`--slot-save-path\``;
- no model loaded: `503 "no model loaded"`;
- invalid slot id: `400 "Invalid slot ID"`;
- invalid filename: `400 "Invalid filename: ..."`;
- save of an empty slot: `400 "slot N has no token history to save"`;
- wrong fingerprint / truncated file / bad footer / wrong part or buffer length: `400` with a specific
  message, before the target slot is left usable;
- unified restore that would evict a non-empty slot: `409` with a message containing `evict`, unless
  `?force=1`;
- other I/O and read/write errors: `400` with the OS error (the route maps every error without `evict`
  to `400`).
- A busy slot is **deferred**, not rejected: the request waits until the slot is idle, as llama.cpp does.

`n_written` and `n_read` are total file bytes; `n_saved`, `n_restored` and `n_erased` are token counts.

### `--slot-save-path`

- New `serve` option: a directory that must exist and be a directory at startup.
- `filename` is a single path component, at most 255 bytes, valid UTF-8, no control characters, no `/`,
  `\`, `.` or `..`, no NUL. Subdirectories are not allowed in v1.
- The final path is `slot_save_path.join(filename)`. The temp file lives in the same directory.

## 7. Unified KV

`--kv-unified` makes `--ctx` a shared token budget. Slots start at `ctx / parallel` and grow by reclaiming
inactive slots down to one token. The existing `drop_slot_cache` clears a victim's `pos`, `saved`,
`saved_ids`, `sys`, and owners, so a reclaimed slot cannot silently reuse another slot's rows.

### Save

Straightforward: the current slot's rows 0..`pos` are streamed. The file records the mode in flags bit 1,
but the format is identical; a file can be restored into a non-unified server if `n_tokens <= ctx` and the
fingerprint matches.

### Restore and growth

1. The scheduler pins the target slot before growing it. `Sched::active()` includes pinned slots, so
   `kv_grow_unified` will not choose the target as a victim.
2. If `n_tokens` fits in the current capacity, no growth and no victims.
3. Otherwise `kv_need` grows exactly as for a prompt. Compute the victims it would reclaim (inactive,
   non-pinned slots sorted by capacity, largest first). If any victim has `pos > 1` or a non-empty
   `history`/`saved_ids`, refuse with `409` unless the request has `force=1`. With `force=1` the victims
   are erased as part of the growth.
4. After a successful restore the slot stays pinned until it is used by a request or erased. Pinning
   prevents a later prompt in another slot from reclaiming the restored state while it is idle.
5. If a pinned slot is somehow reclaimed (should not happen), `drop_slot_cache` clears its state and the
   scheduler clears the pin; the next request prefills from zero.

### Reuse after restore

`prefill_cached_with` calls `cache_begin_unified` when the server runs with `--kv-unified`:

```
hit = saved_ids.len()
if hit > 0 && hit <= keep && saved_owner == owner
   && tokens[..keep].starts_with(&saved_ids)
   && the slot still has its KV rows:
       restore_state()        # recurrent state, pos = hit
       reused = hit
else
       reset()                # no parking in host RAM
       reused = 0
```

There is no `park`/`unstash` in unified mode: a diverging prompt discards the old state instead of copying
a possibly 10 GB KV to host RAM. The existing `saved` buffers are enough for the recurrent state.

After prefilling the reusable prefix, `prefill_cached_with` saves the state as in the non-unified path
(`save_state`, `saved_ids = tokens[..keep]`, `saved_owner = owner`), so subsequent requests can reuse it.
The slot's KV rows 0..hit are valid because:

- growth with `keep=true` preserves the old rows;
- a victim's `drop_slot_cache` clears `saved_ids`, `history` and the pin, so it cannot be reused;
- `reset` clears the state.

This also enables ordinary prompt reuse under `--kv-unified`, which the first unified version deliberately
disabled. The tests cover both the restored-state path and the normal continuation path.

## 8. Scheduler and control plane

### `--parallel N`

The model thread owns the slots. The request threads send:

```
Job::Slot { action: SlotAction::Save { slot, path, owner } | Restore { slot, path, force } | Erase { slot }, back }
Job::Slots { back }
```

`Sched::take` queues them as `Control::Action(action, back)` or `Control::Info(back)` in a
`VecDeque<Control>`; `Sched` also keeps `pinned: Vec<bool>`.

- `Sched::active()` = held || prefill || pinned.
- A save/restore/erase for a slot with `held[slot] || prefill[slot]` is moved to the back of the queue
  and retried after the next forward; the HTTP thread waits.
- Info is answered immediately.
- Save/restore/erase run between forwards, with the model thread blocked.
- The target slot is validated (`0..n`). A restore sets `pinned[slot] = true` before growing the slot and
  clears it if the restore fails; `erase` clears it. On admission of a chat to that slot the scheduler
  clears the pin, since the request now owns the state.
- `GET /slots` is answered on the model thread so `is_processing`, `pinned`, `n_ctx` and `n_past` are
  consistent with the slots.

### `--parallel 1` / no slots

`handle` owns `Runner::Gpu(g)` directly. The request is serialized with chats by the server's model lock;
save/restore/erase call the `GpuModel` methods directly. `GET /slots` reports the single slot 0 with
`is_processing: false` and `pinned: false`.

## 9. Security and privacy

- Slot endpoints are admin-only. Customer keys keep the existing `/v1/...` restrictions.
- Filename validation prevents path traversal, absolute paths, subdirectories, and symlink tricks; the
  temp file is created with `create_new` in the destination directory and renamed over the target.
- Files are created `0600` on Unix. They contain plaintext KV: prompt, system prompt, tools, and replies.
  Treat the directory as sensitive; do not expose it over HTTP.
- Stored `owner` is the API key id. A restored state is reused only by the same owner; another key's prompt
  will not match `saved_owner` and will reset. The admin can still copy files between slots deliberately.
- A corrupt restore clears the target slot rather than leaving a half-written state.

## 10. Tests

### Unit / format (`src/slot_file.rs`, 13 tests)

- header/tokens/footer round-trip, including an empty token list;
- wrong magic, version, flag, footer magic;
- footer payload-length mismatch and truncated footer;
- truncated token list and `n_tokens` over the rows before allocating;
- `n_tokens` mismatch between header and token list;
- filename matrix: empty, `.`, `..`, `../x`, `/abs`, `a/b`, control characters, >255 bytes, and accepted
  plain names.

### GPU round-trip

- `slot_save_restore_round_trip`: prefill + generate, save, erase, restore, continue; the continuation's
  next-token logits match an uninterrupted run (`|dlogit| 0.000e0`).
- `slot_history_tracks_prefill_step_verify_and_rollback`: the history covers prefill, `step`, `verify`
  and `rollback`, and stays equal to `pos`.
- `unified_cache_reuses_the_saved_prefix`: under `--kv-unified` a continuation reuses the saved prefix.
- The round-trip covers prompt + generated tokens, KV rows, DeltaNet/MTP/PLE state and `q8` KV.

### Cross-slot and restart (E2E)

- save slot 1, restore into slot 0, continue there (`tools/test_slot_save.sh`, `full`);
- stop the server, start a fresh process with the same model, restore the file and continue (`restart`);
- `truncated.bin` and `corrupt.bin` are rejected with `400` while the server stays healthy.

### Unified

- `unified_kv_budget_grows_one_slot` stays green;
- E2E `unified`: a slot grown past `ctx / parallel`, restored into another slot without `force` → `409`;
  with `force=1` → `200` and `n_restored == n_saved`; the restored slot reports `pinned: true`;
- `unified_cache_reuses_the_saved_prefix` covers the next-request prefix reuse;
- `drop_slot_cache` invalidates a reclaimed slot's history.

### API (E2E `tools/test_slot_save.sh`)

- `501` without `--slot-save-path`;
- `400` for bad slot, filename, missing/truncated/corrupt file and path traversal;
- save/restore/erase response fields and timings, including `n_saved`/`n_written` and
  `n_restored`/`n_read`;
- a busy slot defers the save and then succeeds (`busy`);
- admin-only: a customer key gets `403` (checked by the route guard; the E2E script runs as admin).

### Regression

- 179 lib tests;
- `prefill_matches_reference`;
- `unified_kv_budget_grows_one_slot`;
- the E2E script's 21 `full` checks, 21 restart/busy/unified checks and the `501` check.

## 11. Rollout

The branch lands the feature in reviewable commits:

1. `35e5a8d` — this design document.
2. `5be4ad4` — the `STRATASLOT` v1 format module and its 13 unit tests.
3. `cccf98e` — token-history tracking and `--kv-unified` prefix reuse.
4. `36beb76` — streaming GPU save/restore/erase.
5. `af43fe1` — the history, unified-reuse and round-trip tests.
6. `1677ba1` — the HTTP routes, scheduler control jobs, pinning and `--slot-save-path`.

The feature is opt-in: without `--slot-save-path` the server behaves exactly as before. Update a
production command only when the operator wants explicit slot files and has provisioned the directory.

## 12. Open decisions

- Whether to add a checksum in v1.1 (current plan: framing + exact lengths).
- Whether to support image/DFlash/external-draft slots in v2.
- Whether to expose saved-file listing and deletion over HTTP (`GET /slots/files`,
  `DELETE /slots/files/{name}`) in a follow-up.
- Whether to add a web-UI button; not part of the first PR.
