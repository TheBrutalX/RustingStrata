# Parallel speculative decoding (MTP and K4v)

**Status: design / first implementation branch.** Today `--parallel > 1` disables both MTP (`--mtp`) and the ngram
drafter (`--spec-type ngram-map-k4v`): `resolve_flags` forces the draft count to 0, and `Engine::load` only attaches
`NgramDrafter` when `parallel <= 1`. Batched decode (`step_rows`) has no path that verifies draft tokens for several
slots.

This document defines how to make them work with several slots without changing the single-slot behavior.

## 1. Why the current code cannot do it

- `Sched` gives each request one `Job::Step { slot, token, logits }` per decode step; `step_rows` feeds exactly one
  token per slot and returns one greedy token per row. There is no per-slot draft list or acceptance count.
- The sampler (`Sampler`, RNG, temperature/penalties) lives in the request thread. The model thread only returns
  logits/greedy tokens. MTP/ngram need the sampler for speculative sampling.
- `GpuModel.ngram` is one drafter for the current slot; `use_slot` does not swap it. MTP state (`mtp.saved_x`) is
  per slot but the draft/verify path is single-sequence.
- `spec_step`/`ngram_spec_step` run on one sequence with `verify`; `step_rows` rejects MTP and has no draft verify.

## 2. Goals and non-goals

### Goals

- `--parallel N` with `--spec-type ngram-map-k4v` produces the same tokens as the equivalent non-speculative run
  for greedy and sampled requests (up to RNG; speculative acceptance must be distribution-preserving).
- MTP works with `--parallel N` on models with an MTP head.
- No regression for `--parallel 1`; a feature flag can disable the parallel path.
- Drafts still accelerate a single long agent conversation on a multi-slot server.

### Non-goals (first phase)

- Perfect batching of draft verification. The first phase may verify one slot at a time; a later phase adds
  `verify_rows`.
- DFlash/EAGLE3/external draft models in parallel (they have their own restrictions).
- Cross-slot draft sharing.

## 3. Architecture options

### A. Request-thread drafter + model-thread verification (K4v first)

The ngram drafter is CPU-side and only needs the token history, which the request thread already has. Keep the
sampler in the request thread:

1. The request thread owns a per-request `NgramDrafter` (or per-slot drafter in the scheduler), initialized from
   the prompt and updated with accepted tokens.
2. Before a decode step it asks for `drafts = drafter.draft_with_order(sampled, k)`.
3. It sends `Job::Verify { slot, tokens: [next, drafts...], logits: true }` to the model thread.
4. The model thread runs `g.use_slot(slot); g.verify(tokens)` (or a new `verify_rows` for several slots) and
   returns logits for each position.
5. The request thread samples each position with its `Sampler`, accepts the draft prefix that matches, and keeps
   the first rejection sample. This is standard speculative sampling; all randomness stays in the request thread.

Advantages: sampler/RNG untouched, per-request drafter state, K4v can land without touching MTP. Disadvantage:
a verify for one slot is not batched with other slots' normal steps; phase 2 adds `verify_rows`.

### B. Model-thread draft head (MTP)

MTP drafts depend on the hidden state of the target model, so the model thread must produce them:

1. The model thread keeps per-slot MTP state (`saved_x`/draft head) in `Seq`/`GpuModel`.
2. `Job::Draft { slot, token, k }` returns draft tokens (or a `Job::Verify` that first drafts internally).
3. The request thread verifies/samples as in A, or the model thread verifies and returns accepted tokens plus logits.

The sampler still has to be involved; the cleanest split is: model thread drafts and verifies, request thread samples
and accepts. This needs a per-slot sampler state or a callback; design phase 2.

## 4. First milestone (this branch)

### 4.1 Per-slot ngram state

- Add `ngrams: Vec<Option<NgramDrafter>>` to the model (or to `Seq`) and replace direct `self.ngram` use with an
  accessor for the current slot. `use_slot` selects the matching drafter.
- `Engine::load` creates one drafter per slot when spec is enabled and `parallel > 1`; keep the single-slot path
  unchanged.
- Memory: one ngram table per active request/slot; bound it with the existing config and only create drafters for
  slots that actually use ngram.

### 4.2 Verification job

- Add `Job::Verify { slot, tokens: Vec<u32>, logits: bool, back }`.
- `Sched::run`: process queued verifies before/alongside normal `Step`s. Phase 1 may run one verify on the model
  thread (`g.use_slot(slot); g.verify(&tokens)`) while the other slots wait; it must not interleave with a batch
  `step_rows` on the same slots.
- Return logits for every verified token and the number of accepted draft positions, or let the request thread do
  the acceptance from logits.
- Keep the slot's history/pos updated exactly as `step`/`verify` do; a rejected suffix is rolled back.

### 4.3 Request thread

- On a `Runner::Slot` in `decode`, if spec is enabled and there is a drafter:
  - generate drafts from the current sampled token,
  - send `Verify`,
  - sample/accept on the request thread,
  - emit all accepted tokens and continue.
- If the draft list is empty, fall back to the existing one-token `Step`.
- Preserve stop sequences, thinking budget, tool-call holdback and usage accounting across the accepted tokens.

### 4.4 Feature flag and limits

- `--parallel-spec` (or reuse `--spec-type`) only enables the parallel path when the model/request supports it.
- If the drafter is unavailable, log once and fall back to the normal batch path.
- Do not enable MTP in this milestone; MTP parallel is phase 2.

## 5. Correctness invariants

- A verified draft suffix that is not accepted must be rolled back from KV, recurrent state, history and ngram
  state; otherwise later tokens diverge.
- The slot must be held/pinned during verify; no other request can use it.
- The request thread's sampler is the only source of randomness; the model thread must not sample.
- Greedy verification can accept the longest matching prefix; sampled verification must use the speculative
  accept/reject rule with the target distribution.
- Token counts (`usage`, `cached_tokens`, decode tok/s) count accepted tokens, not drafted ones.
- Drafted tokens never change the final token stream relative to a non-speculative run (greedy exactly; sampled in
  distribution).

## 6. Batching phase

After K4v works serially:

- Add `verify_rows(rows: &[(slot, Vec<u32>, logits)])` that feeds multiple draft lists through one forward (or
  several rows per slot) and returns per-position logits.
- Batch compatible slots; keep a per-slot draft length and rollback.
- Compare decode throughput on 1/2/3 slots.

## 7. MTP phase

- Enable the MTP head when `parallel > 1` (`resolve_flags` stops forcing 0), with per-slot `saved_x`/draft state.
- Model thread generates `k` drafts from the hidden row for the slot; request thread verifies/samples.
- EAGLE3/DFlash remain single-slot until their cross-layer feature requirements are batched.

## 8. Tests

### Unit / CPU

- Ngram drafter per slot: independent histories, draft/accept/reset, no cross-slot state.
- Speculative accept/reject sampler: greedy longest prefix, sampled distribution, rejection at first mismatch.
- Rollback of KV/history/ngram after a rejected suffix.

### GPU

- `verify` for a slot equals sequential `step` logits.
- K4v parallel: token stream equals a non-speculative `--parallel 3` run for the same prompts; accepted drafts > 0.
- Three concurrent conversations with different drafters do not share state.
- A failed verify/rollback leaves the slot usable.

### E2E / benchmark

- `--parallel 3 --spec-type ngram-map-k4v`: 3 concurrent agent prompts; compare decode tok/s and output with/without
  spec; verify `drafts made/accepted` stats.
- MTP parallel after phase 2; 1/2/3 slots.

## 9. Risks

- The request thread/verify round trip can be slower than batching; measure before enabling by default.
- Ngram tables per slot use host memory; cap them or share read-only tables with per-slot cursors.
- Rollback is subtle for DeltaNet/MTP/PLE/QSA; reuse the existing `rollback` path, test it.
- Sampling acceptance must not bias output; use the standard rule.
- The saving/restoring archive paths must not run while a slot has pending drafts.

## 10. Acceptance

- K4v works with `parallel 3`, produces the same tokens as non-spec, and improves decode tok/s for a
  draft-friendly conversation.
- MTP works with `parallel 3` after phase 2, with per-slot state.
- `parallel 1` behavior and existing tests unchanged.
