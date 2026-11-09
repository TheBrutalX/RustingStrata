# V100 results

Machine: Tesla V100-SXM2-32GB, driver 580.178.04, Vulkan 1.4.312, CachyOS.
`strata` v1.2.0 + the `gemmi_dp` shared-memory fix (branch `v100`).

## 2026-11-06 — Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf, Vulkan

```
plan: weights+state+scratch 2580 MB (KV 642 MB for 32896 tokens, prefill 512 MB),
      expert slots 10040 MB, resident experts 10240/10240
pp    512: prefill  635.6 (361.6-774.2) tok/s | tg 128: decode 109.8 (82.7-123.4) tok/s
pp   4096: prefill 1423.9 (1422.0-1427.0) tok/s | tg 128: decode 108.6 (91.1-117.7) tok/s
pp  32768: prefill  523.5 (522.1-525.1) tok/s | tg 128: decode  95.7 (95.4-96.1) tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
```

Note: the first repetition of `pp 512` includes Vulkan pipeline compilation;
the steady-state value is ~775 tok/s.

## 2026-11-06 — Qwen3.8-27B-UD-Q4_K_XL.gguf, Vulkan (dense, no tensor cores)

```
plan: weights+state+scratch 17118 MB (KV 512 MB for 8192 tokens, prefill 706 MB)
pp    512: prefill 317.9 (315.8-319.4) tok/s | tg 128: decode 27.6 (27.5-27.7) tok/s
pp   4096: prefill 291.1 (290.5-291.6) tok/s | tg 128: decode 26.6 (26.1-26.9) tok/s
```

## llama.cpp reference (same card)

Measured with `llama-server` (`/completion`, `cache_prompt: false`, temp 0):

| Model | prompt | prefill | decode |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B Q2_K_XL | 416 tok | 496 tok/s | 101.9 tok/s |
| Qwen3.6-35B-A3B Q2_K_XL | 1261 tok | 1243 tok/s | 101.9 tok/s |
| Qwen3.6-35B-A3B Q2_K_XL | 2941 tok | 1352 tok/s | 99.3 tok/s |
| Qwen3.8-27B Q4 (production logs) | 1092 / 4381 tok | 292 / 491 tok/s | ~49-53 tok/s |

## Notes

- Before the fix, any chunked prefill on Vulkan ended with
  `Xid 13 SKEDCHECK18_L1_CONFIG_TOO_SMALL` and a lost device.
- sm_80 `mma` and `cp.async` do not exist on sm_70: the CUDA backend requires
  the port described in `docs/ROADMAP.md`.

## 2026-11-07 — M0-T4 shared-memory preflight validation (Vulkan, V100)

```
strata check --backend vulkan: maxComputeSharedMemorySize 49152
  60 (kernel, spec) pairs checked, largest gemmi_dp = 49152 (at the limit, accepted)
  no cooperative-matrix kernels (gemmi, gemmi_s, gemm_f, gemm_fs) on this device
  cost 1.1-4.8 ms

STRATA_VK_SMEM_LIMIT=40000: exit 1
  vulkan: kernel gemmi_dp with specialization [[0, 13, 176, 176947200, 0]]
  needs 40960 bytes of shared memory; this device allows 40000
  (raised before any weight upload)

bench --pp 512  --tg 128 --ctx 32768: prefill 761.6, decode 121.4 tok/s
bench --pp 4096 --tg 128 --ctx 32896 (x3): prefill 1401-1404, decode 115.5-116.2 tok/s
same-window v100 baseline: prefill 1398-1407, decode 113.9-115.5 tok/s -> no regression
no new Xid; device usable after the forced-oversize run
```

Note: the 1423.9 / 108.6 figures earlier in this file come from a different
session; compare within the same window.

## 2026-11-07 — M0-T5 GGUF type matrix (Vulkan, V100, sm_70)

Commit: `1b19139` | driver 580.178.04 | Vulkan 1.4.312 | maxComputeSharedMemorySize 49152
Command: `STRATA_TEST_BACKENDS=vulkan cargo test --release --test kernels vulkan_type_matrix -- --nocapture --exact --test-threads=1`

The test builds one synthetic, finite block per `GgmlType` with the scale fields at the
offsets each dequantizer reads, uploads raw bytes (bypassing `Backend::upload`, so native
kernels run) and compares dequant, GEMV, GEMV-N, GEMV-Q (bit-equal to GEMV-N on the same
`quant_xq`), MoE GEMV (first expert, `-1` skipped slot, last expert), dense GEMM over 130
positions and the tiled `gemm` with 3 tiles against the CPU reference over the same bytes.
`strata check` and `Vk::max_shared_bytes` use the committed SPIR-V accounting.

| Type | Block elems | Upload path (loader) | dequant | gemv | gemvn | gemvq | moe_gemv | gemm_dense | gemm (tiled) | max shared (B) | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F32 | 1 | native | pass | pass | pass | pass | pass | pass | pass | 27664 | pass | f16 GEMM inputs truncate (Volta `OpFConvert`); reference matches |
| F16 | 1 | native F32 (`upload_type`) | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported | 49152 | unsupported | no F16 shader path; loader uploads it as F32 |
| Q4_0 | 32 | native Q8_0 (`upload_type`) | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | raw native kernels tested; loader uploads Q8_0 |
| Q4_1 | 32 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| Q5_0 | 32 | native Q8_0 (`upload_type`) | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | raw native kernels tested; loader uploads Q8_0 |
| Q5_1 | 32 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| Q8_0 | 32 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| Q2_K | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | |
| Q3_K | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | |
| Q4_K | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| Q5_K | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| Q6_K | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | |
| IQ2_XXS | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| IQ2_XS | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | f16 GEMM inputs truncate (Volta `OpFConvert`); reference matches |
| IQ3_XXS | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| IQ1_S | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| IQ4_NL | 32 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | |
| IQ3_S | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| IQ2_S | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | f16 GEMM inputs truncate (Volta `OpFConvert`); reference matches |
| IQ4_XS | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| IQ1_M | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | |
| BF16 | 1 | native | pass | pass | pass | pass | pass | pass | pass | 27664 | pass | f16 GEMM inputs truncate (Volta `OpFConvert`); reference matches |
| Q2_0 | 64 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |

Summary: 22/23 types pass, 1 documented unsupported (F16, no shader path; the loader
uploads it as F32), 0 fail. Largest shared memory 49152 B (`gemmi_dp`, at the 49152 B
limit and accepted) — the `COOP = 0` table path of `53f823e` still fits. No Xid.

Note: on this driver the Vulkan f16 GEMM path (`packFloat2x16`, SPIR-V `OpFConvert`)
truncates its f16 inputs toward zero, while CUDA's `__float2half` rounds to nearest
even. The GEMM reference uses the conversion the backend performs, so the matrix measures
the kernel. The two pre-existing Vulkan tests that assumed round-to-nearest
(`gemv_partial_superblocks_and_q2_0`, `gemm_matches_reference_for_every_type`, plus
`f16_weight_uploads_as_f32`) needed the same backend-aware reference; no shader or
numerics changed. Residual risk: the Vulkan f16 GEMM differs from CUDA by up to ~6e-4
relative on these synthetic inputs.

Attention: type-independent (f32 after dequant). Covered by
`attn_prefill_matches_decode` and `strata generate` on both models.

### Checks

```
strata check --model Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --backend vulkan
  60 (kernel, specialization) pairs checked, largest 49152 bytes (gemmi_dp) of 49152 allowed
  exit 0

strata check --model Qwen3.8-27B-UD-Q4_K_XL.gguf --backend vulkan
  66 (kernel, specialization) pairs checked, largest 49152 bytes (gemmi_dp) of 49152 allowed
  exit 0
```

### End-to-end

```
strata generate --model Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --gpu --backend vulkan \
  --prompt "The river town woke before dawn." -n 32
  prefill 94.48 tok/s | gen 125.86 tok/s, exit 0
  "The first light of day was just beginning to touch the eastern horizon when the first
   sounds of activity could be heard. The clatter of hooves on cobble"

strata generate --model Qwen3.8-27B-UD-Q4_K_XL.gguf --gpu --backend vulkan \
  --prompt "The river town woke before dawn." -n 32
  prefill 27.10 tok/s | gen 27.97 tok/s, exit 0
  "Not with birdsong or the creak of shutters, but with the low, insistent hum of the
   *Königstein*—the old"
```

### Bench (informational, same window)

```
strata bench --pp 4096 --tg 128 --reps 1 --ctx 32896
  prefill 1402.2 tok/s | decode 116.3 tok/s
  adjacent M0-T4 window: prefill 1401-1404, decode 115.5-116.2 -> no regression
```

### Xid

`dmesg -T | grep -i Xid` before the window ended at `Fri Nov 6 20:40:46 2026`
(the M0-T4 forced-oversize run); after all M0-T5 runs the tail is unchanged.
No new Xid in the window.

## 2026-11-07 — M0-T7 Vulkan parity baseline (V100, sm_70)

Commit `d6c2beb` (v100) | driver 580.178.04 | Vulkan 1.4.312 | maxComputeSharedMemorySize 49152
Window: production (`llama.router.service`) stopped by the coordinator, V100 free (0 MiB used);
no other process on the card. Total window 13 min 43 s.

```
strata d6c2beb, --gpu --backend vulkan --llama-numerics --fixed-experts, tag vk-sm70
llama-server 0.9.0-dev build 12110 (cd959bae9), -ngl 99 -fa auto -c 4096, CUDA on the V100
20 prompts x 32 greedy tokens = 640 positions
strict vs llama.cpp token-by-token: 625/640 = 97.66%
either llama.cpp mode:              629/640 = 98.28%
llama.cpp self-agreement:           630/640 = 98.44%
verdict: FAIL (629/640, below 634/640 either-mode); documented deviation, see below
Xid: none in the window (dmesg tail unchanged since Fri Nov 6 20:40:46 2026)
```

Environment and reference build:

- `llama-server` `/srv/llama.cpp/build/bin/llama-server`, `version: 0.9.0-dev (build 12110,
  commit cd959bae9)`, `/props` `build_info: b12110-cd959bae9`, on `127.0.0.1:8199`.
- Flags: `-m Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --host 127.0.0.1 --port 8199 -ngl 99 -fa auto
  -c 4096 --seed 1234`; `n_ctx_slot = 4096`, `n_slots = 4`, `kv_unified = true`, model on CUDA0,
  12185 MiB resident.
- `-fa auto` resolved to **disabled**: the server logged
  `resolve_fused_ops: Flash Attention over K/V rows not supported, set to disabled`
  (the 3060 gate ran with `-fa 1`; recorded here because the reference differs).
- Model load 4.4 s (page cache warm), ref cache + one-prompt smoke 32 s, `tools/selfparity.py`
  300 s (640 requests, 0 refused), gated run 145 s.

Per-prompt results for `vk-sm70` (parity mode):

| # | prompt | prompt tokens | agree/n | first_miss | top2_gap_at_miss |
| --- | --- | --- | --- | --- | --- |
| 1 | Write a Rust function that returns the n-th Fibona | 26 | 32/32 | None | None |
| 2 | In Python, how do I read a CSV file and compute th | 33 | 32/32 | None | None |
| 3 | Explain what this C code does: int f(int *a, int n | 54 | 32/32 | None | None |
| 4 | Write a SQL query that returns the top 5 customers | 42 | 32/32 | None | None |
| 5 | Write a regular expression that matches an IPv4 ad | 27 | 32/32 | None | None |
| 6 | Fix the bug: def avg(xs): return sum(xs) / len(xs) | 36 | 32/32 | None | None |
| 7 | Give a bash one-liner that finds the 10 largest fi | 31 | 31/32 | 9 | 0.0329 |
| 8 | Why is the sky blue? Answer in three sentences. | 23 | 32/32 | None | None |
| 9 | Розкажи коротко про історію Києва. | 26 | 32/32 | None | None |
| 10 | Объясни, что такое квантовая запутанность, простым | 31 | 28/32 | 3 | 0.5055 |
| 11 | 请用简单的语言解释什么是机器学习。 | 19 | 30/32 | 23 | 0.0837 |
| 12 | A train leaves at 9:15 and travels 180 km at 72 km | 44 | 32/32 | None | None |
| 13 | Define the word 'entropy' as used in information t | 24 | 32/32 | None | None |
| 14 | Give me a simple recipe for pancakes. | 20 | 31/32 | 11 | 0.1449 |
| 15 | Summarize this design document in five bullet poin | 995 | 32/32 | None | None |
| 16 | What license terms apply here? Summarize: | 444 | 30/32 | 0 | 0.0319 |
| 17 | Find the error in this log: | 1367 | 30/32 | 8 | 0.1615 |
| 18 | Which of these cities is the largest by population | 281 | 31/32 | 30 | 0.1016 |
| 19 | List the keys whose value is false: | 553 | 30/32 | 0 | 0.3904 |
| 20 | Which row has the highest score? | 541 | 32/32 | None | None |

Second block — same server, same window:

```
tag vk-sm70-r2 (repeat of parity mode): strict 625/640 = 97.66% | either 629/640 = 98.28%
tag vk-sm70-default (--gpu --backend vulkan only): strict 626/640 = 97.81% | either 630/640 = 98.44%
```

The repeat is position-identical to the first run (625/629), so the gap to the bar is not
sampling noise from expert placement (`--fixed-experts`) or scheduling.

Deviation note (below the 99% either-mode bar; issue #6 requirement "if below the bar, do not
write verdict PASS"):

- 629/640 (98.28%) is 5 positions short of the 634/640 gate; the 3060 parity-mode runs scored
  635-637/640 either-mode (99.32%), so this is 6-8 positions and ~1 point below that range.
  The default path is *better* here (630/640), the opposite of the 3060 (98.85% default vs
  99.32% parity mode).
- The 3060 bar was measured against a **CPU** llama.cpp build; this run used the **CUDA** build
  with flash attention disabled by the build itself (issue #34 already saw 97.34% strict against
  a CUDA-build reference). llama.cpp's own two modes agree on only 630/640 (98.44%) here, so the
  either-mode gate at 634/640 is stricter than llama.cpp's self-consistency on this card.
- The engine was not changed: no shader, SPIR-V or numerics edit is part of M0-T7. Diagnosing
  the 11 remaining positions (prompts 10, 16, 17, 19 concentrate 10 of them; prompt 10 misses at
  a 0.5055 top-2 gap, so it is not a near-tie) is a follow-up task.
- No new Xid, no device lost, no `strata score` failure; VRAM peak with server + score was
  ~29 GB of 32 GB and neither process was evicted.

## 2026-11-07 — Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf, CUDA sm_70 (M1-T7 bring-up)

First run of the CUDA backend on this card (issue #20, branch `task/20-cuda-sm70-bringup`).
Bring-up numbers, **not** the M2 benchmark protocol: one repetition, one window, no tuning,
`llama.router.service` stopped. Model, machine and driver as above; NVRTC 12.9.

```
cuda device 0: Tesla V100-SXM2-32GB, sm_70 (Volta fallback)
cuda shared memory: opt-in 98304 B per block, attn_decode_gqa*: 2112 B static, dynamic max 73728 B
cuda shared memory: opt-in 98304 B per block, attn_prefill_*: 0 B static, dynamic max 67584 B
cuda shared memory: opt-in 98304 B per block, gemvn_*: 0 B static, dynamic max 98304 B
plan: weights+state+scratch 1984 MB (KV 80 MB for 4096 tokens, prefill 512 MB),
      expert slots 10040 MB, resident experts 10240/10240
pp    512: prefill 836.5 (836.5-836.5) tok/s | tg 32: decode 116.2 (116.2-116.2) tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
```

`generate --prompt "The capital of France is"`: `-n 8` gen 122.78 tok/s, `-n 32`
prompt 5 tok 84.03 tok/s | gen 32 tok 124.27 tok/s. Output coherent ("…Paris is the capital and
most populous city of France…"). The Vulkan baseline of 2026-11-06 on the same model was
`pp 512` 635.6 tok/s (steady state ~775) and `tg 128` 109.8 tok/s; the CUDA figures above are
higher but come from a shorter protocol, so they are indicative only.

Device facts measured here (they answer the M1-T7 open question): the driver reports a
dynamic shared memory opt-in of **98304 B (96 KiB) with no reserve**, `attn_decode_gqa_*` uses
**1920-2112 B of static shared memory** (1920 in `--llama-numerics`), `attn_prefill_*` and
`gemvn_*` use 0 B. `optin - static` is therefore 96192 B for `attn_decode_gqa_*`, below the
98304 B the sm_70 build used to request for every one of those functions.

Kernels that actually executed (`nsys profile --trace=cuda --cuda-graph-trace=node`):

- `generate -n 8`: `gemv_q4_k`, `gemv_q5_k`, `gemv_q6_k`, `gemv_q8_0`, `gemv_f32`,
  `moe_gemv_iq2_xs`, `moe_gemv_iq3_xxs`, `moe_gemv_iq4_xs`, `moe_route`, `moe_combine`,
  `rms_norm`, `rms_norm_rope`, `swiglu_q`, `deltanet_seq`, `conv_silu`, `norm_gate`,
  `rope_freq`, `attn_decode_256`, `attn_merge`, `argmax_part`, `argmax_rows`.
- `bench --pp 512 --tg 32` adds the prefill set: `attn_prefill_256` (10 launches, 2.04 s total,
  i.e. the 67584 B dynamic shared memory launch), `gemm_q5_k`, `gemm_q6_k`, `gemm_q8_0`,
  `gemm_iq2_xs`, `gemm_iq3_xxs`, `gemm_iq4_xs`, `gemms_q5_k`, `gemms_q6_k`, `gemms_iq2_xs`,
  `gemms_iq3_xxs`, `gemms_iq4_xs`, `gemms_f32`, plus `l2_norm`, `rope_neox`, `conv_seq`,
  `conv_state`, `router_topk`, `ssm_gates`, `moe_combine_seq`, `kv_store`.
- No `gemmi_*`/`gemmis_*` and no `quant_q8`: as designed, the sm_70 build has no int8 prefill
  GEMM (`Backend::mmq` is false there and `gemm_raw` takes the f16 `gemm_*` path).
- `gemvn_*` was **not** launched by either command (the lm head ran as `gemv_q4_k`, one input),
  so its 98304 B opt-in was set but the kernel itself is not yet exercised.

### Test suite (same window)

`STRATA_MODEL=$M cargo test --release --no-fail-fast`: 121 passed, 2 failed, 5 ignored,
`grep -c "SKIP: cuda"` = 0 (the CUDA backend opens in every GPU test binary). Failures:

- `backend::spirv_is_fresh` — the known pre-existing Vulkan/toolchain drift (40 of 47 committed
  `.spv` byte-differ with shaderc 2025.2-1); untouched by this task.
- `gpu_model::prefill_matches_reference` — fails on **Vulkan only**: at the step after a 73-token
  prefill the Vulkan argmax is 15019 where CPU and CUDA give 4938 (`max |dlogit|` 1.4263 prefill
  vs 1.4200 decode). Reproduced identically on the base commit `f295a6c` with
  `STRATA_TEST_BACKENDS=vulkan`, so it predates M1-T7 and is outside its scope (Vulkan numerics).

Cross-backend GEMM figures now recorded by `kernels::gemm_matches_reference_for_every_type`
(each backend matches its own 1e-4 reference; the cross-backend figure is kept as data):

- sm_70 CUDA f16 vs Vulkan int8, same tensor, different GEMM math: 3.17e-3 - 4.08e-3 relative
  (the int8 quantization of X, not summation order).
- f16 path with different input conversions (CUDA `__float2half` rounds, this driver's
  `OpFConvert` truncates): 5.41e-4 (IQ2_XS), 4.79e-5 (F32).
- CUDA vs CUDA-path references: 2.6e-7 - 8.2e-7 relative per backend.

No new `Xid`/`NVRM` line in `dmesg` before or after any run; the GPU was left at 0 MiB.

## 2026-11-07 — M1-T8 CUDA parity baseline (V100, sm_70)

Commit `ae95739` (v100, branch `task/22-cuda-parity`, issue #22) | driver 580.178.04 |
CUDA toolkit 12.9.2 (nvcc 12.9.86) | NVRTC 12.9.86
Window: production (`llama.router.service`) stopped by the coordinator, V100 free (0 MiB,
0 %); no other process on the card. Parity run 124 s, whole sequence (integrity check,
20 `strata score` invocations, artifact) 2 min 04 s wall.

```
strata ae95739, --gpu --backend cuda --llama-numerics --fixed-experts, tag cuda-sm70
reference: target/parity-ref.json (sha256 0c30267c…, llama.cpp 0.9.0-dev build 12110
           cd959bae9, FA disabled by the build) reused as-is; llama-server not started
           (0 cache misses, 0 "ref <n> prompt tokens" lines)
20 prompts x 32 greedy tokens = 640 positions
strict vs llama.cpp token-by-token: 630/640 = 98.44%
either llama.cpp mode:              636/640 = 99.38%
llama.cpp self-agreement:           630/640 = 98.44%
verdict: PASS (636/640, above the 634/640 either-mode bar)
Xid: none in the window (42 lines before and after, last Fri Nov 6 20:40:46 2026)
```

Reference integrity before the run: `sha256sum` matched both M0-T7 values
(`0c30267cb673…` / `ba5e28ac2368…`), 20 × 32 rows, `prompt_tokens` equal to
`docs/parity/vk-sm70.json` ([26,33,54,42,27,36,31,23,26,31,19,44,24,20,995,444,1367,281,
553,541]), llama.cpp self-agreement 630/640. `tools/parity.py` printed no cache-miss line,
so the engine was the only process that touched the model. First `strata score` paid the
cold NVRTC/cubin compile (sm_70, ~10 s); the remaining 19 reuse the cache.

Per-prompt results for `cuda-sm70` (parity mode, same reference as `vk-sm70`):

| # | prompt | prompt tokens | agree/n | first_miss | top2_gap_at_miss |
| --- | --- | --- | --- | --- | --- |
| 1 | Write a Rust function that returns the n-th Fibona | 26 | 32/32 | None | None |
| 2 | In Python, how do I read a CSV file and compute th | 33 | 32/32 | None | None |
| 3 | Explain what this C code does: int f(int *a, int n | 54 | 32/32 | None | None |
| 4 | Write a SQL query that returns the top 5 customers | 42 | 32/32 | None | None |
| 5 | Write a regular expression that matches an IPv4 ad | 27 | 32/32 | None | None |
| 6 | Fix the bug: def avg(xs): return sum(xs) / len(xs) | 36 | 32/32 | None | None |
| 7 | Give a bash one-liner that finds the 10 largest fi | 31 | 31/32 | 9 | 0.0329 |
| 8 | Why is the sky blue? Answer in three sentences. | 23 | 32/32 | None | None |
| 9 | Розкажи коротко про історію Києва. | 26 | 32/32 | None | None |
| 10 | Объясни, что такое квантовая запутанность, простым | 31 | 28/32 | 3 | 0.5055 |
| 11 | 请用简单的语言解释什么是机器学习。 | 19 | 30/32 | 23 | 0.0837 |
| 12 | A train leaves at 9:15 and travels 180 km at 72 km | 44 | 31/32 | 5 | 0.0564 |
| 13 | Define the word 'entropy' as used in information t | 24 | 32/32 | None | None |
| 14 | Give me a simple recipe for pancakes. | 20 | 31/32 | 11 | 0.1449 |
| 15 | Summarize this design document in five bullet poin | 995 | 32/32 | None | None |
| 16 | What license terms apply here? Summarize: | 444 | 32/32 | None | None |
| 17 | Find the error in this log: | 1367 | 31/32 | 13 | 0.2825 |
| 18 | Which of these cities is the largest by population | 281 | 32/32 | None | None |
| 19 | List the keys whose value is false: | 553 | 32/32 | None | None |
| 20 | Which row has the highest score? | 541 | 32/32 | None | None |

The 10 strict misses sit on 6 prompts (7, 10, 11, 12, 14, 17); six of them (prompt 10
positions 3 and 12, prompt 11 position 23, prompt 12 position 5, prompt 14 position 11,
prompt 17 position 13) are the position the second llama.cpp mode (context as one batch,
`target/llama-batch-argmax.json`) also produces, which is what carries the either-mode
count to 636/640. Prompt 10 (4 misses, 0.5055 top-2 gap at the first) is not a near-tie.

Comparison with the M0-T7 Vulkan run (same reference, same flags; `d6c2beb`, strict
625/640 = 97.66 %, either 629/640 = 98.28 %):

- CUDA strict is 630/640, i.e. exactly llama.cpp's own self-agreement (630/640), 5
  positions better than Vulkan; either-mode is +7 (636 vs 629).
- Token-level CUDA-vs-Vulkan divergence: 9/640 positions, on 5 prompts.
  - prompt 12 (pos 5): CUDA takes the second llama.cpp mode, Vulkan the reference.
  - prompt 16 (pos 0, 23), prompt 18 (pos 30), prompt 19 (pos 0, 3): CUDA matches the
    reference, Vulkan does not.
  - prompt 17 (pos 8, 13, 16): CUDA matches the reference at 8 and 16, Vulkan at 13;
    both miss one position of that 32-token continuation.
  - Prompts 7, 10, 11, 14 are position-identical in both backends, so the Vulkan
    deviation diagnosed by M3-T5 is not inherited by CUDA on this prompt set.
- This is the first numerical measurement of the CUDA `--llama-numerics` path
  (`attn_prefill_llama` / `attn_decode_llama` on sm_70); it completed with no NVRTC
  error, no launch failure, no NaN and no `strata score` failure (20/20 invocations,
  exit 0).

No new `Xid`/`NVRM` line in `dmesg` (`grep -ic Xid` = 42 before and after the window, the
tail still ends at `Fri Nov 6 20:40:46 2026`). No device lost, no engine crash; the GPU was
left at 0 MiB. Artifact: `docs/parity/cuda-sm70.json` (`gate: pass`; the generated
`target/parity-result-cuda-sm70.json` is not committed).

## 2026-11-07 — Context scaling, CUDA, KV q8 (V100, sm_70)

Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`. Build: `v100` at `b055eff` (M1 complete;
CUDA uses the sm_70 FMA fallback, no tensor cores until M2).

```
strata bench --gpu --backend cuda --kv q8 --ctx 262272 \
  --pp 32768,65536,131072,262144 --tg 128 --reps 1

plan: weights+state+scratch 25280 MB (KV 8708 MB for 262272 tokens, prefill 706 MB)
pp  32768: prefill 104.3 tok/s | tg 128: decode 25.0 tok/s
pp  65536: prefill  79.5 tok/s | tg 128: decode 22.0 tok/s
```

128k and 256k were not completed: the run was stopped at the production
window limit. Estimated on this build: 128k ~30-40 min, 256k ~80-100 min of
prefill. KV q8 at 256k is 8708 MB; the plan fits in 32 GB.

### Follow-up: 128k measured, 256k not completed

```
strata bench --gpu --backend cuda --kv q8 --ctx 262272 \
  --pp 131072,262144 --tg 128 --reps 1

pp 131072: prefill 53.9 tok/s | tg 128: decode 17.8 tok/s
```

256k did not complete: the prefill was still running when the production
window closed and the run was stopped. At this rate it needs roughly 100-120
minutes of prefill on this build (sm_70 FMA fallback).

## 2026-11-08 — Volta tensor cores in the prefill GEMM (M2-T1, V100, sm_70)

Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`. Build: `task/24-cuda-tensor-core-gemm`
(PR #25), based on `v100` at `58fc084`. The f16 prefill GEMM (`gemm_q`,
`gemm_*`/`gemms_*`) now uses `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`
on sm_70; the `m16n8k16` path is unchanged for sm_80+ and `STRATA_GEMM_FMA=1`
selects the old CUDA-core fallback.

### Fragment mapping

The per-lane mapping is the PTX ISA 8.5 §9.7.16.5.1 one (four computations
selected by `g = lane >> 2 & 3`): A row `t + 4*(g >= 4)` (row-major, k 0..3),
B column `t + 4*(g >= 4)` (column-major, k 0..3), and D register
`i = 4*p4 + 2*p2 + e` at row `(t & 1) + 2*p2 + 4*(g >= 4)`, column
`(t & 2) + 4*p4 + e`, with `t = lane & 3`. `SM == false` maps the four
computations to the four n8 tiles (`m8 = j`, `n8 = c = g & 3`, 16 ops per
16-deep k step); `SM == true` maps them to the four 8x8 quadrants of the 16x16
warp tile (`m8 = c >> 1`, `n8 = c & 1`, 4 ops). The standalone asm compiled by
NVRTC 12.9 for sm_70 lowers to `HMMA.884.F32.F32.STEP0..3` (`cuobjdump -sass`),
and the sm_80+ preprocessed source is byte-identical to `v100` (1861 lines).

### Correctness (device-free + V100)

```sh
STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda cargo test --release --test kernels \
  gemm_matches_reference_for_every_type -- --nocapture --test-threads=1

cuda device 0: Tesla V100-SXM2-32GB, sm_70 (Volta, m8n8k4)
Q5_K     blk.0.attn_gate.weight: gemm rel err 9.58e-7
IQ3_XXS  blk.0.ffn_down_exps.weight: gemm rel err 6.31e-7
Q6_K     blk.0.ffn_down_shexp.weight: gemm rel err 6.47e-7
IQ2_XS   blk.0.ffn_gate_exps.weight: gemm rel err 8.95e-7
F32      blk.0.ffn_gate_inp.weight: gemm rel err 4.87e-7
IQ4_XS   blk.1.ffn_down_exps.weight: gemm rel err 6.42e-7
Q8_0     blk.1.ffn_down_shexp.weight: gemm rel err 6.01e-7
Q4_K     output.weight: gemm rel err 9.13e-7
ok
```

All relative errors are `<= 1e-6`, against the 1e-4 assertion; both `gemm_*`
and `gemms_*` (including partial tiles) are covered. `strata generate
--prompt "The capital of France is" -n 32` produced coherent text (" Paris. ...
That is correct.") with no NaN. CPU side, `cargo test --release --lib cuda::`
compiles sm_70 with `GEMM_TC=1`, sm_70 with `GEMM_TC=0` and sm_80 with NVRTC
12.9 (14/14 tests green).

### Benchmarks (same window, median of 3 reps)

```sh
M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
target/release/strata bench --model "$M" --gpu --backend cuda \
  --pp 512,4096 --tg 128 --reps 3 --ctx 32896            # GEMM_TC = 1
STRATA_GEMM_FMA=1 target/release/strata bench ...        # FMA fallback
```

| run | pp 512 (tok/s) | pp 4096 (tok/s) | tg 128 @512 (tok/s) | tg 128 @4096 (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| `GEMM_TC = 1` (m8n8k4) | 1880.4 (1767.1-1948.2) | 1976.0 (1974.1-1977.6) | 123.0 (121.8-123.7) | 119.6 (118.6-120.1) |
| `STRATA_GEMM_FMA=1` | 840.9 (839.5-842.2) | 1036.1 (1034.9-1037.3) | 123.3 (122.3-123.9) | 119.3 (119.0-119.6) |
| delta | **+123.6 %** | **+90.7 %** | -0.2 % | +0.3 % |

Prefill is 2.24x faster at 512 and 1.91x at 4096; decode is unchanged within
noise (99.8 % and 100.3 % of the FMA run, as expected: decode does not use
`gemm_q`). The absolute 512 value is also 2.25x the M1-T7 record of 836.5 tok/s
(`docs/V100-results.md`, M1-T7; same model, same protocol) because that record
was taken on the FMA fallback.

No new `Xid`/`NVRM` line in `dmesg` (`grep -ic Xid` = 42 before and after the
window), no device lost, no engine crash; the GPU was left at 0 MiB and
`llama.router.service` was restarted by the coordinator after the window.

## 2026-11-08 — Volta tensor cores in the prefill attention (M2-T2, V100, sm_70)

Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`. Build: `task/26-cuda-attn-tc`
(PR #27), based on `v100` at `1aaf6af`. `attn_prefill_t` now computes both its
products (QKᵀ and PV) with `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`
on sm_70, under the same `GEMM_TC` switch as the M2-T1 prefill GEMM; the
`m16n8k16` path stays for sm_80+ and `STRATA_GEMM_FMA=1` selects the
CUDA-core fallback for both.

### Fragment mapping

Constants as in `gemm_q` (`l = lane`, `t = l & 3`, `g = l >> 2`, `c = g & 3`,
`hi = g >> 2`). QKᵀ: A = Q rows `8*j + t + 4*hi` of the 16-row warp tile, B = K
row `8*(4*ng + c) + t + 4*hi`, eight `mma884` per 16-deep k step (2 `j` x 4
`ks4`), accumulators `s2[ng][j][4*p4 + 2*p2 + e]` =
`S[8*j + (t & 1) + 2*p2 + 4*hi][8*(4*ng + c) + (t & 2) + 4*p4 + e]`. Softmax in
that layout: rows `qp2[j][p2]`, keys `k0 + 8*(4*ng + c) + (t & 2) + 4*p4 + e`,
row max/sum across the eight lanes of a row with `__shfl_xor_sync(.., 2/4/8)`
(`xor 2` swaps `t & 2`, `xor 4`/`8` swap `c`), `mT`/`lT` of four entries each,
`lT` reduced once after the last tile. P is staged per warp into a
`FA_BQ x (FA_BK + 8)` half tile appended after `vs` in `fa_sm` (5120 B,
`__syncwarp()` before the reads); PV reads A as two `ld_s32` from it, B from
the transposed V tile as four `ld.shared.u16` (`vs + (4*kg + {0..3})*PQ + ...`),
64 `mma884` per tile at HD = 128, accumulators
`o2[j][grp][4*p4 + 2*p2 + e]`, and the store loop mirrors `gemm_q`'s sm_70 one.
SASS (`cuobjdump -sass`, NVRTC 12.9 sm_70 cubin): `attn_prefill_128` has 128
`HMMA.884.F32.F32` (64 QKᵀ + 64 PV, the same FLOPs as 32 `m16n8k16`),
`REG:159 LOCAL:0` (FMA build: `REG:248`). The sm_80+ preprocessed source is
byte-identical to `v100` (1861 lines) and the sm_70 `GEMM_TC=0` variant is
source-identical (1763 lines); `kernels.cu` is +187 lines and -0.

### Correctness (same window)

```sh
M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda cargo test --release --test kernels \
  attn_prefill_matches_decode -- --nocapture --test-threads=1
STRATA_GEMM_FMA=1 STRATA_MODEL=$M ... (same test)
STRATA_MODEL=$M cargo test --release --test kv_q8 -- --test-threads=1
```

`attn_prefill_{hd}` rel err vs decode, assertion `<= 5e-4`:

| case | `GEMM_TC = 1` | `STRATA_GEMM_FMA=1` |
| --- | ---: | ---: |
| hd 64, nh 4/nkv 2 | 6.42e-5 | 6.44e-5 |
| hd 128, nh 4/nkv 2 | 7.26e-5 | 7.39e-5 |
| hd 256, nh 4/nkv 2 | 6.37e-5 | 6.40e-5 |
| hd 256, nh 16/nkv 2 | 6.11e-5 | 6.11e-5 |
| hd 128, nh 8/nkv 8 | 6.13e-5 | 6.17e-5 |
| hd 128, nh 32/nkv 8 | 6.24e-5 | 6.26e-5 |
| hd 128, nh 32/nkv 8, pos 2100 | 2.98e-5 | 2.83e-5 |
| hd 256, nh 16/nkv 2, pos 1000 | 3.51e-5 | 3.36e-5 |
| hd 256, nh 24/nkv 2, pos 0 | 6.14e-5 | 6.18e-5 |

The decode reference itself is `8.7e-8 … 1.4e-7` against the CPU. The two
implementations agree with decode in the third significant digit of the error
metric, i.e. the difference between them is far below the f16 rounding error.
`kv_q8`: `q8 prefill vs decode` 6.16e-5 / 7.67e-5 / 6.16e-5 / 5.98e-5
(hd 64 / 128 / 256 / nh 8,nkv 8), `q8 vs f16 decode` 1.55e-3 … 2.06e-3
(cache quantisation, unchanged). The QSA mask path
(`qwen4exp_qsa_gpu_matches_cpu`) was **not run**: the qwen4exp GGUF is not
present on this host.

### Benchmarks (median of 3 reps, `--pp 512,4096,32768 --tg 128 --ctx 32896`)

| run | pp 512 | pp 4096 | pp 32768 | tg 128 @512 | @4096 | @32768 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GEMM_TC = 1` (m8n8k4) | 2045.3 | 2795.3 | 2263.7 | 123.1 | 119.1 | 105.5 |
| `STRATA_GEMM_FMA=1` | 833.6 | 1030.7 | 489.2 | 121.9 | 118.4 | 105.3 |
| delta | **+145.3 %** | **+171.2 %** | **+362.7 %** | +1.0 % | +0.6 % | +0.2 % |

Against M2-T1 (same model and protocol, `docs/V100-results.md` block above):
prefill @512 1880.4 → **2045.3** (+8.8 %), @4096 1976.0 → **2795.3** (+41.5 %),
so the M2 exit target of 2000 tok/s at 4096 is met with the attention on the
tensor cores. Decode is unchanged within noise (decode does not use
`attn_prefill_t`). Reps ranges: 2034.8-2062.3 / 2786.7-2803.5 / 2263.1-2264.2.

`strata generate --prompt "The capital of France is" -n 32` (TC build) produced
"That is correct. Paris is the capital and most populous city of France. …"
at 123.38 tok/s, no NaN.

### Kernel time (`nsys profile --trace=cuda`, pp 4096, 10 prefill chunks)

```sh
nsys profile --trace=cuda -o nsys_tc target/release/strata bench --model "$M" \
  --gpu --backend cuda --pp 4096 --tg 1 --reps 1 --ctx 4200
nsys stats --report cuda_gpu_kern_sum nsys_tc.nsys-rep
```

| kernel | `GEMM_TC = 1` | `STRATA_GEMM_FMA=1` | delta |
| --- | ---: | ---: | ---: |
| `attn_prefill_256` self time per launch | 6.124 ms | 70.760 ms | **-91.3 %** |
| `attn_prefill_256` share of kernel time | 4.5 % | 18.3 % | |
| total kernel time (10 chunks) | 1.371 s | 3.869 s | -64.6 % |

The per-kernel requirement (at least 20 % lower self time per launch) is met by
a wide margin. The end-to-end prefill gain (+171 % vs the FMA build) also
includes the M2-T1 `gemm_q` win, since this build carries both.

No new `Xid`/`NVRM` line in `dmesg` (`grep -ic Xid` = 42 before and after both
windows), no device lost, no NaN in any of the 3 reps; the GPU was left at
0 MiB and `llama.router.service` was left stopped for the coordinator to
restart.

## 2026-11-08 — Shared-memory carveout and tile sizing (M2-T3, V100, sm_70)

Window: `llama.router.service` stopped and queue drained by the coordinator,
V100 free, `dmesg | grep -ic Xid` = **42 before and after**. 50 min budget.
Build: `59bf0d4` + the M2-T3 plumbing (Rust-owned tiles, `attn_prefill_smem`,
`carveout()` decision). Protocol unchanged from M2-T1/T2:
`--pp 512,4096 --tg 128 --reps 3 --ctx 32896`, median of 3 reps.

### Starting resource usage (`cuobjdump -res-usage`, sm_70 cubin)

| kernel | REG | static smem | threads | blocks/SM by regs | by smem | warps/SM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemm_q2_k` | 157 | 27648 | 256 | 1 | 1 at the chosen 32 KiB config | 8 (12.5 %) |
| `gemms_q2_k` | 64 | 13824 | 256 | 4 | 4 at 64 KiB | 32 (50 %) |
| `attn_prefill_64` | 123 | 0 (+23552 dyn) | 128 | 4 | 4 | 16 |
| `attn_prefill_128` | 159 | 0 (+39936 dyn) | 128 | 3 | 2 only with a 96 KiB carveout | 8 |
| `attn_prefill_256` | 244 | 0 (+72704 dyn) | 128 | 2 | 1 (142 KiB for two) | 4 (6.25 %) |
| `attn_decode_gqa_128` | 113 | 2112 (+36864 dyn) | 256 | 2 | 2 | 16 |
| `attn_decode_gqa_256` | 164 | 2112 (+73728 dyn) | 256 | 1 | 1 | 8 |

This confirms the issue's table: `gemm_*` is register-bound at one block/SM,
`attn_prefill_256` is shared-memory-bound at one block.

`ncu` LaunchStats/Occupancy (`--pp 4096 --tg 1 --launch-count 6`) captured the
first six `gemm_*` launches before the count ran out; the attention kernels
were not re-profiled (time budget). The important read-back: with the shipped
default (no carveout attribute) the driver already selects the **minimal**
configuration for `gemm_q5_k` (`Shared Memory Configuration Size` 32768 B for
27648 B static), so there was no L1 to reclaim there:

| kernel | REG | shared config | static | dynamic | block limit (regs/smem) | theoretical occ. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemm_q5_k` | 157 | 32768 | 27648 | 0 | 1 / 1 | 12.50 % |
| `gemm_q6_k` | 128 | 65536 | 27648 | 0 | 2 / 2 | 25 % |
| `gemms_f32` | 58 | 65536 | 13824 | 0 | 4 / 4 | 50 % |

### Candidates (median of 3 reps)

All candidates are env-selectable, no rebuild: `STRATA_SMEM_CARVEOUT=off|policy|min`,
`STRATA_GEMM_MIN_BLOCKS`, `STRATA_GEMM_BK`. C0 is `off`, i.e. the previous
behaviour; the startup line reads
`cuda shared memory: opt-in 98304 B per block, attn_prefill_*: 0 B static, dynamic max 72704 B, carveout default`.

| run | changes | pp 512 | pp 4096 | tg @512 | tg @4096 | vs C0 512 / 4096 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C0 | none (carveout off) | 2050.8 | 2779.1 | 123.0 | 119.0 | — |
| C1 policy | carveout 82 % (`attn_prefill_*`), 78 % (`attn_decode_gqa_256`) | 2027.2 | 2781.4 | 122.9 | 119.1 | −1.15 % / +0.08 % |
| C1 min | policy + minimal gemm carveout (29 %) | 1476.7 | 2466.6 | 123.0 | 119.3 | **−28.0 % / −11.3 %** |
| C2 | `__launch_bounds__(·, 2)` (REG 157→128) | 2012.5 | 2989.4 | 122.5 | 118.6 | −1.87 % / **+7.57 %** |
| C3 | `GEMM_BK = 32` (REG 157→80, smem 15360) | 1940.8 | 2884.6 | 122.8 | 119.2 | −5.36 % / +3.80 % |
| C2+C3 | both | 1952.9 | 2862.5 | 123.1 | 119.4 | −4.77 % / +3.00 % |
| final (default build) | same cubin as C0 | 2029.2 | 2758.8 | 122.9 | 118.7 | −1.05 % / −0.73 % |

Reps ranges: C0 2044.9-2054.0 / 2776.3-2781.7; C2 2001.2-2023.4 /
2981.9-2999.9; C3 1931.4-1946.1 / 2871.3-2894.8. The final line is a rerun of
the byte-identical C0 cubin after the correctness tests (2029.2/2758.8,
2029.2/2765.1 and 2016.2/2765.1); its 0.5-1 % spread is run-to-run drift, not
a code change.

### Decision

**No candidate adopted.** The adoption rule is pp 4096 **≥ +2 %** *and*
pp 512 **≥ −1 %**:

- **C1 rejected** — +0.08 % at 4096, −1.15 % at 512, decode unchanged. The
  `ncu` read-back shows the driver already picks the smallest fitting
  configuration for `gemm_*`, so the explicit request could only add a hint.
  `min` mode was actively harmful (−28 % at 512): lowering the carveout below
  the driver's choice costs L1 outright.
- **C2 rejected** — the only candidate with a real prefill win (+7.6 % at
  4096) but −1.87 % at 512. `cuobjdump`: `REG:128 STACK:16 LOCAL:0`, i.e. no
  local-memory spill flag but a 16-byte stack frame from the 157→128 register
  cap; `gemms_*` rises 64→101 REG. The rule is strict, so this is documented
  for a follow-up (why does the small-batch path lose?).
- **C3 rejected** — +3.8 % at 4096, −5.4 % at 512. `REG:80 SHARED:15360
  LOCAL:0` for `gemm_q2_k`; `gemms_*` 48 REG. `GEMM_BK=32` halves the K step
  and doubles the step count.
- **C2+C3 rejected** — +3.0 % at 4096, −4.8 % at 512.
- **C4 dropped** — `GEMM_BM=64` changes the tile-list stride shared with
  Vulkan (`backend.rs`, `gpu_model/prefill.rs`); out of scope per the issue.

The shipped default therefore keeps the previous constants (`GEMM_BM 128`,
`GEMM_BK 64`, `FA_BQ 64`, `FA_BK 32`, no minimum-block hint) and
`STRATA_SMEM_CARVEOUT` defaults to `off`. R1/R2 remain: the values are
Rust-owned, injected as `#define`s with `static_assert`s, and
`attn_prefill_smem` is the single owner of the attention layout.

### Correctness and safety

```sh
M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda cargo test --release --test kernels matches \
  -- --nocapture --test-threads=1        # tensor-core path: 10 passed
STRATA_GEMM_FMA=1 STRATA_MODEL=$M ...    # FMA fallback: 10 passed
```

Both runs include `gemm_matches_reference_for_every_type` and
`attn_prefill_matches_decode`; no failure, no NaN. `dmesg | grep -ic Xid` = 42
after the window (unchanged), no `NVRM`/`device lost` line, GPU left at
**0 MiB**. `llama.router.service` was left stopped for the coordinator.

## 2026-11-08 — Decode GEMV bandwidth (M2-T5, V100, sm_70)

Window: `llama.router.service` stopped and queue drained by the coordinator,
V100 free, `dmesg | grep -ic Xid` = **42 before and after**, GPU left at
**0 MiB**, service left stopped for the coordinator. Build: `task/30` on top of
`d59a52b`. Roofline: peak HBM2 = 2 x 877 MHz x 4096 bit / 8 = **898.0 GB/s**;
85 % target = 763 GB/s. Bench protocol unchanged from M2-T3:
`--pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896`, median of 3.

### Per-token byte budget (the roofline denominator)

From the GGUF tensor table of the reference model, 100 % expert hit rate, no
cross-token L2 reuse, `token_embd.weight` excluded (one row per token):

| category | MB/token | share | family |
| --- | ---: | ---: | --- |
| attention / SSM 2-D | 713.3 | 41.2 % | dense GEMV |
| routed experts, top-8 | 329.0 | 19.0 % | `moe_gemv_*` |
| `output.weight` (lm head) | 286.1 | 16.5 % | dense GEMV |
| SSM / delta 2-D | 226.1 | 13.1 % | dense GEMV |
| shared expert (`*_shexp`) | 92.9 | 5.4 % | dense GEMV |
| router (`gate_inp`) | 83.9 | 4.8 % | dense GEMV |
| **total** | **1731.3** | 100 % | dense 81.0 %, routed 19.0 % |

Roofline: 1.7313 GB x 898.0 GB/s -> **518.7 tok/s at 100 %**, 440.9 at 85 %.
Measured baseline 119.9 tok/s = **207.6 GB/s = 23.1 %** end-to-end; the
adopted build 123.7 tok/s = **214.2 GB/s = 23.9 %**.

### Per-shape achieved bandwidth and % of roofline (baseline, `gemv_bandwidth`)

200 launches after a 300 ms clock ramp, one launch per row (MoE: 8 experts,
`grid.y = 8`) — `at 100 %` is `bytes / 898 GB/s`, so `measured - at 100 %` is
the launch/ramp overhead of that shape.

| tensor | type | shape | measured | GB/s | % of 898 | at 100 % | bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `blk.0.attn_gate` | Q5_K | 4096x2048 | 10.47 us | 551.0 | 61.4 % | 6.42 us | 5.77 MB |
| `blk.0.attn_qkv` | Q5_K | 8192x2048 | 25.51 us | 452.2 | 50.4 % | 12.84 us | 11.53 MB |
| `blk.0.ffn_down_exps` (8) | IQ3_XXS | 2048x512 | 15.39 us | 208.7 | 23.2 % | 3.58 us | 3.21 MB |
| `blk.0.ffn_down_shexp` | Q6_K | 2048x512 | 4.89 us | 175.9 | 19.6 % | 0.96 us | 0.86 MB |
| `blk.0.ffn_gate_exps` (8) | IQ2_XS | 512x2048 | 10.15 us | 238.9 | 26.6 % | 2.70 us | 2.42 MB |
| `blk.0.ffn_gate_inp` | F32 | 256x2048 | 4.95 us | 423.8 | 47.2 % | 2.34 us | 2.10 MB |
| `blk.0.ffn_gate_shexp` | Q5_K | 512x2048 | 5.05 us | 142.6 | 15.9 % | 0.80 us | 0.72 MB |
| `blk.0.ssm_alpha` | F32 | 32x2048 | 3.46 us | 75.8 | 8.4 % | 0.29 us | 0.26 MB |
| `blk.0.ssm_out` | Q6_K | 2048x4096 | 21.46 us | 320.7 | 35.7 % | 7.66 us | 6.88 MB |
| `blk.1.attn_gate` | Q6_K | 4096x2048 | 15.62 us | 440.4 | 49.0 % | 7.66 us | 6.88 MB |
| `blk.1.attn_qkv` | Q6_K | 8192x2048 | 28.31 us | 486.1 | 54.1 % | 15.33 us | 13.76 MB |
| `blk.1.ffn_down_exps` (8) | IQ4_XS | 2048x512 | 16.80 us | 265.2 | 29.5 % | 4.96 us | 4.46 MB |
| `blk.1.ffn_down_shexp` | Q8_0 | 2048x512 | 4.69 us | 237.7 | 26.5 % | 1.24 us | 1.11 MB |
| `blk.1.ffn_gate_exps` (8) | IQ3_XXS | 512x2048 | 10.94 us | 293.4 | 32.7 % | 3.58 us | 3.21 MB |
| `blk.1.ffn_gate_shexp` | Q6_K | 512x2048 | 5.18 us | 166.1 | 18.5 % | 0.96 us | 0.86 MB |
| `blk.11.attn_output` | Q5_K | 2048x4096 | 11.90 us | 484.5 | 54.0 % | 6.42 us | 5.77 MB |
| `output.weight` (lm head) | Q4_K | 248320x2048 | 594.41 us | 481.3 | 53.6 % | 318.56 us | 286.06 MB |
| `token_embd.weight` (1 row) | Q5_K | 248320x2048 | 603.66 us | 579.2 | 64.5 % | 389.35 us | 349.63 MB |

`token_embd` is printed for completeness only: decode reads one row of it, so
it is not part of the budget.

### Candidates (median of 3 reps, one process each)

| id | env | tg @512 | tg @4096 | pp 4096 | tg @32768 | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B0 | — | 123.8 | 119.9 | 2934.2 | 106.5 | baseline |
| R1 | `STRATA_RPW_MOE=4` | 128.2 (+3.55 %) | 124.0 (+3.42 %) | 2929.4 (-0.16 %) | 109.7 | prototype |
| B2 | `STRATA_GEMV_BLOCK=512` | 124.3 (+0.40 %) | 120.5 (+0.50 %) | 2930.7 (-0.12 %) | 107.0 | rejected: < +2 % |
| W | R1 + B2 | 128.2 (+3.55 %) | 124.1 (+3.50 %) | 2922.3 (-0.40 %) | 110.0 | rejected: per-shape |
| **D** | **rpw: experts 2048-col `1`, 512-col `4`** (Rust default) | **127.9 (+3.31 %)** | **123.7 (+3.17 %)** | **2933.1 (-0.04 %)** | **109.5 (+2.82 %)** | **adopted** |

Rejected on the per-shape guard or on their own numbers:

- **R1** (all experts 4 rows per warp): the gain is entirely the down GEMV —
  `ffn_down_exps` 208.7 -> 262.4 GB/s (+25.7 %, 15.4 -> 12.2 us) and 265.2 ->
  328.8 (+24.0 %) — but gate/up (2048 columns) lost 7.4 % (238.9 -> 221.2) and
  2.8 %. The 2048-column launch is 2.4 MB, short enough that fewer blocks hurt
  more than the larger rows/block help.
- **W** (R1 + 512-thread blocks): same decode gain, but `attn_gate` -5.4 %
  (551.0 -> 521.5), `ffn_gate_exps` -4.4 %, `blk.1.attn_qkv` -2.8 %: fails the
  "no per-shape regression > 2 %" guard.
- **B2** (512-thread blocks alone): `attn_output` +6.2 % and `ssm_out` +3.3 %,
  but `attn_qkv` -2.5 %, and decode only +0.5 % — below the +2 % threshold.
- **L1** (`STRATA_GEMV_LOADW=1`, one 16-byte Q4_K header load): **rejected**,
  lm head 481.3 -> 376.7 GB/s (-21.7 %, 594 -> 759 us). The uniform alignment
  branch and the vector load cost more than the four scalar loads they replace
  at this block layout.
- **R2** (`STRATA_RPW_DENSE=4`): lm head +2.2 % but `ssm_out` -33 %
  (320.7 -> 214.1) and `attn_output` -30 % (484.5 -> 338.6). Rejected.

Adopted default: `Gpu::rpw` uses **4 rows per warp for the 512-column expert
GEMV** (was 16) and keeps 1 (2 after the MoE bump) for 2048-column gate/up.
Per-shape after adoption: down 262.0/327.0 GB/s (+25.5 %/+23.3 %), gate/up
238.8/295.6 (unchanged), every dense shape unchanged. No env var is needed.

### Why 85 % is not reached (ceiling)

`ncu` attribution could not be completed in this window: the unbounded
`tools/ncu_gemv.sh` run (kernel replay, 4 metrics, all decode kernels) was
still running after 17 min and was stopped, and a bounded re-run
(`--launch-count 240`, `gemv|gemvn|moe_gemv`) produced no metric rows — ncu
2025.2.1 does not profile the CUDA-graph launches of this build on sm_70 with
driver 580 (the app saw `decode 0.0 tok/s` with the profiler attached). No
counter is reported that was not measured; the attribution below is the
harness's own bytes/time per shape, as the issue allows.

The arithmetic per shape (bytes / 898 GB/s vs measured) shows a fixed
launch/ramp cost of roughly 4-13 us that dominates the small launches:

- MoE gate/up: 2.42-3.21 MB -> 2.70-3.58 us at 100 %, measured 10.2-10.9 us.
- MoE down: 3.21-4.46 MB -> 3.58-4.96 us, measured 15.4-16.8 us (before the
  adopted rpw change) and 12.3-13.7 us after.
- lm head: 286.06 MB -> 318.6 us at 100 %, measured 594.4 us: 276 us of
  overhead on the one shape large enough to amortize everything, i.e. the
  stream itself runs at ~53.6 %.
- Best streamed shape: `attn_gate` 5.77 MB at 61.4 % (551 GB/s).

Occupancy explains the common factor: `cuobjdump -res-usage` gives
`gemv_q4_k` 56 REG, `moe_gemv_q4_k` 64, `gemvn_q4_k` 96, `moe_gemv_iq3_xxs` 56,
`gemv_q5_k` 64 — at 56-64 registers an SM holds 4 blocks of 256 threads
(32 of 64 warps, 50 % occupancy) whether the block is 128, 256 or 512 threads,
which is why the block-size sweep moved almost nothing. sm_70 has no
`cp.async` (M1-T6), so the only in-flight bytes are the 8 B per lane per
super-block (`ldu32`+`ldu32`), and 50 % occupancy cannot hide the remaining
latency. 85 % (763 GB/s) would require ~1 us of launch cost on a 5 MB launch
and a stream several times more parallel than the current 8 B/lane: that is a
kernel restructuring (more independent loads per lane, fewer per-row shuffles),
not a launch-shape knob. Documented as the ceiling; no re-fusion of gate/up or
down/combine was attempted.

### Correctness and safety

```sh
STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda cargo test --release --test kernels matches \
  -- --nocapture --test-threads=1        # 11 passed, 0 failed (gemvn_matches_gemv, gemm, moe, attn)
dmesg -T | grep -ic Xid                  # 42 before and after (unchanged)
nvidia-smi --query-gpu=memory.used       # 0 MiB at the end
```

`gemv_byte_budget_matches_issue` (device-free) pins the budget table above.
`llama.router.service` was left stopped for the coordinator.

## 2026-11-08 — MoE expert residency and next-layer prefetch (M2-T4, V100, sm_70)

Window: `llama.router.service` stopped and queue drained by the coordinator, V100 free,
`dmesg | grep -c Xid` = **42 before and after** (the Nov 6 entries, unchanged), GPU left at
**0 MiB** / 50 °C idle, service left stopped for the coordinator. Build:
`task/34-cuda-moe-residency` merged with `v100` @ `fd78100` (M2-T5 `b4d7408`), one binary per arm
(CUDA sm_70 via NVRTC 12.9). Protocol unchanged: `tools/bench_v100.sh` →
`--pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896`, median of 3 reps, min–max in parentheses,
one process per arm, ~60 s per arm.

### Fully resident fast path (R1/R2)

The reference model is 100 % resident, so the log now reads every counter of the task as a
measured zero instead of omitting it (log of run 1; the other arms differ only in the numbers):

```
plan: weights+state+scratch 2560 MB (KV 642 MB for 32896 tokens, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
expert traffic: 10.4 MB per layer active set (top-8), 313.8 MB per token over 40 MoE layers
pp    512: prefill 2135.7 (2129.6-2144.9) tok/s | tg 128: decode 129.5 (128.9-129.8) tok/s
expert traffic: 313.8 MB/token -> 42.7 GB/s at 129.7 tok/s
pp   4096: prefill 2970.2 (2966.9-2973.0) tok/s | tg 128: decode 125.2 (124.9-125.4) tok/s
expert traffic: 313.8 MB/token -> 41.2 GB/s at 125.2 tok/s
pp  32768: prefill 2386.8 (2385.6-2388.1) tok/s | tg 128: decode 110.6 (110.5-110.6) tok/s
expert traffic: 313.8 MB/token -> 36.4 GB/s at 110.6 tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
worker busy 0.0 µs/token, worker stalls 0
mailbox traffic: 0 GPU writes, 0 host routing reads (fast path on, 10240/10240 experts resident)
```

`10.4 MB` is the **largest** layer's active set (biggest expert size class, 1.30 MB per slot); the
mean layer is **7.85 MB** (313.8 MB / 40), which matches the 7.9 MB of the issue. Both exceed the
6 MB L2 and the token set is 313.8 MB, read at **36.4–42.7 GB/s = 4.1–4.8 %** of the 898.0 GB/s
HBM2 roofline of M2-T5: decode is not expert-bandwidth bound, which is what R3 assumes.

| arm | pp 512 | pp 4096 | pp 32768 | tg @512 | tg @4096 | tg @32768 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v100` `fd78100` (no #34) | 2122.7 (2116.9–2129.0) | 2939.3 (2935.7–2941.4) | 2364.8 (2361.2–2368.3) | 128.1 (127.4–128.4) | 123.7 (123.4–123.9) | 109.5 (109.5–109.5) |
| #34, run 1 | 2135.7 (2129.6–2144.9) | 2970.2 (2966.9–2973.0) | 2386.8 (2385.6–2388.1) | 129.5 (128.9–129.8) | 125.2 (124.9–125.4) | 110.6 (110.5–110.6) |
| #34, run 2 | 2113.5 (2106.1–2123.9) | 2924.7 (2921.2–2929.1) | 2360.3 (2358.9–2362.4) | 129.4 (128.6–129.8) | 125.1 (125.0–125.3) | 110.6 (110.5–110.7) |

Against the same-code reference, decode gains **+1.1 / +1.2 / +1.0 %** (`tg @512/@4096/@32768`,
run 1) and **+1.0 / +1.1 / +1.0 %** (run 2); prefill is within −0.5 … +1.1 % of the reference's own
spread. Removed per token, all now printed as zero: the worker request, the worker's scan of every
layer's REQ/SKIP words, the host routing read + `record` per layer, and `adapt`'s scan of all 10240
entries every 8 tokens. What remains is the GEMV work of M2-T5.

Caveat on the size of the gain: the *same* `v100` reference binary measured **121.2 (120.9–121.3)
@4096 and 107.4 @32768** at 07:33 UTC and **123.7 / 109.5** at 08:42 UTC in this window, i.e. the
window itself drifts ~2 % (clock ramp), more than the +1 % this task earns. The direction is
consistent across two runs and three sizes, but the margin is inside the window-to-window noise.

Issue acceptance thresholds (derived from M2-T2/T3): `tg @4096 ≥ 121.8` → 125.2 **pass**,
`pp 512 ≥ 2010` → 2135.7 **pass**, `pp 4096 ≥ 2730` → 2970.2 **pass**, `tg @32768 ≥ 117.5` → 110.6
**not met**. That bar predates M2-T5, whose own window measured 109.5 @32768 after the 512-column
expert `rpw` change (block above), and the same-code reference here is 109.5 as well: #34 is
**+1.0 % on it**, so the miss is the stale bar rather than a regression of this task.

### Forced partial residency (R2c)

`STRATA_EXPERT_MB=6000` on the same binary, `--pp 4096 --tg 128 --reps 1`: 6117/10240 experts
resident, the miss path runs to completion with no hang, no trap, no `Xid`:

```
plan: weights+state+scratch 2560 MB (KV 642 MB for 32896 tokens, prefill 512 MB), expert slots 5997 MB, resident experts 6117/10240
pp   4096: prefill 2648.3 (2648.3-2648.3) tok/s | tg 128: decode 72.0 (72.0-72.0) tok/s
expert hit rate 98.3% (702 misses), cache swaps 594
worker busy 5937.3 µs/token, worker stalls 0
mailbox traffic: 80 GPU writes, 5120 host routing reads (fast path off, 6117/10240 experts resident)
```

Hit rate < 100 %, non-zero swaps (594) and 5.9 ms/token of CPU expert work prove the partial path
still publishes to the mailbox, waits on the worker and adapts, exactly as before #34. The 702
misses are 1.7 % of the 40 960 routed experts of the run. `80 GPU writes` counts the `moe_route`
launches the **host** enqueued (two graph recordings for 40 layers each); a replayed CUDA graph
re-writes the same mailbox without a host call, so the figure is a lower bound on the partial path
and exactly 0 on the fast path, where the recorded graph carries `mb = 0`.

### Next-layer L2 prefetch (`STRATA_MOE_PREFETCH=1`) — rejected

Same binary, same window, same protocol, one arm with the knob (default off), interleaved with the
two baseline runs above:

| arm | pp 512 | pp 4096 | pp 32768 | tg @512 | tg @4096 | tg @32768 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| #34, no prefetch (2 runs) | 2113.5–2135.7 | 2924.7–2970.2 | 2360.3–2386.8 | 129.4–129.5 | 125.1–125.2 | 110.6 |
| #34, `STRATA_MOE_PREFETCH=1` | 2121.7 (2117.8–2124.6) | 2933.7 (2930.4–2937.2) | 2361.6 (2358.9–2365.4) | 126.7 (126.0–127.1) | 122.5 (122.2–122.7) | 108.5 (108.4–108.6) |
| Δ decode | | | | −2.2 % | −2.1 % | −1.9 % |

Adoption rule (from the issue): adopt only if the median decode gain exceeds the rep spread at
both 4K and 32K. Here it is a **loss of 1.9–2.2 % with a ±0.3 % spread** at all three sizes, i.e.
well outside noise and in the opposite direction: **rejected** and left off by default.

The result matches the prediction of the analysis: the previous token's routing of the next layer
(8 experts ≈ 7.85 MB mean, 10.4 MB largest) does not fit the 6 MB L2 alongside the layer being
computed, and decode runs at 4.1–4.8 % of the roofline, so a miss there costs nothing to hide.
The prefetch adds ~256 blocks per layer per token (one 128-byte line per thread over the whole
slot) whose issue bandwidth is not free, and no arm shows one L2 hit's worth of gain. The knob
stays as the measurement hook (`STRATA_MOE_PREFETCH=1`), default off, no CLI flag.

### Safety

```sh
dmesg -T | grep -c Xid        # 42 before the window and 42 after
nvidia-smi --query-gpu=...    # 0 MiB used, 50 C, 0 % utilisation at the end
```

No `Xid`, no `device lost`, no trap (`worker stalls 0` in every arm). `llama.router.service` was
left **stopped** for the coordinator.

## 2026-11-08 — Flags on sm_70: KV q8, serve --parallel, MTP (M2-T6, V100, sm_70)

Build `c54714b` (`task/32-cuda-flags`, current `v100` merged in, so M2-T4/M2-T5 are included). Reference
`M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (no `nextn` head),
MTP model `M27=/mnt/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf` (`n_nextn: 1`).
One window, `llama.router.service` stopped by the coordinator; `cuda` backend only.

### Correctness after the merge

`STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda cargo test --release --test kv_q8 -- --test-threads=1`
→ 2 passed. q8 prefill vs decode 5.98e-5…7.67e-5 (limit 5e-4), q8 vs f16 decode 1.55e-3…2.06e-3
(limit 3e-3), 3-slot plan `KV 3 MB for 256 tokens in each of 3 slots`.

`cargo test --release --test serve serve_parallel_smoke -- --nocapture` → 1 passed: `serve --parallel 2
--kv q8` answered two concurrent chat requests with `200`/`choices`, plan `KV 5 MB for 512 tokens in each
of 2 slots`.

### KV cache: f16 vs q8 (`--pp 512,4096 --tg 128 --reps 3 --ctx 32896`)

| cache | KV @32896 | pp 512 | pp 4096 | tg @512 | tg @4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| f16 | 642 MB | 2132.3 (2119.5–2139.5) | 2945.2 (2941.8–2948.6) | 129.5 (128.7–129.9) | 125.1 (124.8–125.3) |
| q8 | 341 MB | 2129.6 (2124.6–2132.1) | 2950.9 (2948.0–2956.8) | 129.1 (128.5–129.5) | 124.2 (124.0–124.3) |

q8 halves the KV (341/642 = 53.1 %) and moves throughput by −0.1…+0.2 % for prefill and −0.3…−0.7 %
for decode (inside the rep spread). Regression gate (0.97 × M2-T2 medians, pp 4096 ≥ 2711, tg @4096 ≥
115.5): **pass** on both arms.

`generate --kv q8 -n 64` produced coherent text and stopped at EOS after 22 tokens
(`prompt 5 tok 102.96 tok/s | gen 22 tok 130.32 tok/s`, 100 % expert hit rate).

### `serve --parallel`: aggregate decode

`serve --model $M --gpu --backend cuda --kv q8 --parallel 4 --ctx 4096`, one warm-up then one round per N,
four identical greedy requests of 64 tokens, completion tokens from `/metrics`:

| N concurrent | wall | completion tokens | aggregate tok/s | per request (s) |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.57 s | 64 | 112.4 | 0.56 |
| 2 | 0.84 s | 128 | 153.1 | 0.84 / 0.83 |
| 4 | 1.37 s | 256 | 187.2 | 1.35 / 1.35 / 1.37 / 1.35 |

Scaling over one request: **1.36× at N=2, 1.67× at N=4**. Plan: `KV 42 MB for 4096 tokens in each of
4 slots`.

### `--mtp` on Qwen3.8-27B (`--pp 512 --tg 128 --reps 1 --ctx 8192`)

The reference model has no draft head, so MTP numbers are from `$M27` (different model: not comparable
with the M2-T8 gate). `bench` does not sweep k; four separate runs, same prompt:

| drafts/step | prefill 512 | decode | drafts accepted |
| ---: | ---: | ---: | ---: |
| 0 (no `--mtp`) | 377.1 | 28.7 | — |
| 1 | 370.7 | 41.5 | 63/65 (96.9 %) |
| 2 | 372.9 | 47.1 | 85/90 (94.4 %) |
| 3 | 372.1 | 46.5 | 94/102 (92.2 %) |

Best k = 2 (**+64 %** over plain decoding). `generate --mtp 3 -n 64` produced coherent text,
`prompt 5 tok 26.83 tok/s | gen 64 tok 32.86 tok/s`, `MTP drafts accepted 40/72 (55.6 %)` (a different,
free-running prompt than the bench; acceptance falls over a longer greedy chain).

### Unsupported combinations

All messages name the flag(s); no silent fallback.

| command | exit | message |
| --- | ---: | --- |
| `bench --kv q8 --llama-numerics …` | 1 | `--kv q8 does not combine with --llama-numerics (its attention reads f16 KV)` |
| `serve --parallel 0` | 1 | `--parallel 0: 1 to 255 slots` |
| `serve --parallel 256` | 1 | `--parallel 256: 1 to 255 slots` |
| `serve --parallel 4 --mtp 1` | serving | `--parallel 4 and --mtp 1: not supported together, decoding without MTP drafts` (MTP off, exit 0) |
| `generate --mtp 1` on `$M` (no head) | 0 | `--mtp 1: this model has no MTP head: decoding without drafts` |
| `generate --mtp 1` without `--gpu` | 0 | `--mtp 1: requires --gpu: decoding without drafts` |

### Safety

```sh
dmesg -T | grep -c Xid        # 42 before the window and 42 after
nvidia-smi --query-gpu=...    # 0 MiB used, 47 C, 0 % utilisation at the end
```

No `Xid`, no `device lost`, no trap. One `serve` process from the parallel test was still listening after
the wrapper shell was signalled; it was sent SIGTERM and the GPU returned to 0 MiB before the Xid/`nvidia-smi`
reading above. `llama.router.service` was left **stopped** for the coordinator.

## 2026-11-08 — Cross-backend numerical consistency (M2-T7, CPU part)

Issue #36, branch `task/36-cuda-consistency-report`, base `v100` after M2-T6. This block is the CPU
half of the report: the reproducible comparison script, the legitimate-difference registry and the
dry run that reproduces the documented 9/640 on the **pre-M2** artifacts. The M2-build numbers
(per-op, parity, attribution arms, KLD) are appended in the GPU-window section right below, same
heading, same artifacts.

### The comparison script

`tools/consistency.py` (new, no GPU, no model, no network) turns two `target/parity-result-<tag>.json`
files into `docs/parity/cross-backend.json` plus the markdown rows of this section:

- end-to-end: strict/either per backend, the cross-backend divergence count and its classification
  (CUDA-only correct, Vulkan-only correct, both wrong);
- the divergence list: prompt, position, both backends' token id, the llama.cpp reference token, the
  one-batch llama.cpp token, each backend's top-5 gap at that position when the artifact carries
  `top5` (recorded from this change on, `tools/parity.py`), and the registry entry that explains it;
- the per-op table, aggregated from the `--nocapture` logs of `tests/kernels.rs`,
  `tests/llama_numerics.rs` and `tests/kv_q8.rs` (`--kernels`), with CUDA-vs-CPU, Vulkan-vs-CPU and
  CUDA-vs-Vulkan columns; measurement lines that no pattern matches are listed in
  `per_op_unparsed` instead of being dropped;
- attribution arms (`--arm name=file`) compared against the CUDA artifact: how many of the base
  divergences an arm resolves and how many it leaves;
- full-vocabulary KLD from two `perplexity --kld-base` dumps (`--kld`), pure Python (no numpy on
  this host), computing mean/max KLD, RMS Δp, top-1 agreement and Σp as a normalization check;
- exit 1 if any divergence is unexplained. The committed summary has no full-vocabulary arrays: no
  640-position `argmax` and no per-position logprobs. The only arrays are the five `(token, logprob)`
  pairs at each divergent position, for both backends (7 positions × 2 × 5 in the M2 run) — the
  evidence the report needs for the top-5 gaps, not a logits dump.

### Dry run on the pre-M2 artifacts (history, not current evidence)

```
python3 -I tools/consistency.py \
  --cuda   target/parity-result-cuda-sm70.json \
  --vulkan target/parity-result-vk-sm70.json \
  --cuda-tag cuda-sm70 --vulkan-tag vk-sm70 --status pre-m2 \
  --out docs/parity/cross-backend.json     # exit 0
```

The CUDA artifact is from the M1 build `ae95739`, the Vulkan one from M0 `d6c2beb`: this run only
proves the comparison reproduces the known 9/640 and that the registry covers it. It is **not**
evidence about the M2 build.

| backend | tag | strict | either | divergent vs the other backend |
| --- | --- | ---: | ---: | ---: |
| CUDA sm_70 | `cuda-sm70` (M1 `ae95739`) | 630/640 (98.44 %) | 636/640 (99.38 %) | 9 of 640 (1.41 %) |
| Vulkan sm_70 | `vk-sm70` (M0 `d6c2beb`) | 625/640 (97.66 %) | 629/640 (98.28 %) | same positions |

- CUDA matches the reference and Vulkan does not: 7 positions.
- Vulkan matches the reference and CUDA does not: 2 positions (both are also the one-batch
  llama.cpp token: a near-tie where llama.cpp's own two modes disagree).
- both miss the reference: 0. unexplained: 0.
- top-5 gaps: `n/a` in these artifacts, they predate the `top5` field; the M2 run fills them.

### Divergent positions (pre-M2 dry run)

| # | prompt | pos | CUDA | Vulkan | reference | llama one-batch | explained by |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 12 | A train leaves at 9:15 and travels 180 km at 72 km | 5 | 314 | 11 | 11 | 314 | `either-mode-ambiguity` |
| 16 | What license terms apply here? Summarize: | 0 | 27775 | 760 | 27775 | 27775 | `gemm-int8-vs-f16` |
| 16 | What license terms apply here? Summarize: | 23 | 3665 | 4530 | 3665 | 3665 | `gemm-int8-vs-f16` |
| 17 | Find the error in this log: | 8 | 29269 | 12662 | 29269 | 29269 | `gemm-int8-vs-f16` |
| 17 | Find the error in this log: | 13 | 8240 | 332 | 332 | 8240 | `either-mode-ambiguity` |
| 17 | Find the error in this log: | 16 | 4490 | 1412 | 4490 | 4490 | `gemm-int8-vs-f16` |
| 18 | Which of these cities is the largest by population | 30 | 364 | 12041 | 364 | 364 | `gemm-int8-vs-f16` |
| 19 | List the keys whose value is false: | 0 | 27775 | 760 | 27775 | 27775 | `gemm-int8-vs-f16` |
| 19 | List the keys whose value is false: | 3 | 4566 | 3766 | 4566 | 4566 | `gemm-int8-vs-f16` |

This is the list first summarized from prose in the M1-T8 block above, now machine-generated. Seven
positions are Vulkan-only misses that CUDA resolves; two are positions where llama.cpp's own
token-by-token and one-batch runs disagree and the backends split across the tie.

### End-to-end on the M2 build

Both backends re-run on this build (`8200a75`, `v100` + this PR's tools/docs only) with the same
cached reference (`target/parity-ref.json`, sha256 `0c30267c…`, unchanged; `target/llama-batch-argmax.json`,
sha256 `ba5e28ac…`):

```
python3 -I tools/parity.py "$M" --extra "--gpu --backend cuda   --llama-numerics --fixed-experts" --tag m2-cuda-sm70
python3 -I tools/parity.py "$M" --extra "--gpu --backend vulkan --llama-numerics --fixed-experts" --tag m2-vk-sm70
STRATA_MMQ=      python3 -I tools/parity.py "$M" --extra "--gpu --backend vulkan --llama-numerics --fixed-experts" --tag m2-vk-sm70-f16
STRATA_GEMM_FMA=1 python3 -I tools/parity.py "$M" --extra "--gpu --backend cuda   --llama-numerics --fixed-experts" --tag m2-cuda-sm70-fma
```

CUDA takes **628/640 strict, 634/640 either** (98.12 % / 99.06 %): **2 positions below the M1
figures** (630/636, `ae95739`). Vulkan takes **625/640 strict, 629/640 either** (97.66 % /
98.28 %), bit-identical to the M0 baseline (`d6c2beb`), exit 1 as `tools/parity.py:140` designs for
Vulkan. The CUDA drop is the M2 tensor-core path and is isolated by the arms below; it is a measured
regression against the M1 parity record and is reported here, not hidden. M2-T8 is gated on this
report and must weigh it.

| backend | tag | strict | either | divergent vs the other backend |
| --- | --- | ---: | ---: | ---: |
| CUDA sm_70 | `m2-cuda-sm70` | 628/640 (98.12 %) | 634/640 (99.06 %) | 7 of 640 (1.09 %) |
| Vulkan sm_70 | `m2-vk-sm70` | 625/640 (97.66 %) | 629/640 (98.28 %) | same positions |

- CUDA matches the reference and Vulkan does not: 5
- Vulkan matches the reference and CUDA does not: 2
- both miss the reference: 0
- unexplained: 0

### Divergent positions

| # | prompt | pos | CUDA | Vulkan | reference | alt | explained by | CUDA top-5 gap | Vulkan top-5 gap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 12 | A train leaves at 9:15 and travels 180 km at 72 km | 5 | 314 | 11 | 11 | 314 | `either-mode-ambiguity` | 0.08893 | 0.2579 |
| 16 | What license terms apply here? Summarize: | 0 | 27775 | 760 | 27775 | 27775 | `gemm-int8-vs-f16` | 0.3123 | 0.0319 |
| 16 | What license terms apply here? Summarize: | 23 | 3665 | 4530 | 3665 | 3665 | `gemm-int8-vs-f16` | 0.5384 | 0.4321 |
| 17 | Find the error in this log: | 13 | 8240 | 332 | 332 | 8240 | `either-mode-ambiguity` | 0.03493 | 0.02267 |
| 17 | Find the error in this log: | 16 | 4490 | 1412 | 4490 | 4490 | `no-coop-matrices` | 0.2655 | 0.2728 |
| 18 | Which of these cities is the largest by population | 30 | 364 | 12041 | 364 | 364 | `gemm-int8-vs-f16` | 0.249 | 0.1016 |
| 19 | List the keys whose value is false: | 3 | 4566 | 3766 | 4566 | 4566 | `gemm-int8-vs-f16` | 0.2722 | 0.2307 |

### Attribution arms

| arm | file | strict | either | divergent vs CUDA | base divergences resolved | still divergent |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `f16` | `target/parity-result-m2-vk-sm70-f16.json` | 629/640 | 635/640 | 3 | 6 | 3 |
| `cuda-fma` | `target/parity-result-m2-cuda-sm70-fma.json` | 630/640 | 636/640 | 2 | 7 | 2 |

**Attribution**

- The CUDA M2 artifact differs from the M1 artifact at exactly **2 positions**: prompt 17 pos 8
  (29269 → 12662) and prompt 19 pos 0 (27775 → 760). The `STRATA_GEMM_FMA=1` arm reproduces the M1
  artifact **bit for bit** (0 differing positions, 630/636): the move is the `mma.sync.m8n8k4`
  summation order of M2-T1/M2-T2, not a kernel error. Both positions are near-ties (CUDA top-2 gaps
  at 17/8 = 0.1714, at 19/0 = 0.0642).
- The Vulkan `STRATA_MMQ=` (empty, f16 GEMM) arm moves **8 positions** and reaches **629/640 strict,
  635/640 either** (up from 625/629). It resolves **6 of the 7** cross-backend divergences, so those
  six are the int8 GEMM path, not attention or the KV cache (M3-T5's hypothesis, now measured). It
  leaves **3** divergences:
  - prompt 17 pos 16 **persists**: the f16 Vulkan GEMM still picks 1412 where CUDA and the reference
    take 4490. The arm refutes `gemm-int8-vs-f16` for this position: it is the same-math reduction
    order of a f16 GEMM without cooperative matrices, so it is attributed to `no-coop-matrices`
    (M3-T5) instead.
  - prompt 17 pos 8 and prompt 19 pos 0 become divergences **in the opposite direction**: the f16
    Vulkan GEMM is on the reference there and the M2 CUDA tensor-core path is not (the two positions
    isolated by the FMA arm above).
- Where the table below shows `0 bits`, the kernels are bit-identical across backends (decode
  attention, DeltaNet, KV store, dequant, GEMV); the end-to-end differences are GEMM arithmetic only.
  MoE combine, conv, router top-k, norm/rope/gate and swiglu/add are assert-only in the tests, so
  they carry measurement-only lines added by this PR (see the per-op table): they are bit-identical
  across backends except one bit of `ssm_gates_g` (see the highlights).

### Per-op table (M2 build)

`STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda,vulkan cargo test --release --test kernels -- --nocapture
--test-threads=1` (23 passed, 4 ignored, 11.97 s), the same for `llama_numerics` (1 passed, 48.29 s)
and `kv_q8` (2 passed, 4.03 s); `grep -c "SKIP: cuda"` = 0, `grep -c "SKIP: model file missing"` = 0
in both logs, `per_op_unparsed` = 0.

The assert-only tests now print measurement lines (`op <op> <backend> [<case>]: rel err vs CPU …,
<n> of <m> differ from the first backend`), added by this PR without changing an assertion: MoE GEMV,
conv (`conv_seq`, `conv_state`, `conv_silu+l2_norm`), MoE combine, `rms_norm`/`l2_norm`/`rope_neox`/
`norm_gate`/`ssm_gates`, router top-k and `swiglu`/`add` now have rows for both backends.

Caveat: in a single kernels process Vulkan stops enumerating after ~20 backend instantiations
(`no usable Vulkan device 0 (… llvmpipe …)`). `router_topk_matches_ops_top_k`,
`ssm_gates_match_across_backends` and `vulkan_type_matrix` therefore ran CUDA-only in the full log and
were rerun in a fresh process (`cargo test --test kernels -- --nocapture --test-threads=1 router_topk
ssm_gates_match_across_backends`, 0.69 s): `router_topk` now has both backends, `ssm_gates` prints no
`op` line (it is still assert-only, cross-backend bit equality), and `vulkan_type_matrix` is M0-T5
territory and not re-recorded here. No `Xid`, GPU 0 MiB.

**add**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `n 1000, not a block multiple` | L2 6.03e-09 | L2 6.03e-09 | 0 bits | — |

**attn_decode**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `pos 0 nh 24 nkv 2 hd 256 gq_min 0` | L2 8.69e-08 | L2 8.69e-08 | 0 bits | `same-kernel (bit-exact)` |
| `pos 0 nh 24 nkv 2 hd 256 gq_min 2048` | L2 8.69e-08 | L2 8.69e-08 | 0 bits | `same-kernel (bit-exact)` |
| `pos 1000 nh 16 nkv 2 hd 256 gq_min 0` | L2 1.28e-07 | L2 1.28e-07 | 0 bits | `same-kernel (bit-exact)` |
| `pos 1000 nh 16 nkv 2 hd 256 gq_min 2048` | L2 1.09e-07 | L2 1.09e-07 | 0 bits | `same-kernel (bit-exact)` |
| `pos 2100 nh 32 nkv 8 hd 128 gq_min 0` | L2 1.38e-07 | L2 1.38e-07 | 0 bits | `same-kernel (bit-exact)` |
| `pos 2100 nh 32 nkv 8 hd 128 gq_min 2048` | L2 1.38e-07 | L2 1.38e-07 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 16 nkv 2 hd 256 gq_min 0` | L2 9.74e-08 | L2 9.74e-08 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 16 nkv 2 hd 256 gq_min 2048` | L2 9.74e-08 | L2 9.74e-08 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 32 nkv 8 hd 128 gq_min 0` | L2 9.65e-08 | L2 9.65e-08 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 32 nkv 8 hd 128 gq_min 2048` | L2 9.65e-08 | L2 9.65e-08 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 4 nkv 2 hd 128 gq_min 0` | L2 1.04e-07 | L2 1.04e-07 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 4 nkv 2 hd 128 gq_min 2048` | L2 1.04e-07 | L2 1.04e-07 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 4 nkv 2 hd 256 gq_min 0` | L2 9.91e-08 | L2 9.91e-08 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 4 nkv 2 hd 256 gq_min 2048` | L2 9.91e-08 | L2 9.91e-08 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 4 nkv 2 hd 64 gq_min 0` | L2 9.56e-08 | L2 9.56e-08 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 4 nkv 2 hd 64 gq_min 2048` | L2 9.56e-08 | L2 9.56e-08 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 8 nkv 8 hd 128 gq_min 0` | L2 1.06e-07 | L2 1.06e-07 | 0 bits | `same-kernel (bit-exact)` |
| `pos 5 nh 8 nkv 8 hd 128 gq_min 2048` | L2 1.06e-07 | L2 1.06e-07 | 0 bits | `same-kernel (bit-exact)` |

**attn_prefill**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `pos 0 nh 24 nkv 2 hd 256` | n/a | n/a | L2 6.14e-05 | `summation-order` |
| `pos 1000 nh 16 nkv 2 hd 256` | n/a | n/a | L2 3.52e-05 | `summation-order` |
| `pos 2100 nh 32 nkv 8 hd 128` | n/a | n/a | L2 3.02e-05 | `summation-order` |
| `pos 5 nh 16 nkv 2 hd 256` | n/a | n/a | L2 6.11e-05 | `summation-order` |
| `pos 5 nh 32 nkv 8 hd 128` | n/a | n/a | L2 6.24e-05 | `summation-order` |
| `pos 5 nh 4 nkv 2 hd 128` | n/a | n/a | L2 7.26e-05 | `summation-order` |
| `pos 5 nh 4 nkv 2 hd 256` | n/a | n/a | L2 6.37e-05 | `summation-order` |
| `pos 5 nh 4 nkv 2 hd 64` | n/a | n/a | L2 6.42e-05 | `summation-order` |
| `pos 5 nh 8 nkv 8 hd 128` | n/a | n/a | L2 6.13e-05 | `summation-order` |

**attn_prefill_vs_decode**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `hd 128 nh 32 nkv 8 gq_min 0` | L2 2.98e-05 | L2 1.36e-06 | n/a | — |
| `hd 128 nh 32 nkv 8 gq_min 2048` | L2 2.98e-05 | L2 1.36e-06 | n/a | — |
| `hd 128 nh 4 nkv 2 gq_min 0` | L2 7.26e-05 | L2 2.45e-07 | n/a | — |
| `hd 128 nh 4 nkv 2 gq_min 2048` | L2 7.26e-05 | L2 2.45e-07 | n/a | — |
| `hd 128 nh 8 nkv 8 gq_min 0` | L2 6.13e-05 | L2 2.68e-07 | n/a | — |
| `hd 128 nh 8 nkv 8 gq_min 2048` | L2 6.13e-05 | L2 2.68e-07 | n/a | — |
| `hd 256 nh 16 nkv 2 gq_min 0` | L2 3.51e-05 | L2 4.42e-07 | n/a | — |
| `hd 256 nh 16 nkv 2 gq_min 2048` | L2 3.51e-05 | L2 4.28e-07 | n/a | — |
| `hd 256 nh 24 nkv 2 gq_min 0` | L2 6.14e-05 | L2 1.21e-07 | n/a | — |
| `hd 256 nh 24 nkv 2 gq_min 2048` | L2 6.14e-05 | L2 1.21e-07 | n/a | — |
| `hd 256 nh 4 nkv 2 gq_min 0` | L2 6.37e-05 | L2 1.95e-07 | n/a | — |
| `hd 256 nh 4 nkv 2 gq_min 2048` | L2 6.37e-05 | L2 1.95e-07 | n/a | — |
| `hd 64 nh 4 nkv 2 gq_min 0` | L2 6.42e-05 | L2 2.22e-07 | n/a | — |
| `hd 64 nh 4 nkv 2 gq_min 2048` | L2 6.42e-05 | L2 2.22e-07 | n/a | — |

**conv**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `ch 300 kw 4, 2 steps` | L2 5.49e-08 | L2 5.49e-08 | 0 bits | `summation-order` |

**conv_l2_norm**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `ch 424 kw 4` | n/a | n/a | 0 bits | `summation-order` |

**conv_seq**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `T 10 ch 96 kw 4` | n/a | n/a | 0 bits | `summation-order` |

**conv_state**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `ch 300 kw 4, 2 steps` | L2 0.00e+00 | L2 0.00e+00 | 0 bits | `summation-order` |

**deltanet**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `ilv false` | L2 1.96e-07 | L2 1.96e-07 | 0 bits | `same-kernel (bit-exact)` |
| `ilv true` | L2 2.02e-07 | L2 2.02e-07 | 0 bits | `same-kernel (bit-exact)` |

**dequant+gemv**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `F32 blk.0.ffn_gate_inp.weight` | L2 7.17e-07 / max 0.00e+00 | L2 7.17e-07 / max 0.00e+00 | n/a | — |
| `IQ2_XS blk.0.ffn_gate_exps.weight` | L2 8.18e-07 / max 0.00e+00 | L2 8.18e-07 / max 0.00e+00 | n/a | — |
| `IQ3_XXS blk.0.ffn_down_exps.weight` | L2 4.13e-07 / max 0.00e+00 | L2 4.13e-07 / max 0.00e+00 | n/a | — |
| `IQ4_XS blk.1.ffn_down_exps.weight` | L2 4.12e-07 / max 0.00e+00 | L2 4.12e-07 / max 0.00e+00 | n/a | — |
| `Q4_K output.weight` | L2 8.16e-07 / max 0.00e+00 | L2 8.16e-07 / max 0.00e+00 | n/a | — |
| `Q5_K blk.0.attn_gate.weight` | L2 8.36e-07 / max 0.00e+00 | L2 8.36e-07 / max 0.00e+00 | n/a | — |
| `Q6_K blk.0.ffn_down_shexp.weight` | L2 4.19e-07 / max 0.00e+00 | L2 4.19e-07 / max 0.00e+00 | n/a | — |
| `Q8_0 blk.1.ffn_down_shexp.weight` | L2 4.19e-07 / max 0.00e+00 | L2 4.19e-07 / max 0.00e+00 | n/a | — |

**gemm**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `F32 blk.0.ffn_gate_inp.weight` | L2 4.87e-07 | L2 0.00e+00 | n/a | — |
| `F32 blk.0.ffn_gate_inp.weight [different f16 input conversion]` | n/a | n/a | L2 4.79e-05 | `f16-round-vs-trunc` |
| `IQ2_XS blk.0.ffn_gate_exps.weight` | L2 8.95e-07 | L2 0.00e+00 | n/a | — |
| `IQ2_XS blk.0.ffn_gate_exps.weight [different f16 input conversion]` | n/a | n/a | L2 5.41e-04 | `f16-round-vs-trunc` |
| `IQ3_XXS blk.0.ffn_down_exps.weight` | L2 6.31e-07 | L2 4.02e-07 | n/a | — |
| `IQ3_XXS blk.0.ffn_down_exps.weight [different GEMM math]` | n/a | n/a | L2 3.51e-03 | `gemm-int8-vs-f16` |
| `IQ4_XS blk.1.ffn_down_exps.weight` | L2 6.42e-07 | L2 4.39e-07 | n/a | — |
| `IQ4_XS blk.1.ffn_down_exps.weight [different GEMM math]` | n/a | n/a | L2 4.08e-03 | `gemm-int8-vs-f16` |
| `Q4_K output.weight` | L2 9.13e-07 | L2 7.71e-07 | n/a | — |
| `Q4_K output.weight [different GEMM math]` | n/a | n/a | L2 3.17e-03 | `gemm-int8-vs-f16` |
| `Q5_K blk.0.attn_gate.weight` | L2 9.58e-07 | L2 8.16e-07 | n/a | — |
| `Q5_K blk.0.attn_gate.weight [different GEMM math]` | n/a | n/a | L2 3.32e-03 | `gemm-int8-vs-f16` |
| `Q6_K blk.0.ffn_down_shexp.weight` | L2 6.47e-07 | L2 4.12e-07 | n/a | — |
| `Q6_K blk.0.ffn_down_shexp.weight [different GEMM math]` | n/a | n/a | L2 3.78e-03 | `gemm-int8-vs-f16` |
| `Q8_0 blk.1.ffn_down_shexp.weight` | L2 6.01e-07 | L2 4.12e-07 | n/a | — |
| `Q8_0 blk.1.ffn_down_shexp.weight [different GEMM math]` | n/a | n/a | L2 3.60e-03 | `gemm-int8-vs-f16` |

**gemm_dense**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `dense 4096` | L2 5.56e-06 | L2 6.92e-07 | n/a | — |

**l2_norm**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `128 x 2 per token, 3 tokens` | L2 5.89e-09 | L2 5.89e-09 | 0 bits | — |

**moe_combine**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `shared false` | L2 3.71e-08 | L2 3.71e-08 | 0 bits | `expert-slot-layout` |
| `shared true` | L2 4.45e-08 | L2 4.45e-08 | 0 bits | `expert-slot-layout` |

**moe_gemv**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `3 experts 512x2048 IQ2_XS` | L2 8.86e-07 | L2 8.86e-07 | 0 bits | `expert-slot-layout` |

**norm_gate**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `4 heads of 128` | L2 4.31e-08 | L2 4.31e-08 | 0 bits | — |

**rms_norm**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `2048 x 3 rows` | L2 4.30e-07 | L2 4.30e-07 | 0 bits | — |

**rope_neox**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `hd 128, 64 rotated, pos 30000` | L2 2.52e-08 | L2 2.52e-08 | 0 bits | — |

**router_topk**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `n 256 k 8 t 3` | max 1.19e-07 | max 1.19e-07 | 0 bits | — |

**ssm_gates**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `beta, 3 tokens of 5 heads` | L2 3.00e-08 | L2 3.00e-08 | 0 bits | `summation-order` |

**ssm_gates_g**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `g, 3 tokens of 5 heads` | L2 1.44e-10 | L2 1.44e-10 | 1 bits | `summation-order` |

**swiglu**

| case / shape | CUDA vs CPU | Vulkan vs CPU | CUDA vs Vulkan | explained by |
| --- | ---: | ---: | ---: | --- |
| `n 1000, not a block multiple` | L2 4.24e-09 | L2 4.24e-09 | 0 bits | — |

**Other measurements** (comparisons that are not CUDA/Vulkan vs the CPU reference)

| op | case / shape | comparison | value |
| --- | --- | --- | --- |
| attn_decode_llama | `n_pos 1400 gq_min 0` | llama_vs_cpu | L2 1.45e-05 |
| attn_decode_llama | `n_pos 1400 gq_min 0` | default_vs_cpu | L2 6.23e-04 |
| attn_decode_llama | `n_pos 1400 gq_min 2048` | llama_vs_cpu | L2 1.45e-05 |
| attn_decode_llama | `n_pos 1400 gq_min 2048` | default_vs_cpu | L2 6.23e-04 |
| attn_decode_llama | `n_pos 300 gq_min 0` | llama_vs_cpu | L2 2.94e-05 |
| attn_decode_llama | `n_pos 300 gq_min 0` | default_vs_cpu | L2 6.19e-04 |
| attn_decode_llama | `n_pos 300 gq_min 2048` | llama_vs_cpu | L2 2.94e-05 |
| attn_decode_llama | `n_pos 300 gq_min 2048` | default_vs_cpu | L2 6.19e-04 |
| attn_decode_llama | `n_pos 511 gq_min 0` | llama_vs_cpu | L2 2.96e-05 |
| attn_decode_llama | `n_pos 511 gq_min 0` | default_vs_cpu | L2 6.22e-04 |
| attn_decode_llama | `n_pos 511 gq_min 2048` | llama_vs_cpu | L2 2.96e-05 |
| attn_decode_llama | `n_pos 511 gq_min 2048` | default_vs_cpu | L2 6.22e-04 |
| attn_decode_llama | `n_pos 512 gq_min 0` | llama_vs_cpu | L2 2.96e-05 |
| attn_decode_llama | `n_pos 512 gq_min 0` | default_vs_cpu | L2 6.23e-04 |
| attn_decode_llama | `n_pos 512 gq_min 2048` | llama_vs_cpu | L2 2.96e-05 |
| attn_decode_llama | `n_pos 512 gq_min 2048` | default_vs_cpu | L2 6.23e-04 |
| attn_decode_llama | `n_pos 7 gq_min 0` | llama_vs_cpu | L2 5.10e-05 |
| attn_decode_llama | `n_pos 7 gq_min 0` | default_vs_cpu | L2 3.32e-04 |
| attn_decode_llama | `n_pos 7 gq_min 2048` | llama_vs_cpu | L2 5.10e-05 |
| attn_decode_llama | `n_pos 7 gq_min 2048` | default_vs_cpu | L2 3.32e-04 |
| attn_prefill_llama | `hd 128` | llama_vs_cpu | L2 2.39e-07 |
| attn_prefill_llama | `hd 128` | default_vs_cpu | L2 7.26e-05 |
| attn_prefill_llama | `hd 256` | llama_vs_cpu | L2 1.73e-07 |
| attn_prefill_llama | `hd 256` | default_vs_cpu | L2 6.11e-05 |
| attn_prefill_llama | `hd 64` | llama_vs_cpu | L2 2.18e-07 |
| attn_prefill_llama | `hd 64` | default_vs_cpu | L2 6.42e-05 |

### Per-op highlights

- dequant is **bit-exact** on every type on both backends (`max 0.00e+00` against the CPU reference);
  GEMV vs CPU is **4.12e-7 … 8.36e-7** on both backends (identical figures: the GEMV outputs are the
  same bits), GEMM vs CPU is 4.0e-7 … 9.6e-7.
- GEMM across backends: the Q4_K … IQ4_XS int8-vs-f16 rows are 3.17e-3 … 4.08e-3 and the
  f16-conversion rows 4.79e-5 (F32) and 5.41e-4 (IQ2_XS) — the pre-M2 registry magnitudes,
  reproduced on the M2 build. No `[same math]` row is produced in this window: for every type the
  two backends either take different GEMM math (int8 vs f16) or different f16 input conversions, so
  the test's same-math branch is never entered.
- attention: decode is **bit-identical** across backends (0 differing bits, 18 shapes, rel err vs
  CPU 9.6e-8 … 1.4e-7); prefill is 3.0e-5 … 7.3e-5 cross-backend (tensor cores vs no cooperative
  matrices), prefill-vs-decode 1.2e-7 … 4.4e-7 on Vulkan and 3.0e-5 … 7.3e-5 on CUDA — i.e. the M2
  tensor-core prefill is what moves, the Vulkan f16 prefill stays closer to its own decode.
- `--llama-numerics` attention: decode vs the CPU model 1.45e-5 … 5.10e-5 (llama mode) against
  3.32e-4 … 6.23e-4 for the default kernel, prefill 1.73e-7 … 2.39e-7 against 6.11e-5 … 7.26e-5;
  Vulkan is bit-identical to CUDA in llama mode (asserted by the test).
- q8 KV: 1.55e-3 … 2.06e-3 q8 vs f16 decode, 5.98e-5 … 7.67e-5 q8 prefill vs decode (CUDA; the
  test binary does not open Vulkan).
- MoE GEMV (IQ2_XS, 3 experts of 512×2048): 8.86e-7 vs CPU on both backends, **0 differing bits**
  across backends. MoE combine (1500 values, shared expert on and off): 3.71e-8 / 4.45e-8 vs CPU,
  0 bits.
- conv: `conv_silu` 5.49e-8 vs CPU, `conv_state` **0.00e+00** (bit-exact, the state is just a shift),
  `conv_seq` chunk-vs-steps and `conv_silu+l2_norm` 0 differing bits across backends. Router top-k:
  max |Δweight| 1.19e-7, ids identical, 0 differing weight bits.
- norm/rope/gate/elementwise vs CPU: `rms_norm` 4.30e-7, `l2_norm` 5.89e-9, `rope_neox` 2.52e-8,
  `norm_gate` 4.31e-8, `swiglu` 4.24e-9, `add` 6.03e-9, `ssm_gates` 3.00e-8; all 0 differing bits
  across backends except `ssm_gates_g`: 1 of 15 bits differs on Vulkan at rel err 1.44e-10 (one ULP
  in `softplus`/`expf` on this driver). The dedicated `ssm_gates_match_across_backends` test (4000
  values, different shapes) still asserts and passes bit equality, so this is a shape-dependent
  last-bit difference the test does not cover — recorded here as a measured one-ULP item, not a
  failure.

### Legitimate-difference registry

| id | mechanism | measured (M2 build; the window reproduces the pre-M2 magnitudes) | arm that isolates it | follow-up |
| --- | --- | --- | --- | --- |
| `gemm-int8-vs-f16` | int8-quantized GEMM (Vulkan) vs f16 tensor-core GEMM (CUDA); sm_70 CUDA has no int8 tensor cores | 3.17e-3 .. 4.08e-3 relative L2 per-op (M1-T7 block) | `STRATA_MMQ=` forces the f16 GEMM on Vulkan | M3-T5 |
| `f16-round-vs-trunc` | f16 conversion: CUDA `__float2half` rounds, this driver's `OpFConvert` truncates | 5.41e-4 (IQ2_XS), 4.79e-5 (F32) relative L2, recorded not asserted (`tests/kernels.rs:409`) | — | M3-T5 |
| `summation-order` | `mma.sync.m8n8k4` tile order vs Vulkan workgroup reduction vs FMA accumulation | M2 build: 4.0e-7 .. 9.6e-7 per backend vs the CPU reference (GEMV 4.12e-7 .. 8.36e-7, identical on both); cross-backend prefill attention 3.0e-5 .. 7.3e-5, decode attention 0 bits; the `STRATA_GEMM_FMA=1` arm moves 2 argmax positions | `STRATA_VK_NO_COOP=` / `STRATA_GEMM_FMA=1` | — |
| `no-coop-matrices` | Vulkan 1.4.312 on this host exposes no `VK_KHR_cooperative_matrix` | M2 build: 3.0e-5 .. 7.3e-5 relative L2 cross-backend on prefill attention, and the one end-to-end position the `STRATA_MMQ=` arm does not move (17/16) | `STRATA_VK_NO_COOP=` | M3-T5 |
| `q8-kv` | `--kv q8` stores K/V as int8 with per-32 scales; not combinable with `--llama-numerics` by design | 1.55e-3 .. 2.06e-3 relative L2 q8 vs f16; KV 341/642 = 53.1 % (M2-T6) | `--kv q8` | — |
| `expert-slot-layout` | MoE slot gather/combine order differs; parity passes `--fixed-experts` so routing is identical | not measured above the summation-order floor yet; re-recorded in the window | `--fixed-experts` | — |
| `either-mode-ambiguity` | llama.cpp's token-by-token and one-batch runs disagree here; both backends sit on a near-tie | llama.cpp self-agreement 630/640 (`docs/m6-results.md`) | — | — |

### Unexplained

None, on the pre-M2 dry run (9/640) or on the M2 build (**7/640**). Every M2 divergence maps to a
registry entry: 4 to `gemm-int8-vs-f16` (Vulkan int8 GEMM, M3-T5), 2 to `either-mode-ambiguity`
(prompt 12 pos 5, prompt 17 pos 13 — CUDA takes the one-batch llama.cpp token, Vulkan the reference
token) and 1 to `no-coop-matrices` (prompt 17 pos 16, where the `STRATA_MMQ=` f16 arm does not move
the token at all, so quantization cannot be the cause). The arm evidence sets each attribution's
`confidence`; the FMA arm confirms that the two positions lost against the M1 parity record are
tensor-core summation order, not a fault.

**Open follow-ups** (not fixed here, per the issue's out-of-scope): M3-T5 for the Vulkan int8 GEMM
gap (with the arm as evidence), and the CUDA 628/634 vs M1 630/636 tensor-core delta for M2-T8.
`tools/parity.py` writes top-5 logprobs from this PR on, and the M2 artifacts carry them; the pre-M2
ones cannot be back-filled.

### Safety and window notes

```sh
dmesg -T | grep -ic Xid     # 42 before the window and 42 after
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader   # 0 MiB, 0 % at the end
```

No `Xid`, no `device lost`, no `strata score` failure (20/20 exit 0 in each of the four parity
runs). `llama.router.service` was stopped by the coordinator before the window and is left stopped.
The full-vocabulary KLD between the backends was **not run**: the window budget went to the four
parity runs and the two test binaries. The command is in `tools/consistency.py --kld` and in the
issue; it is the one remaining optional item, recorded as not run rather than skipped silently.

### Commands (GPU window)

```sh
M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda,vulkan cargo test --release --no-fail-fast --test kernels -- --nocapture --test-threads=1
STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda,vulkan cargo test --release --no-fail-fast --test llama_numerics -- --nocapture --test-threads=1
STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda,vulkan cargo test --release --no-fail-fast --test kv_q8 -- --nocapture --test-threads=1
python3 -I tools/parity.py "$M" --extra "--gpu --backend cuda   --llama-numerics --fixed-experts" --tag m2-cuda-sm70
python3 -I tools/parity.py "$M" --extra "--gpu --backend vulkan --llama-numerics --fixed-experts" --tag m2-vk-sm70
STRATA_MMQ=       python3 -I tools/parity.py "$M" --extra "--gpu --backend vulkan --llama-numerics --fixed-experts" --tag m2-vk-sm70-f16
STRATA_GEMM_FMA=1 python3 -I tools/parity.py "$M" --extra "--gpu --backend cuda   --llama-numerics --fixed-experts" --tag m2-cuda-sm70-fma
python3 -I tools/consistency.py --cuda target/parity-result-m2-cuda-sm70.json   --vulkan target/parity-result-m2-vk-sm70.json --cuda-tag m2-cuda-sm70 --vulkan-tag m2-vk-sm70   --kernels /tmp/m2t7-kernels.txt --kernels /tmp/m2t7-llama.txt   --arm f16=target/parity-result-m2-vk-sm70-f16.json --arm cuda-fma=target/parity-result-m2-cuda-sm70-fma.json   --status m2 --date 2026-11-08 --out docs/parity/cross-backend.json   # 7 of 640, unexplained 0
```

### Commands (CPU part)

```sh
cargo build --release
cargo test --release --lib cuda::            # 21 passed
python3 -I tools/consistency.py --cuda target/parity-result-cuda-sm70.json \
  --vulkan target/parity-result-vk-sm70.json --cuda-tag cuda-sm70 --vulkan-tag vk-sm70 \
  --status pre-m2 --out docs/parity/cross-backend.json    # 9 of 640, unexplained 0, exit 0
```

## 2026-11-08 09:32 UTC — Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf

```
$ strata bench --backend cuda --pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896
cuda device 0: Tesla V100-SXM2-32GB, sm_70 (Volta, m8n8k4)
cuda shared memory: opt-in 98304 B per block, attn_decode_gqa*: 2112 B static, dynamic max 73728 B, carveout default
cuda shared memory: opt-in 98304 B per block, attn_prefill_*: 0 B static, dynamic max 72704 B, carveout default
cuda shared memory: opt-in 98304 B per block, gemvn_*: 0 B static, dynamic max 98304 B, carveout default
plan: weights+state+scratch 2560 MB (KV 642 MB for 32896 tokens, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
expert traffic: 10.4 MB per layer active set (top-8), 313.8 MB per token over 40 MoE layers
pp    512: prefill 2127.5 (2120.4-2132.9) tok/s | tg 128: decode 129.4 (128.6-129.8) tok/s
expert traffic: 313.8 MB/token -> 42.7 GB/s at 129.8 tok/s
pp   4096: prefill 2945.7 (2940.8-2949.2) tok/s | tg 128: decode 125.1 (124.8-125.3) tok/s
expert traffic: 313.8 MB/token -> 41.2 GB/s at 125.3 tok/s
pp  32768: prefill 2369.8 (2365.9-2372.9) tok/s | tg 128: decode 110.6 (110.5-110.6) tok/s
expert traffic: 313.8 MB/token -> 36.4 GB/s at 110.6 tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
worker busy 0.0 µs/token, worker stalls 0
mailbox traffic: 0 GPU writes, 0 host routing reads (fast path on, 10240/10240 experts resident)
```

## 2026-11-08 09:33 UTC — Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf

```
$ strata bench --backend vulkan --pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896
vulkan device 0: Tesla V100-SXM2-32GB
vulkan: Tesla V100-SXM2-32GB (NVIDIA 580.178.04), cooperative matrix f16 no int8 no, conditional rendering yes, maxComputeSharedMemorySize 49152
vulkan: shared-memory preflight: 60 (kernel, specialization) pairs checked, largest 49152 bytes (gemmi_dp) of 49152 allowed, 0.6 ms
plan: weights+state+scratch 2580 MB (KV 642 MB for 32896 tokens, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
expert traffic: 10.4 MB per layer active set (top-8), 313.8 MB per token over 40 MoE layers
pp    512: prefill 766.5 (761.8-769.1) tok/s | tg 128: decode 124.2 (122.3-125.3) tok/s
expert traffic: 313.8 MB/token -> 41.1 GB/s at 125.0 tok/s
pp   4096: prefill 1418.4 (1416.2-1420.3) tok/s | tg 128: decode 118.7 (118.6-118.7) tok/s
expert traffic: 313.8 MB/token -> 39.0 GB/s at 118.7 tok/s
pp  32768: prefill 521.7 (520.7-523.0) tok/s | tg 128: decode 95.8 (95.7-96.1) tok/s
expert traffic: 313.8 MB/token -> 31.5 GB/s at 95.7 tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
worker busy 106.1 µs/token, worker stalls 0
mailbox traffic: 80 GPU writes, 46080 host routing reads (fast path off, 10240/10240 experts resident)
```

## 2026-11-08 — M2 gate (M2-T8, V100, sm_70)

Window: `llama.router.service` stopped and queue drained by the coordinator,
V100 free (0 MiB, 0 %), one process per arm, CUDA first then Vulkan in the same
window (09:32 UTC and 09:33 UTC blocks above). Build: `v100` at
`e4b0477` (`task/38-cuda-gate`), driver **580.178.04**, NVRTC **12.9**
(`cuda_12.9.r12.9/compiler.36037853_0`), Vulkan **1.4.312**. Model:
`Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`, 10240/10240 experts resident, 100.0 % hit
rate, 0 cache swaps. Protocol unchanged:
`tools/bench_v100.sh` -> `strata bench --gpu --backend <b> --pp 512,4096,32768
--tg 128 --reps 3 --ctx 32896`. **Statistic: mean over the 3 repetitions, with
min-max in parentheses, exactly as the binary prints it** (`src/main.rs:583-587`);
the median is used only for the expert-traffic bandwidth line.

### Target vs measured (M2 exit criteria, CUDA)

| criterion | target | measured (mean, min-max) | delta | verdict |
| --- | ---: | --- | ---: | --- |
| prefill @4096 | >= 2000 tok/s | **2945.7** (2940.8-2949.2) | **+47.3 %** | **MET** |
| prefill @512 | - | 2127.5 (2120.4-2132.9) | - | reference |
| prefill @32768 | - | 2369.8 (2365.9-2372.9) | - | reference |
| decode @512 (reference depth) | >= 130 tok/s | **129.4** (128.6-129.8) | **-0.5 %** | **NOT MET** (see decision) |
| decode @4096 (sustained) | >= 130 tok/s | **125.1** (124.8-125.3) | **-3.8 %** | **NOT MET** (see decision) |
| decode @32768 | - | 110.6 (110.5-110.6) | - | reference |

Against M2-T4 (one window earlier, same protocol) every figure is within
-0.8..0.0 %: prefill 2127.5/2945.7/2369.8 vs 2135.7/2970.2/2386.8, decode
129.4/125.1/110.6 vs 129.5/125.2/110.6. The M2 build is reproducible across
windows; the gate did not move the numbers.

### CUDA vs Vulkan, same window, and delta vs the M0 baseline

| depth | CUDA (this window) | Vulkan (this window) | CUDA lead | Vulkan baseline | Vulkan delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill 512 | 2127.5 | 766.5 (steady ~775) | 2.78x | 635.6 (steady 775) | +20.6 % (0 % steady) |
| prefill 4096 | 2945.7 | **1418.4** | 2.08x | **1423.9** | **-0.4 %** |
| prefill 32768 | 2369.8 | 521.7 | 4.54x | 523.5 | -0.3 % |
| decode 512 | 129.4 | 124.2 | +4.2 % | 109.8 | +13.1 % |
| decode 4096 | 125.1 | 118.7 | +5.4 % | 108.6 | +9.3 % |
| decode 32768 | 110.6 | 95.8 | +15.4 % | 95.7 | +0.1 % |

The M0 Vulkan baseline is the 2026-11-06 entry at the top of this file. Vulkan
decode improved at all three depths since then (+13.1 % / +9.3 % / +0.1 %). The
one bar missed is **prefill @4096: 1418.4 vs 1423.9 (-0.4 %, 5.5 tok/s)** with
disjoint min-max (1416.2-1420.3 vs 1422.0-1427.0), i.e. a real but tiny
regression; `pp 512` first-repetition overhead is excluded by comparing the
steady-state value, as the 2026-11-06 note requires. Recorded here as
**M3-T4 evidence** and not fixed in T8 (Vulkan tuning is out of scope).

### Safety

```sh
dmesg -T | grep -ic Xid                                   # 42 before, 42 after
dmesg -T | grep -i "device lost"                          # no match
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader   # 0 MiB, 0 %
```

No `Xid`, no `device lost`, no trap during either arm; all 42 `Xid` lines are
the historical Nov 6 entries. The GPU was left at 0 MiB and
`llama.router.service` left stopped for the coordinator.

### Verdict per exit criterion

1. prefill @4096 >= 2000 tok/s — **MET**: 2945.7 tok/s (+47.3 % margin).
2. decode >= 130 tok/s — **NOT MET** at any depth: 129.4 @512 (-0.5 %),
   125.1 @4096 (-3.8 %), 110.6 @32768. **Coordinator-approved revised target**
   (recorded in the issue #38 comment): *decode >= 129 tok/s @512 and >= 125
   tok/s @4096 sustained*, pending a decode follow-up. Evidence for the
   revision is the M2-T5 ceiling: 214.2 GB/s = **23.9 % of the 898.0 GB/s
   HBM2 roofline**, **50 % occupancy** (56-96 registers/thread pin the SM to
   4x256-thread blocks), **no `cp.async` on sm_70** so the only in-flight bytes
   are 8 B per lane per super-block, and 85 % of the roofline requires a GEMV
   restructuring (more independent loads per lane, fewer per-row shuffles), not
   a launch-shape knob. If 130 tok/s is required, the coordinator opens that
   restructuring as a new task; it is not in M2/T8.
3. Vulkan not below the M0 baseline — decode **PASS** (+13.1 % / +9.3 % /
   +0.1 %); prefill @4096 **-0.4 %** vs 1423.9, stated with numbers and
   attributed to M3-T4.
4. no `Xid` — **PASS**: 42 before and after, no `device lost`.
5. M2 summary (T1..T8) present — **PASS**, below.
6. only `docs/V100-results.md` changed on the branch — **PASS**
   (`git diff --stat`: 1 file, append-only).
7. entry self-contained (commit, driver, NVRTC, backend, command, mean
   wording) — **PASS**, see the Window paragraph.

### M2 summary (T1..T8)

| task | issue / PR | headline measured result |
| --- | --- | --- |
| M2-T1 Volta tensor cores in the prefill GEMM | #24 / #25 (`3d1aa82`) | +123.6 % prefill @512, +90.7 % @4096 over the FMA fallback |
| M2-T2 Volta tensor cores in prefill attention | #26 / #27 (`0273d6e`) | prefill 2795 tok/s @4096 (+171 % vs FMA), attention kernel -91 % |
| M2-T3 shared-memory carveout and tile sizing | #28 / #29 (`a14d3e0`) | 96 KB dynamic limit validated on sm_70, occupancy/register report captured, no regression |
| M2-T4 MoE expert residency and next-layer prefetch | #34 / #35 (`64d690e`) | 100 % hit rate, 0 worker/mailbox traffic, decode +1.0-1.2 % |
| M2-T5 decode GEMV bandwidth | #30 / #31 (`b4d7408`) | 214.2 GB/s = 23.9 % of the 898.0 GB/s roofline, decode +3.2-3.3 % |
| M2-T6 CUDA flags: KV q8, parallel, MTP | #32 / #33 (`54a0c1f`) | q8 halves KV at equal speed, `--serve --parallel 4` = 1.67x, MTP k=2 +64 % on the 27B |
| M2-T7 cross-backend numerical consistency | #36 / #37 (`78dbbcd`) | CUDA 634/640 either-mode (2 near-tie positions from tensor-core summation order), Vulkan 629/640 (int8 GEMM, M3-T5) |
| M2-T8 CUDA performance gate | #38 / #39 (`07e2cba`) | prefill @4096 2945.7 tok/s (met), decode 129.4/125.1/110.6 with a coordinator-approved revised target, no `Xid`, Vulkan prefill @4096 -0.4 % (M3-T4) |

Net M2 effect, as recorded by the individual tasks: prefill @512 +123.6 % and
@4096 +90.7 % from the Volta tensor cores (T1), a further prefill gain of
+171 % vs the FMA fallback path in attention (T2), decode +3.2-3.3 % from the
GEMV rework (T5) and +1.0-1.2 % from residency (T4), with numerical parity held
at 634/640 either-mode (T7). Both arms in this window were single-process, model
fully resident, 0.0 µs/token of CPU worker cost on CUDA.

## 2026-11-08 — repeatable A/B on Model R (dense, anonymised)

The model is anonymised at the owner's request: a dense 27B-class GGUF with an
MTP head. Prompt: `tools/prompts/bench-natural.txt` (~423 tokens of natural
text), identical for both engines. Greedy (`temperature 0`), `n_predict 128`,
`ctx 4096`. V100-SXM2-32GB, driver 580.178.04; value pairs are two consecutive
runs of the same binary.

| engine | config | prefill tok/s | decode tok/s | drafts accepted |
| --- | --- | ---: | ---: | ---: |
| strata | MTP off | 408.6–409.4 | 30.50–30.53 | — |
| strata | MTP 2 | 402.5–403.4 | 37.74–37.75 | 68/122 (55.7 %) |
| strata | MTP 3 | 394.7–400.6 | 33.76–33.81 | 77/162 (47.5 %) |
| llama.cpp | no speculative | 481.3 | 37.26 | — |
| llama.cpp | draft-mtp + ngram | 411.9 | 52.64 | 67/120 (55.8 %) |

Reading: on this dense model llama.cpp leads on prefill (+18 %) and decode
(+22 % without speculative decoding, +39 % with both speculative modes at the
same acceptance rate), even though the prompt is natural text and not a
repetitive pattern. The fork's MTP closes part of the decode gap (30.5 → 37.7)
but not all of it. No `Xid` during the run; production was restarted afterwards.

## 2026-11-08 13:12 UTC — Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf

First block with the machine-readable `strata.bench.v1` record (M0-T8, #40):
`tools/bench_v100.sh` writes the fenced `json` block below and appends the same
line to `target/v100-bench.jsonl`.

```
$ strata bench --backend vulkan --pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896 --json-out /tmp/strata-bench.DTGAKE.json
vulkan device 0: Tesla V100-SXM2-32GB
vulkan: Tesla V100-SXM2-32GB (NVIDIA 580.178.04), cooperative matrix f16 no int8 no, conditional rendering yes, maxComputeSharedMemorySize 49152
vulkan: shared-memory preflight: 60 (kernel, specialization) pairs checked, largest 49152 bytes (gemmi_dp) of 49152 allowed, 0.7 ms
plan: weights+state+scratch 2580 MB (KV 642 MB for 32896 tokens, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
expert traffic: 10.4 MB per layer active set (top-8), 313.8 MB per token over 40 MoE layers
pp    512: prefill 773.6 (770.8-778.0) tok/s | tg 128: decode 123.3 (122.3-124.0) tok/s
expert traffic: 313.8 MB/token -> 40.7 GB/s at 123.7 tok/s
pp   4096: prefill 1425.2 (1422.6-1430.1) tok/s | tg 128: decode 117.6 (116.8-118.0) tok/s
expert traffic: 313.8 MB/token -> 38.8 GB/s at 117.9 tok/s
pp  32768: prefill 522.0 (520.6-523.8) tok/s | tg 128: decode 95.7 (95.6-95.7) tok/s
expert traffic: 313.8 MB/token -> 31.5 GB/s at 95.6 tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
worker busy 105.2 µs/token, worker stalls 0
mailbox traffic: 80 GPU writes, 46080 host routing reads (fast path off, 10240/10240 experts resident)
```

```json
{"backend":"vulkan","backend_arg":"vulkan","chunk":4096,"ctx":32896,"device":[0],"expert":{"bytes_per_token":328990720,"hit_rate_pct":100.0,"mb_per_token":313.8,"misses":0,"resident":10240,"swaps":0,"total":10240},"kv":"f16","model":{"name":"Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf","path":"/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"},"mtp":0,"reps":3,"results":[{"decode_tok_s":{"max":124.0,"mean":123.3,"min":122.3},"expert":{"at_decode_tok_s":123.7,"bytes_per_token":328990720,"gb_per_s":40.7,"mb_per_token":313.8},"pp":512,"prefill_tok_s":{"max":778.0,"mean":773.6,"min":770.8}},{"decode_tok_s":{"max":118.0,"mean":117.6,"min":116.8},"expert":{"at_decode_tok_s":117.9,"bytes_per_token":328990720,"gb_per_s":38.8,"mb_per_token":313.8},"pp":4096,"prefill_tok_s":{"max":1430.1,"mean":1425.2,"min":1422.6}},{"decode_tok_s":{"max":95.7,"mean":95.7,"min":95.6},"expert":{"at_decode_tok_s":95.6,"bytes_per_token":328990720,"gb_per_s":31.5,"mb_per_token":313.8},"pp":32768,"prefill_tok_s":{"max":523.8,"mean":522.0,"min":520.6}}],"schema":"strata.bench.v1","tg":128,"unix_time":1794143766}
```

## 2026-11-08 14:54 UTC — M0-T6 Vulkan flag/knob matrix (Vulkan, V100)

Backend `vulkan`, device 0, reference MoE Q2_K_XL (12.3 GB) as `$M`. Each bench row is
a fresh process; the JSON records live under `target/m0t6/` and are not committed.

| row | setting | status | numbers | message/exit code |
| --- | --- | --- | --- | --- |
| C0 | control (no knob) | supported | pp 4096: prefill 1424.9 (1414.3-1435.5) tok/s, decode 117.2 (116.2-118.2) tok/s; KV 642 MB; experts 10040 MB (10240/10240) | exit 0 |
| C1 | short chunk 512 | supported | pp 512: prefill 772.1 (769.3-774.9) tok/s, decode 122.6 (122.0-123.2) tok/s; pp 4096: prefill 748.9 (748.1-749.8) tok/s, decode 117.4 (117.1-117.6) tok/s; KV 642 MB; experts 10040 MB (10240/10240) | exit 0 |
| K1 | q8 KV | supported | pp 4096: prefill 1020.7 (1016.0-1025.3) tok/s, decode 117.0 (116.1-118.0) tok/s; KV 341 MB; experts 10040 MB (10240/10240) | exit 0 |
| K2 | q8 KV + llama numerics | rejected (Error: --kv q8 does not combine with --llama-numerics (its attention reads f16 KV)) | — | exit 1; Error: --kv q8 does not combine with --llama-numerics (its attention reads f16 KV) |
| K3 | llama numerics alone | supported | pp 512: prefill 1119.0 (1119.0-1119.0) tok/s, decode 106.0 (106.0-106.0) tok/s; KV 642 MB; experts 10040 MB (10240/10240) | exit 0 |
| N1 | coop disabled (STRATA_VK_NO_COOP=1) | supported | pp 4096: prefill 1425.0 (1417.8-1432.1) tok/s, decode 116.9 (115.7-118.1) tok/s; KV 642 MB; experts 10040 MB (10240/10240) | exit 0 |
| N2 | graphs disabled (STRATA_NO_GRAPH=1) | supported | pp 4096: prefill 1424.1 (1419.2-1429.0) tok/s, decode 90.1 (89.9-90.4) tok/s; KV 642 MB; experts 10040 MB (10240/10240) | exit 0 |
| M1 | MTP on MoE (no MTP head) | supported (caveat: --mtp 1: this model has no MTP head: decoding without drafts) | pp 512: prefill 771.8 (771.8-771.8) tok/s, decode 122.0 (122.0-122.0) tok/s; KV 642 MB; experts 10040 MB (10240/10240) | exit 0; --mtp 1: this model has no MTP head: decoding without drafts |
| P2M | parallel + MTP | degraded (--parallel 2 and --mtp 1: not supported together, decoding without MTP drafts) | KV 160 MB; experts 10040 MB (10240/10240) | exit 0 (stopped with SIGTERM); --parallel 2 and --mtp 1: not supported together, decoding without MTP drafts |

### Plan lines

```
C0: plan: weights+state+scratch 2580 MB (KV 642 MB for 32896 tokens, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
C1: plan: weights+state+scratch 2187 MB (KV 642 MB for 32896 tokens, prefill 118 MB), expert slots 10040 MB, resident experts 10240/10240
K1: plan: weights+state+scratch 2280 MB (KV 341 MB for 32896 tokens, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
K3: plan: weights+state+scratch 2580 MB (KV 642 MB for 32896 tokens, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
N1: plan: weights+state+scratch 2580 MB (KV 642 MB for 32896 tokens, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
N2: plan: weights+state+scratch 2580 MB (KV 642 MB for 32896 tokens, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
M1: plan: weights+state+scratch 2580 MB (KV 642 MB for 32896 tokens, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
P2M: plan: weights+state+scratch 2449 MB (KV 160 MB for 8192 tokens in each of 2 slots, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
```

### JSON records

- `target/m0t6/C0.json` — valid JSON (`strata.bench.v1`), key fields in the table above.
- `target/m0t6/C1.json` — valid JSON (`strata.bench.v1`), key fields in the table above.
- `target/m0t6/K1.json` — valid JSON (`strata.bench.v1`), key fields in the table above.
- `target/m0t6/K3.json` — valid JSON (`strata.bench.v1`), key fields in the table above.
- `target/m0t6/N1.json` — valid JSON (`strata.bench.v1`), key fields in the table above.
- `target/m0t6/N2.json` — valid JSON (`strata.bench.v1`), key fields in the table above.
- `target/m0t6/M1.json` — valid JSON (`strata.bench.v1`), key fields in the table above.

### Drift guard

- C0 control at pp 4096: prefill 1424.9 tok/s, decode 117.2 tok/s.
- N1 (coop off) vs C0: prefill +0.0%, decode -0.3%.
- M1 (MTP downgraded) decode at pp 512 vs C0 decode at pp 4096: +4.1% (different prompt length; the prefill means are not comparable, so only decode is checked).
- N2 (graphs off) vs C0, informational: prefill -0.1%, decode -23.1%. This is the cost of re-recording every submission, not a pass/fail row.
- K1 (q8 KV) is exempt from the drift guard: the numerics differ by design.

### Notes

- K1: KV 341 MB vs 642 MB for C0. Prefill -28.4% and decode -0.2% against C0 (decode is exempt from the drift guard: numerics differ).
- C1: pp 4096 prefill is -47.4% against C0, expected with `--chunk 512` (more expert restaging per chunk); the pp 512 row keeps the short-prompt rate.
- N2: the decode drop against C0 is the `STRATA_NO_GRAPH` re-record cost.

### Output sanity (`strata generate -n 16`)

| row | status | vs C0 | text |
| --- | --- | --- | --- |
| C0 | coherent, non-empty | reference |   ```python import numpy as np import matplotlib.pyplot as plt  # |
| K1 | coherent, non-empty | no C0 reference |   ```python import os import sys import time import json |
| N2 | coherent, non-empty | identical to C0 |   ```python import numpy as np import matplotlib.pyplot as plt  # |

### Commands

```
C0: strata bench --model "$M" --gpu --backend vulkan --json-out target/m0t6/C0.json --ctx 32896 --chunk 4096 --pp 4096 --tg 128 --reps 2 ; strata generate --model "$M" --gpu --backend vulkan --ctx 4096 --chunk 4096 -n 16 --prompt "The quick brown fox jumps over the lazy dog."
C1: strata bench --model "$M" --gpu --backend vulkan --json-out target/m0t6/C1.json --ctx 32896 --chunk 512 --pp 512,4096 --tg 128 --reps 2
K1: strata bench --model "$M" --gpu --backend vulkan --json-out target/m0t6/K1.json --ctx 32896 --kv q8 --chunk 4096 --pp 4096 --tg 128 --reps 2 ; strata generate --model "$M" --gpu --backend vulkan --ctx 4096 --chunk 4096 --kv q8 -n 16 --prompt "The quick brown fox jumps over the lazy dog."
K2: strata bench --model "$M" --gpu --backend vulkan --json-out target/m0t6/K2.json --ctx 32896 --kv q8 --llama-numerics --pp 64 --tg 8
K3: strata bench --model "$M" --gpu --backend vulkan --json-out target/m0t6/K3.json --ctx 32896 --llama-numerics --pp 512 --tg 32 --reps 1
N1: STRATA_VK_NO_COOP=1 strata bench --model "$M" --gpu --backend vulkan --json-out target/m0t6/N1.json --ctx 32896 --chunk 4096 --pp 4096 --tg 128 --reps 2
N2: STRATA_NO_GRAPH=1 strata bench --model "$M" --gpu --backend vulkan --json-out target/m0t6/N2.json --ctx 32896 --chunk 4096 --pp 4096 --tg 128 --reps 2 ; STRATA_NO_GRAPH=1 strata generate --model "$M" --gpu --backend vulkan --ctx 4096 --chunk 4096 -n 16 --prompt "The quick brown fox jumps over the lazy dog."
M1: strata bench --model "$M" --gpu --backend vulkan --json-out target/m0t6/M1.json --ctx 32896 --mtp --chunk 4096 --pp 512 --tg 128 --reps 1
P2M: strata serve --model "$M" --gpu --backend vulkan --ctx 8192 --parallel 2 --addr 127.0.0.1:8201 --mtp
```

### Environment

```
name, memory.used [MiB], utilization.gpu [%]
Tesla V100-SXM2-32GB, 0 MiB, 49 %
no Xid
llama.router.service: inactive
```

## 2026-11-08 14:55 UTC — M0-T6 Vulkan flag/knob matrix (Vulkan, V100)

Backend `vulkan`, device 0, reference MoE Q2_K_XL (12.3 GB) as `$M`. Each bench row is
a fresh process; the JSON records live under `target/m0t6/` and are not committed.

| row | setting | status | numbers | message/exit code |
| --- | --- | --- | --- | --- |
| C2 | full-length chunk 32768 | supported | pp 32768: prefill 541.7 (541.7-541.7) tok/s, decode 93.4 (93.4-93.4) tok/s; KV 642 MB; experts 10040 MB (10240/10240) | exit 0 |
| P2 | 2 slots | supported | wall 1802 ms; aggregate decode 142.1 tok/s (256 tok); #1 1.781778s/128 tok, #2 1.785440s/128 tok; KV 160 MB; experts 10040 MB (10240/10240) | exit 0 (stopped with SIGTERM); HTTP 2/2 |
| P4 | 4 slots | supported | wall 2856 ms; aggregate decode 179.3 tok/s (512 tok); #1 2.826011s/128 tok, #2 2.825075s/128 tok, #3 2.826425s/128 tok, #4 2.835191s/128 tok; KV 160 MB; experts 10040 MB (10240/10240) | exit 0 (stopped with SIGTERM); HTTP 4/4 |
| M2 | MTP on the dense 27B (optional) | supported | pp 512: prefill 311.3 (311.3-311.3) tok/s, decode 40.4 (40.4-40.4) tok/s; KV 512 MB; experts 0 MB (0/0) | exit 0; MTP drafts accepted 63/65 (96.9%) |

### Plan lines

```
C2: plan: weights+state+scratch 5726 MB (KV 642 MB for 32896 tokens, prefill 3657 MB), expert slots 10040 MB, resident experts 10240/10240
P2: plan: weights+state+scratch 2449 MB (KV 160 MB for 8192 tokens in each of 2 slots, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
P4: plan: weights+state+scratch 3024 MB (KV 160 MB for 8192 tokens in each of 4 slots, prefill 512 MB), expert slots 10040 MB, resident experts 10240/10240
M2: plan: weights+state+scratch 17636 MB (KV 512 MB for 8192 tokens, prefill 706 MB), expert slots 0 MB, resident experts 0/0
```

### JSON records

- `target/m0t6/C2.json` — valid JSON (`strata.bench.v1`), key fields in the table above.
- `target/m0t6/M2.json` — valid JSON (`strata.bench.v1`), key fields in the table above.

### Drift guard

- C0 control at pp 4096: prefill 1424.9 tok/s, decode 117.2 tok/s.
- N1 (coop off) vs C0: prefill +0.0%, decode -0.3%.
- M1 (MTP downgraded) decode at pp 512 vs C0 decode at pp 4096: +4.1% (different prompt length; the prefill means are not comparable, so only decode is checked).
- N2 (graphs off) vs C0, informational: prefill -0.1%, decode -23.1%. This is the cost of re-recording every submission, not a pass/fail row.
- K1 (q8 KV) is exempt from the drift guard: the numerics differ by design.

### Notes

- C2: `--chunk 32768` fits: prefill scratch 3657 MB, KV 642 MB, expert slots unchanged.

### Output sanity (`strata generate -n 16`)

| row | status | vs C0 | text |
| --- | --- | --- | --- |
| C2 | coherent, non-empty | no C0 reference |   ```python import numpy as np import matplotlib.pyplot as plt  # |

### Commands

```
C2: strata bench --model "$M" --gpu --backend vulkan --json-out target/m0t6/C2.json --ctx 32896 --chunk 32768 --pp 32768 --tg 64 --reps 1 ; strata generate --model "$M" --gpu --backend vulkan --ctx 4096 --chunk 32768 -n 16 --prompt "The quick brown fox jumps over the lazy dog."
P2: strata serve --model "$M" --gpu --backend vulkan --ctx 8192 --parallel 2 --addr 127.0.0.1:8199
P4: strata serve --model "$M" --gpu --backend vulkan --ctx 8192 --parallel 4 --addr 127.0.0.1:8200
M2: strata bench --model "$M27B" --gpu --backend vulkan --json-out target/m0t6/M2.json --ctx 8192 --mtp --chunk 4096 --pp 512 --tg 128 --reps 1
```

### Environment

```
name, memory.used [MiB], utilization.gpu [%]
Tesla V100-SXM2-32GB, 0 MiB, 0 %
no Xid
llama.router.service: inactive
```


## 2026-11-08 — Vulkan prefill numeric failure is the int8 GEMM (M3-T5 evidence)

`tests/gpu_model.rs::prefill_matches_reference` on the reference model, Vulkan:

| GEMM path | next-step argmax (prefill) | CPU | max abs dlogit | verdict |
| --- | ---: | ---: | ---: | --- |
| default (int8 `gemmi_dp`) | 15019 | 4938 | 1.4263 | FAIL |
| `STRATA_MMQ=` (forced f16 `gemm_f_dp`) | 4938 | 4938 | 0.7616 | PASS |

This is the same root cause measured in M2-T7: the Vulkan parity gap and this
prefill failure both come from the int8 GEMM, not from attention. A targeted
f16-fallback (or an int8 accuracy fix) is the M3-T5 direction; the prefill cost
of that fallback must be measured before adopting it.

## 2026-11-08 — Cost of the f16 GEMM fallback on Vulkan (M3-T5 decision input)

Same model, Vulkan, `--pp 512,4096 --tg 32 --reps 2 --ctx 4200`, mean (min–max):

| GEMM path | pp 512 | pp 4096 | decode @4096 |
| --- | ---: | ---: | ---: |
| default (int8 `gemmi_dp`) | 776.1 (773.6–778.6) | 1431.8 (1426.3–1437.2) | 114.7 |
| `STRATA_MMQ=` (f16 `gemm_f_dp`) | 440.2 (292.0–588.4) | 1135.2 (1131.7–1138.6) | 115.5 |

A blanket f16 fallback costs **−20.7 % prefill at 4K** and −43 % at 512, which
would violate the M3 Vulkan baseline (1423.9 @4K). M3-T5 therefore needs a
per-type ablation (f16 only for the types that cause the divergences) or an
int8 accuracy fix before the parity gap can be closed at an acceptable cost.

## 2026-11-08 — M3-T1 Vulkan profile of prefill and decode (graph off)

Same model and machine (Tesla V100-SXM2-32GB, sm_70, driver 580.178.04,
Vulkan 1.4.312), commit `645891f`, `strata` v1.2.0.
`STRATA_VK_PROF` only writes timestamps outside a graph (`src/vulkan/mod.rs:754`,
`:760`), so every run below uses `STRATA_NO_GRAPH=1`; all tables are
**graph off**.

Profile A measures prefill at 4096 tokens, B prefill + 256 decode steps at
depth 4096, C a short-context decode (`pp 64 + 512 steps`, depth ≤ 576) as a
cross-check of the B − A subtraction. B − A spans **255** decode steps (A decodes
one token), which the call counts confirm: every per-token count is an exact
integer (gemv_q 251, gemv_f 140, rms_norm 101, attn_decode 10).

```
M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
STRATA_NO_GRAPH=1 STRATA_VK_PROF=1 STRATA_VK_STATS=1 \
  target/release/strata bench --model "$M" --gpu --backend vulkan \
  --pp 4096 --tg 1 --reps 1 --ctx 4400 2>target/m3t1/A_prefill.log
STRATA_NO_GRAPH=1 STRATA_VK_PROF=1 \
  target/release/strata bench --model "$M" --gpu --backend vulkan \
  --pp 4096 --tg 256 --reps 1 --ctx 4400 2>target/m3t1/B_decode.log
STRATA_NO_GRAPH=1 STRATA_VK_PROF=1 \
  target/release/strata bench --model "$M" --gpu --backend vulkan \
  --pp 64 --tg 512 --reps 1 --ctx 4400 2>target/m3t1/C_decode_short.log
# control: the same B run without the profiler, to size its own overhead
STRATA_NO_GRAPH=1 \
  target/release/strata bench --model "$M" --gpu --backend vulkan \
  --pp 4096 --tg 256 --reps 1 --ctx 4400 2>target/m3t1/B_control_noprof.log
```

### Prefill at 4096 tokens (run A)

| Kernel | self (ms) | share (%) | calls | µs/call | µs/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| **`gemmi_dp`** (int8 GEMM) | 985.7 | 35.6 | 292 | 3375.75 | 240.65 |
| **`gemm_f_dp`** (f16 GEMM) | 895.9 | 32.3 | 218 | 4109.67 | 218.73 |
| **`attn_prefill_dp`** (attention) | 641.6 | 23.1 | 10 | 64164.49 | 156.64 |
| `deltanet` | 153.9 | 5.6 | 60 | 2564.59 | 37.57 |
| `moe_combine_seq` | 20.7 | 0.7 | 40 | 517.99 | 5.05 |
| `quant_q8` | 18.2 | 0.7 | 200 | 90.86 | 4.44 |
| `conv_seq` | 13.3 | 0.5 | 30 | 444.52 | 3.25 |
| `rms_norm` | 13.2 | 0.5 | 202 | 65.44 | 3.22 |
| `norm_gate` | 10.0 | 0.4 | 60 | 166.54 | 2.44 |
| `l2_norm` | 6.5 | 0.2 | 30 | 218.28 | 1.59 |
| **`attn_decode`** (decode attention, 1 step) | 0.6 | 0.0 | 10 | 60.42 | 0.15 |
| other 16 kernels | 12.2 | 0.4 | 768 | — | 2.98 |
| **sum of kernel self time** | **2771.8** | 100.0 | 1 920 | — | **676.71** |

Round-off: shares are of the profiled kernel time, not of the wall clock
(kernel time / wall = 34.2 % / 31.1 % / 22.3 % for the three top kernels).

### Decode at depth 4096 (run B − run A, 255 steps)

| Kernel | self (ms) | share (%) | calls | µs/call | µs/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| **`gemv_q`** (int8-family GEMV) | 1026.8 | 39.3 | 64 005 | 16.04 | 4026.67 |
| `moe_gemv_q` (MoE expert GEMV) | 263.6 | 10.1 | 10 200 | 25.84 | 1033.73 |
| `gemv_f` (F32/BF16 GEMV) | 254.0 | 9.7 | 35 700 | 7.11 | 996.08 |
| `rms_norm` | 230.8 | 8.8 | 25 755 | 8.96 | 905.10 |
| `moe_down_combine` | 176.4 | 6.8 | 10 200 | 17.29 | 691.76 |
| `moe_route` | 160.0 | 6.1 | 10 200 | 15.69 | 627.45 |
| **`attn_decode`** (decode attention) | 141.9 | 5.4 | 2 550 | 55.65 | 556.47 |
| `swiglu_q` | 130.3 | 5.0 | 20 400 | 6.39 | 510.98 |
| `deltanet` | 97.9 | 3.7 | 7 650 | 12.80 | 383.92 |
| `norm_gate` | 50.2 | 1.9 | 7 650 | 6.56 | 196.86 |
| `conv_silu` | 46.1 | 1.8 | 7 650 | 6.03 | 180.78 |
| `attn_merge` (decode attention) | 24.4 | 0.9 | 2 550 | 9.57 | 95.69 |
| `argmax_rows` | 2.9 | 0.1 | 510 | 5.69 | 11.37 |
| prefill-only kernels (no call delta) | 5.5 | 0.2 | 0 | — | 21.2 |
| **sum of kernel self time** | **2610.8** | 100.0 | 205 020 | — | **10 238.4** |

The "prefill-only" row is the subtraction residual of kernels that do not run
in a decode step (`gemmi_dp` 2.5 ms, `attn_prefill_dp` 2.1 ms, `gemm_f_dp`
0.8 ms): it is timer noise, 0.25 % of the prefill total of the same kernel,
which is also the run-to-run noise of A.

Cross-check with C (`pp 64 + 512 steps`, depth ≤ 576), µs/call: gemv_q 15.86 vs
16.04, moe_gemv_q 25.95 vs 25.84, gemv_f 7.03 vs 7.11, rms_norm 8.95 vs 8.96,
moe_down_combine 17.29 vs 17.29, moe_route 15.71 vs 15.69, swiglu_q 6.28 vs
6.39, deltanet 12.85 vs 12.80, norm_gate 6.50 vs 6.56, conv_silu 6.04 vs 6.03.
Only the attention differs, because it is the one kernel whose cost depends on
the context depth: `attn_decode` 14.84 µs/call at depth ≤ 576 vs 55.65 µs/call
at 4096 (×3.75) and `attn_merge` 8.15 vs 9.57. Everything else is depth-flat,
so the subtraction is sound.

### Coverage

| Phase | kernel self time | wall time of the phase | coverage | unaccounted |
| --- | ---: | ---: | ---: | ---: |
| prefill (A, 4096 tok) | 2771.8 ms | 2879.4 ms (1422.5 tok/s) | 96.3 % | 107.6 ms (3.7 %) |
| decode, depth 4096 (B−A, 255 steps) | 2610.8 ms | 3846.7 ms (15.09 ms/step) | 67.9 % | 1235.4 ms (32.1 %) |

The profiler times only the dispatches: it excludes copies, submissions, waits,
and every host gap between dispatches (`src/vulkan/mod.rs:232-235`), so the
remainder is not GPU work that a shader change could recover. The timestamp
period is effectively 1 ns on this driver: the prefill kernel sum matches the
wall clock to 3.7 %, which rules out a scale error.

The decode remainder is mostly the profiler itself. The same graph-off run
without `STRATA_VK_PROF` decodes at 92.2 tok/s (10.85 ms/step) against
66.3 tok/s (15.09 ms/step) with it: the per-submission
`vkGetQueryPoolResults(... | WAIT)` (`src/vulkan/mod.rs:611-614`) serialises
host and GPU, which costs ≈4.2 ms/step on a phase of 10.2 ms of kernels. That
is why the decode *shares* are usable and the decode *tok/s* is not.

### Sanity of the runs (tok/s as printed)

| Run | prefill | decode |
| --- | ---: | ---: |
| A `pp 4096 --tg 1`, graph off + prof | 1422.5 | 33.0 (one step) |
| B `pp 4096 --tg 256`, graph off + prof | 1421.9 | 66.3 |
| C `pp 64 --tg 512`, graph off + prof | 181.5 | 68.0 |
| control `pp 4096 --tg 256`, graph off, **no** prof | 1433.0 | 92.2 |
| baseline for comparison (graphs on): | 1431.8 | 114.7 |

So graph off alone costs ±0.1 % prefill and −19.6 % decode here (the earlier
−23.1 % of M0-T6 N2 was measured at `ctx 32896`), and adding the profiler costs
another −0.8 % prefill and −28 % decode. The first invocation after the build
(idle GPU, clocks below boost) measured 812.3 tok/s prefill; the three runs
above were made back-to-back on a warm GPU and agree within 0.8 %.

### GEMM path per weight type

The type census of this GGUF (733 tensors, 11.4 GiB of weights) and the policy
in `Gpu::mmq` (`src/cuda/mod.rs:538`, called from `Vk::gemm_raw`,
`src/vulkan/mod.rs:819`) decide the prefill path; the shader specializations
seen in A confirm it (`wspec` passes the ggml type id, `src/vulkan/mod.rs:802`):

| Type | Tensors | Weight bytes | Share of weights | Prefill path | Why |
| --- | ---: | ---: | ---: | --- | --- |
| IQ2_XS (id 17) | 78 | 5772.0 MB | 49.3 % | `gemm_f_dp` | excluded from int8 by default (#31: its scale per 16 makes the int8 path slower) |
| IQ3_XXS (id 18) | 38 | 3724.0 MB | 31.8 % | `gemmi_dp` | `mmq` = true |
| Q5_K (id 13) | 157 | 1030.6 MB | 8.8 % | `gemmi_dp` | `mmq` = true |
| IQ4_XS (id 23) | 4 | 544.0 MB | 4.6 % | `gemmi_dp` | `mmq` = true |
| Q4_K (id 12) | 1 | 272.8 MB | 2.3 % | no GEMM (GEMV only) | `output.weight`: logits come from `gemv_q`, also during prefill |
| Q6_K (id 14) | 93 | 266.6 MB | 2.3 % | `gemmi_dp` | `mmq` = true |
| F32 (id 0) | 361 | 99.7 MB | 0.9 % | `gemm_f_dp` | `float_ty` |
| Q8_0 (id 8) | 1 | 1.1 MB | 0.0 % | `gemmi_dp` | `mmq` = true |

Measured split of the prefill GEMM (**67.9 % of the profiled prefill kernel
time**, 65.3 % of the wall clock): int8 `gemmi_dp` **985.7 ms, 35.6 %** vs f16
`gemm_f_dp` **895.9 ms, 32.3 %**. The f16 side is not a small fallback on this
model: it carries the IQ2_XS half of the weights. `gemm_f_dp` is also the slower
kernel per call (4109.67 µs vs 3375.75 µs), so both are first-class targets for
M3-T2. In decode there is no such split — `gemv_q` runs for every quantized
type including IQ2_XS (`src/vulkan/mod.rs:1062-1076`) and `gemv_f` only for
F32/BF16.

### `STRATA_VK_STATS` per-pipeline statistics

Supported by the 580 driver: one line per (kernel, specialization), 38 lines for
the 27 kernels this bench dispatches (registers, binary size, shared bytes).
Compact view (min–max over specializations):

| Kernel | Specs | Registers | Binary (B) | Shared (B) |
| --- | ---: | ---: | ---: | ---: |
| `gemmi_dp` | 5 | 163–201 | 48 896–55 168 | 41 088–49 152 |
| `gemm_f_dp` | 2 | 115 | 37 504–39 808 | 27 664 |
| `attn_prefill_dp` | 1 | 48 | 11 136 | 0 |
| `attn_decode` | 1 | 128 | 41 344 | 24 672 |
| `attn_merge` | 1 | 26 | 7 168 | 1 024 |
| `gemv_q` | 4 | 32 | 5 504–9 472 | 16 |
| `moe_gemv_q` | 2 | 32–40 | 8 960–9 984 | 16 |
| `gemv_f` | 1 | 64 | 8 320 | 0 |
| `deltanet` | 1 | 123 | 16 896 | 0 |
| `moe_route` | 1 | 64 | 17 024 | 7 184 |
| others (17) | 1–2 | 16–63 | 256–13 696 | 0–1 032 |

`gemmi_dp` is at the 49 152 B shared-memory limit for its largest
specialization (Q6_K: 168 registers, 55 168 B of binary, 49 152 B of shared);
the 41 088 B ones are Q5_K, IQ3_XXS, IQ4_XS and Q8_0. M3-T2 tuning that wants
more shared memory per workgroup has none left on Volta.

### Dominant kernels

**At 4K prefill the dominant kernel is the int8 GEMM `gemmi_dp` (35.6 %), with
the f16 GEMM `gemm_f_dp` a close second (32.3 %); at decode depth 4096 it is the
GEMV `gemv_q` (39.3 %)**, not the attention (5.4 % + 0.9 %), which only grows
with depth.

For the roadmap: M3-T2 should tune `gemmi_dp` *and* `gemm_f_dp` (they are 68 %
of prefill and `gemm_f_dp` already carries half the weights), while M3-T3
(attention) has 23.1 % of prefill in `attn_prefill_dp` (641.6 ms, 64.2 ms per
4K-position dispatch) but only 6.3 % of decode. A decode-side task that is not
on the roadmap today is the 39.3 % `gemv_q`; the SSM/hybrid kernels
(`deltanet`, `conv_*`, `norm_gate`) are 6.6 % of prefill and 7.4 % of decode.
No Xid, no device-lost during any of the runs (`nvidia-smi -q` reports no Xid
counter; the last `dmesg` Xid is the pre-fix SKEDCHECK18 of 2026-11-06).

## 2026-11-08 — Non-cooperative prefill GEMM: packed KS = 16 W rows, tile map (M3-T2, #47)

Target from the M3-T1 profile above: **`gemmi_dp`** (35.6 % of prefill, 982 ms
for 292 calls), the non-cooperative int8 GEMM the reference model actually
runs. `gemm_f_dp` (32.3 %) is not reachable here: it only runs for non-`mmq`
types (`src/vulkan/mod.rs:819`), i.e. not for this model's quantized weights.

### Variant A — drop the 16 zero words of a KS = 16 W row (adopted)

`COOP 0` reads 16 int8 per `h` while `coopMatLoad` reads 32, so the KS = 16 W
row only padded each 16 int8 with 16 zeros to keep the `h * 8` word offset.
The packed layout stores the four decoded words contiguously and reads them at
`h * KS / 4` (the same `h * 8` for KS = 32). Same words, same `h` order, same
`dotPacked4x8EXT` groups, so the int32 sums stay bit-exact; the cooperative
path keeps `PW = 36`.

| `gemmi_dp` Workgroup bytes | KS = 16 (Q6_K, IQ2_XS) | KS = 32 (Q5_K, IQ3_XXS, IQ4_XS, Q8_0) |
| --- | ---: | ---: |
| `xq` 2·128·20 words | 20 480 | 20 480 |
| `wq` 2·64·PW words | **10 240** (was 18 432) | 10 240 |
| `xd` + `xsum` + `wsc` + `wmn` | 10 240 | 10 240 |
| **total** | **40 960** (was exactly 49 152) | 40 960 |

Driver statistics (`STRATA_VK_STATS=1`, this model's types): 163–201 registers,
48 896–54 656 B binary, 41 088 B shared (the driver adds 128 B to the 40 960
accounting). Preflight: 60 `(kernel, specialization)` pairs, largest 40 960 of
49 152.

Isolated GEMM (ms, min–max of 3 runs; the pre-A commit `5a551eb` was measured
from its committed `gemmi_dp.spv` in the same session):

| tensor | pre-A | A |
| --- | ---: | ---: |
| Q5_K `attn_qkv` 8192x2048 x4096 | 8.04–8.05 | 8.02–8.12 |
| Q5_K `attn_gate` 4096x2048 x4096 | 3.96–3.97 | 3.95–3.96 |
| Q6_K `ssm_out` 2048x4096 x4096 | 4.07–4.08 | 4.07–4.08 |
| IQ2_XS `ffn_gate_exps` 512x2048 x4096 | 0.94–0.96 | 0.94–0.95 |
| IQ3_XXS `ffn_down_exps` 2048x512 x4096 | 0.52–0.53 | 0.52–0.52 |

Expert tiles (ms, min–max of 3 runs): IQ2_XS 8/32/64 positions per expert
5.84–5.86 / 5.95–5.97 / 6.05–6.07 before, 5.84–5.86 / 5.94–5.98 / 6.05–6.09
after; IQ3_XXS 2.65–2.67 / 2.94–2.95 / 3.07–3.08 before, 2.66–2.66 / 2.94–2.95 /
3.07–3.07 after. **No measurable per-kernel change**, as expected: the variant
removes shared bytes and one zero-fill pass, not traffic.

End to end (warm, `--pp 4096 --tg 128 --reps 3 --ctx 32896`, 5 measurements):
**prefill 1424.9–1432.5 tok/s (mean 1427.4), decode 116.0–117.8 tok/s**. The
M3-T1 baseline with the `pp 512,4096` command is 1431.8 (1426.3–1437.2): A is
neutral inside the ~0.5 % noise and stays above the 1423.9 M3 exit bar, while
freeing 8 192 B of shared memory and removing the exact-49 152 cliff.

Bit-exactness: `gemm_matches_reference_for_every_type` green (Q5_K 8.16e-7,
Q6_K 4.12e-7, IQ2_XS 0, IQ3_XXS 4.02e-7, IQ4_XS 4.39e-7, Q8_0 4.12e-7).
Parity: `--llama-numerics --fixed-experts --tag m3t2-vk-sm70` gives 625/640
top-1, 629/640 either-mode, per prompt identical to `m2-vk-sm70` (0 differing
prompts). No new Xid (last one is the pre-fix 2026-11-06 SKEDCHECK18).

### Variant C — subgroup → tile permutation (measured, rejected)

The 128x64 tile is 4 row tiles by 2 column tiles, so the map must keep two row
bits and one column bit. `SUBMAP 0` (default) is row = `w >> 1`, column =
`w & 1`; `SUBMAP 1` takes the column from `w >> 2`, so neighbouring subgroups
share their W rows. A first attempt that swapped the bits wholesale put `bn`
up to 96 in a 64-column tile: out-of-range shared reads, caught by the
per-type GEMM test (Q5_K rel err 0.79) and discarded (the wrong writes also
made the dispatch look 20 % faster). The corrected map was measured:

- end to end: prefill 1418.2–1419.7 tok/s, i.e. **−0.6 % vs A**;
- isolated: Q5_K `attn_qkv` 8.17–8.28 ms (+2 %), Q6_K 4.07–4.09, IQ2_XS
  0.95, IQ3_XXS 0.52; expert IQ3_XXS 64 positions 3.237–3.243 ms vs
  3.069–3.074 (+5 %).

Rejected: no gain, a small loss. `SUBMAP` stays 0 and is only a scaffold for a
future experiment (`GLSL_DEFS=-DSUBMAP=1 tools/gen_spirv.sh /tmp/spv`).

### No headroom for the other levers

- **B (occupancy):** A frees 8 KB with no time change, so the 48 KB was not the
  limiter. A Volta SM has 96 KB of shared memory, but the driver's L1/shared
  split is not controllable from Vulkan and two workgroups per SM are not
  guaranteed; nothing here can force or verify it. `BM`/`BK` changes need
  `src/vulkan/mod.rs` (#44) — `BK = 128` alone needs >= 62 KB at `XM = 128`.
- **D (`gemm_f_dp`):** 27 664 B for this model's specializations, but it only
  runs for non-`mmq` types, so it is not a target of the reference workload.
- **E (local size):** `local_size_x = 256` is derived from the X loader (2
  threads per row x 128) and the W loader (4 per row x 64); it can only change
  together with `BM` (out of scope, #44).

### Environment

Tesla V100-SXM2-32GB, sm_70, driver 580.178.04, Vulkan 1.4.312, glslc 2025.2,
commit `5a551eb` + `gemmi.glsl` variant A (`1646a9f`/`2ca5718` rebased), model
`Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`. `spirv_is_fresh` stays red for the 40
kernels whose committed `.spv` predate the local glslc (pre-existing drift);
`gemmi_dp.spv` and the cooperative `gemmi`/`gemmi_s` build byte-identically
from the final source.

## 2026-11-08 — Per-type f16 GEMM ablation on Vulkan (M3-T5, issue #44)

Built from `task/44-vulkan-int8-gemm` `07673a7` (the `STRATA_VK_MMQ` /
`STRATA_VK_F16_TYPES` knobs; `d8e5b1c` before the rebase onto `v100`), in an isolated worktree. Reference model, Vulkan
sm_70, `--llama-numerics --fixed-experts`; parity tags `m3t5-*` in `target/`.
The 7 int8-attributed positions of this run (0-based prompt, 0-based position
of the 32-token continuation) are `(15,0) (15,23) (16,8) (17,30) (18,0) (18,3)`
plus the `no-coop-matrices` residual `(16,16)`. Four further misses
`(6,9) (9,9) (9,28) (10,30)` are in every arm (evidence by both llama modes
absent) and are not part of the int8 gap.

### Single-type arms (`STRATA_VK_F16_TYPES=<id>`)

| type | id | either-mode | int8 positions fixed | new misses |
| --- | ---: | ---: | ---: | --- |
| — (default int8) | — | 629/640 | 0/7 | — |
| Q4_K | 12 | 629/640 | 0/7 | — |
| Q5_K | 13 | 631/640 | 3/7 | (1,12) |
| Q6_K | 14 | 630/640 | 2/7 | (16,20) |
| IQ2_XS | 17 | 631/640 | 3/7 | (17,29) |
| IQ3_XXS | 18 | 630/640 | 3/7 | (1,12) (14,2) |
| IQ4_XS | 23 | 633/640 | 5/7 | (16,20) |
| Q8_0 | 8 | 631/640 | 3/7 | (16,20) |
| all int8 types (`STRATA_VK_MMQ=0`) | — | **635/640** | 6/7 | — |

### Subsets (either-mode; 640 positions)

| subset (ids) | either-mode |
| --- | ---: |
| 23,17 | 633/640 |
| 23,14 / 23,13 | 630/640 |
| 23,8 | 632/640 |
| 23,17,13 | 633/640 |
| 23,17,8 | 630/640 |
| 23,17,13,8 | 632/640 |
| 17,18,23,8 / 13,18,23,8 | 631/640 |
| **13,17,18,23,8** (Q5_K, IQ2_XS, IQ3_XXS, IQ4_XS, Q8_0) | **634/640** |
| 14,17,18,23,8 | 633/640 |
| 13,14,18,23,8 | 633/640 |
| 13,14,17,23,8 | 631/640 |
| 13,14,17,18,8 | 631/640 |
| 13,14,17,18,23,8 (all int8 types minus Q4_K) | 635/640 |

The smallest subset reaching the parity bar is five types, and it still leaves
the `no-coop-matrices` residual `(16,16)`; adding one more type flips the
near-tie positions again (per-type gains are not additive).

### Cost and numeric gate of the winning subset

`--pp 512,4096 --tg 32 --reps 2 --ctx 4200`, mean (min–max), same window:

| arm | pp 512 | pp 4096 | decode @4096 | `prefill_matches_reference` |
| --- | ---: | ---: | ---: | --- |
| default (int8) | 771.9 (770.1–773.7) | 1425.4 (1421.0–1429.8) | 115.2 (114.8–115.7) | FAIL (15019) |
| `STRATA_VK_F16_TYPES=13,17,18,23,8` | 604.5 (602.3–606.7) | 1172.9 (1171.5–1174.3) | 114.6 (112.9–116.2) | **FAIL** (3302 vs 4938, max abs dlogit 1.5835) |
| `STRATA_VK_MMQ=0` (blanket) | — | — | — | PASS (4938, max abs dlogit 0.7616) |

The blanket row is **not** from this window: its 1135.2 tok/s (−20.7 %) and its
`prefill_matches_reference` PASS are those of the earlier
2026-11-08 "Cost of the f16 GEMM fallback on Vulkan" block (same
`--pp 512,4096 --tg 32 --reps 2 --ctx 4200` command, different run). Only the
subset row and the default row were measured here.

The subset costs **−17.7 % prefill at 4096** (−21.7 % at 512) and, although it
reaches 634/640 either-mode, **still fails `prefill_matches_reference`**
(next-step argmax 3302 against CPU 4938). Decode is unchanged within noise
(−0.5 %).

### Decision

Per the coordinator's rule (stop above 5 % prefill cost) and the original T3
tier (>15 %): **no default change**. The knobs ship as measurement tooling
only, and the parity bar alone does not identify an admissible subset: the one
subset that meets it fails the model-level numeric test and costs 17.7 % at
4K. Closing the gap needs either the blanket f16 arm (635/640 and a green
test, −20.7 % at 4K from the earlier cost block) accepted with an M3
exit-criterion amendment, or an int8 GEMM accuracy fix (follow-up). Q4_K and
Q6_K do not move any position they are solely needed for: Q4_K moves nothing,
Q6_K can be dropped from the 5-type subset at the cost of one near-tie
position. The subset rows above are the complete measured set; the earlier dev
report grouped `13,14,18,23,8 / 13,14,17,23,8 / 13,14,17,18,8` as 631–633,
which is the range of those three rows (633, 631, 631).

Environment: `no Xid` before and after (count 42), `nvidia-smi` 0 MiB,
`llama.router.service` inactive for the window and left free.

**Coordinator decision (M3-T5, issue #44):** keep the int8 default and the current numbers as the
documented ceiling. Both f16 arms are rejected: the blanket one passes the test but costs −20.7 % prefill
at 4K, and the 5-type subset costs −17.7 % and still fails `prefill_matches_reference`, so neither can be
shipped against the M3 Vulkan baseline (1425 tok/s at 4K). `STRATA_VK_MMQ` / `STRATA_VK_F16_TYPES` ship as
measurement tooling only; **no default changes**. The accuracy work continues in #50
(`[vulkan] int8 GEMM accuracy (M3-T5 follow-up)`). The default artifact remains 625 strict / 629
either-mode with `prefill_matches_reference` failing (15019 vs 4938).

## 2026-11-08 — M3-T5 control evidence, review round (issue #44)

Same reference model and machine, on `task/44-vulkan-int8-gemm` rebased onto `v100` `25b85d4`
(knobs commit `07673a7`), from the isolated worktree; one short window, GPU free, service stopped.

### Parity controls (`--llama-numerics --fixed-experts`, 20 prompts × 32 tokens)

| arm | top-1 | either-mode | divergent positions |
| --- | ---: | ---: | --- |
| default (no env) | 625/640 | 629/640 | 11, as recorded in the ablation block |
| `STRATA_VK_MMQ=1` (int8 control, tag `m3t5-mmq1`) | 625/640 | 629/640 | **identical set** to the default |
| CUDA (`--backend cuda`, tag `m3-cuda-sm70`) | 628/640 | **634/640** | 6 `(6,9) (9,9) (9,28) (10,30) (16,8) (18,0)` |

The control reproduces the old artifact's parity and positions. Qualification: `STRATA_VK_MMQ=1`
is the old path only for the types the policy already runs as int8 — it also forces IQ2_XS/IQ2_S onto
int8, whereas the policy runs them as f16 outside `--llama-numerics`. With `--llama-numerics` both were
int8 already, so the control and the default are identical here.

### Per-op int8 GEMM (`tests/kernels.rs::gemm_matches_reference_for_every_type`, Vulkan)

PASS, one `MATRIX` line per reference-model type against the per-backend scalar reference (assertion
1e-4): Q5_K 8.16e-7, IQ3_XXS 4.02e-7, Q6_K 4.12e-7, IQ2_XS 0.00e0, F32 0.00e0, IQ4_XS 4.39e-7,
Q8_0 4.12e-7, Q4_K 7.71e-7 relative L2 — the `Backend::mmq` routing keeps the int8 kernel green.

Environment: `no Xid` before and after (count 42), `nvidia-smi` back to idle, `llama.router.service`
inactive and left stopped.

## 2026-11-09 — Prefill attention A/B on Volta: no headroom in variants A and B (M3-T3, #51)

Target from the M3-T1 profile: `attn_prefill_dp` at 4K prefill (641.6 ms, 64 164 µs
per dispatch, 23.1 % of profiled kernel time). Branch `task/51-attn-prefill-tune`,
Tesla V100-SXM2-32GB (sm_70, driver 580.178.04, Vulkan 1.4.312),
`Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (`qwen35moe`), `spirv-val` and a
byte-identical default build checked on CPU first.

The model's shape, from its GGUF and `strata info`:
`qwen35moe.attention.head_count = 16`, `.head_count_kv = 2`,
`.key_length = 256` (`n_head 16`, `n_head_kv 2`, `head_dim 256`), 40 layers of
which **10 are `Attn`** (30 are `Delta`/SSM), which is exactly the 10
`attn_prefill_dp` dispatches of one 4096-token chunk.

Two variants behind selectable defines, both opt-in builds, the default source
path untouched (`attn_prefill_dp.comp`, variant A = `KEYBLOCK`, variant B =
`STAGE`). Variant C (lane owns 4 contiguous dims, `ldkv4`) was not implemented:
it is gated on A and B both failing, and it changes the dot-product summation
order, so it needs its own parity run; it is the recommended follow-up (below).

**Correction to the issue's assumption:** the reference model's head dim is
**256**, not 128 — the profile's specialization is `attn_prefill_dp [0, 8]` and
`PER = hd / 32 = 8`, so `HD = 256` and the f16 tile doubles: `BK = 32` needs
2 × 32 × 256 × 2 B = **32 768 B**, `BK = 16` 16 384 B. (The profile also rules
out the issue's `nh/nkv = 32/8`: it is 16/2.) The first A/B pass was
built with `STAGE_MAXHD=128` and returned early (`HD > STAGE_MAXHD`): it is
discarded, as is a pass made while a concurrent `cargo test --lib cuda::` from
another worktree saturated the host (prefill is ~66 % host-side, so the wall
clock of that pass collapsed to ~460 tok/s while the GPU-timed attention rows
stayed plausible).

### A/B (M3-T1 run A: graph off + profiler + stats, 4096 tokens, warm GPU)

```
M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
GLSL_DEFS=<variant> tools/gen_spirv.sh /tmp/spv && cp /tmp/spv/attn_prefill_dp.spv src/vulkan/spv/ \
  && cargo build --release && cp target/release/strata /tmp/strata-<variant>
STRATA_NO_GRAPH=1 STRATA_VK_PROF=1 STRATA_VK_STATS=1 /tmp/strata-<variant> bench --model "$M" \
  --gpu --backend vulkan --pp 4096 --tg 1 --reps 1 --ctx 4400
```

| Build | `GLSL_DEFS` | BK | Shared (B) | Regs | ISA bin (B) | blocks/SM | self ms | µs/call | Δ attn | prefill tok/s |
| --- | --- | --: | ---: | ---: | ---: | --: | ---: | ---: | ---: | ---: |
| baseline | — | — | 0 | 48 | 11 136 | 10 | 638.6 | 63 863.9 | — | 1389.1 |
| A8 | `-DKEYBLOCK=8` | 8 | 0 | 48 | 28 928 | 10 | 696.4 | 69 640.9 | **+9.0 %** | 1356.9 |
| A16 | `-DKEYBLOCK=16` | 16 | 0 | 64 | 47 232 | 8 | 745.9 | 74 585.4 | **+16.8 %** | 1351.2 |
| A32 | `-DKEYBLOCK=32` | 32 | 8 704* | 80 | 87 296 | 6 | 861.5 | 86 151.5 | **+34.9 %** | 1313.7 |
| B16 | `-DSTAGE=1 -DSTAGE_BK=16 -DSTAGE_MAXHD=256` | 16 | 16 384 | 60 | 39 296 | 6 | 1280.5 | 128 053.2 | **+100.5 %** | 1081.8 |
| B32 | `-DSTAGE=1 -DSTAGE_BK=32 -DSTAGE_MAXHD=256` | 32 | 32 768 | 97 | 67 712 | 3 | 2086.9 | 208 685.7 | **+226.8 %** | 834.1 |

Regs, shared and ISA bin are the driver's `STRATA_VK_STATS` per-pipeline
statistics (`Register Count`, `Shared Memory Size`, `Binary Size`); `Binary Size`
is the driver's compiled binary, **not** the `.spv` module (the baseline's `.spv`
is 10 936 B from the local glslc, 10 920 B committed).

\* The source declares no shared array for variant A, yet the driver's
`Shared Memory Size` statistic reports 8 704 B for A32 (the `Local Memory Size`
field reads 68 719 476 736, i.e. the device's local-memory limit, and
`Stack Size 0`, so it is not a usage figure): most plausibly driver-allocated
spill/scratch space of the fully unrolled 32-key block, consistent with its
87 KB binary. It is what the driver accounts for, so it is what the blocks/SM
formula uses; the registers are the tighter limit either way (8 704 B would
allow 11 blocks/SM, the 80 registers only 6).

`blocks/SM = min(96 KiB / shared, floor(65536 / (128 × regs)), 2048 / 128)`:
baseline 65536/6144 = 10.67 → 10; A16 65536/8192 = 8; A32 65536/10240 = 6.4 → 6
(shared 8 704 B would still allow 11); B16 min(6, 8, 16) = 6; B32 min(3, 5, 16)
= 3. Every build is ≤ 49 152 B (preflight: largest pair still `gemmi_dp`,
40 960 B).

### Decision: no variant adopted, the default kernel stays

The gain bar was ≤ 60 950 µs per dispatch (−5 %). Nothing came close, so the
committed SPIR-V is unchanged (the build from the branch's source is
byte-identical to the build of the pre-M3-T3 source: 10 936 B, `cmp` clean; the
committed `attn_prefill_dp.spv` is left as it was, like the other kernels whose
`.spv` predate the local glslc).

What the numbers rule out:

- **Not rescale-bound.** A8 has the baseline's exact occupancy (48 regs, 10
  blocks/SM, 0 B) and removes 7 of 8 `acc[j] * c` per key plus the per-key
  `fma(l, c, e)`; it is 9.0 % *slower*. The block form costs a second sweep over
  the BK scores plus a max scan, and it lengthens the dependency chain instead
  of shortening it. The per-key rescale was never the bottleneck.
- **Occupancy is the currency.** A16 (64 regs, 8 blocks) and A32 (80 regs, 6
  blocks) degrade roughly in proportion, and A32's 87 KB binary with the
  driver's 8 704 B of shared spill space shows the unrolled block outgrowing the
  register file. The kernel needs many in-flight lanes to hide the K/V load
  latency of the per-key chain (loads → 8 FMAs → 5-shuffle subgroup reduction →
  `expf_` → fma).
- **Not proven to be free of a load-path limit.** B16/B32 cut the global K/V
  reads 4 × (the 4 queries of a workgroup share the rows) and make them
  coalesced, but pay 2 barriers per BK keys, one f16 shared read per dim per key
  per lane, 16-bit shared loads (2-way bank conflicts on Volta) and 4 and 7
  blocks/SM of occupancy: +100.5 % and +226.8 %. That does **not** close the
  bandwidth question, because the staged variant changed three other things at
  once; it only says that staging K/V in shared is not the way to exploit the
  reuse.
- **Arithmetic.** One dispatch is one full-attention layer over a 4096-token
  chunk, causal: `n_head × T(T+1)/2` = 16 × 8 390 656 = 1.3425e8 query-key
  pairs, each 2·HD FLOP for QK plus 2·HD for PV with HD = 256 → 1 024 FLOP →
  **≈ 137 GFLOP per dispatch**, i.e. **≈ 2.15 TFLOP/s** in 63.9 ms = **6.9 % of
  the V100's 31 TFLOP/s FP16 CUDA-core peak**. The dispatch is therefore not
  FLOP-bound. (The issue's 4.3 GFLOP / 67 GFLOP/s estimate was for HD = 128 and
  seems to drop a factor; the corrected figure matters for the next paragraph.)
- **The loads are the same count as the FLOPs and are close to the L2 ceiling.**
  Each pair reads 2 × HD × 2 B = 1 024 B of K and V, so the **logical** K/V
  traffic is **≈ 137 GB per dispatch = 2.15 TB/s**, 2.4 × the V100's 900 GB/s
  HBM. The kernel cannot be DRAM-bound on that count — at least half of the
  reads must be served by cache — but how much reaches DRAM depends on the reuse
  the L2 keeps: the unique K/V of one layer (`n_head_kv 2 × 4096 keys × 256 ×
  2 B × K and V`) is **8.4 MB**, above the V100's **6 MB L2**, so the 8 query
  heads per KV head and the 4-query workgroup sharing are only partly cached;
  with the workgroup's 4 × alone the DRAM traffic would be ≈ 34 GB/dispatch
  ≈ 538 GB/s = 60 % of HBM peak. A V100 L2 delivers on the order of 2-3 TB/s,
  so 2.15 TB/s of logical reads sits at or near that ceiling. **Conclusion: not
  FLOP-bound (6.9 %) and not DRAM-bound on the logical count, but a
  load-path/L2-bandwidth limit is plausible and this A/B does not exclude it**;
  the per-SM logical rate (17.6 B/cycle at 80 SMs × 1.53 GHz) is well under the
  L1's ~128 B/cycle, so a pure `LSU`-throughput limit is less likely than L2
  bandwidth plus latency.

**Next lever (follow-up, not in #51):** variant C, one `ldkv4` 64-bit load per
lane for 4 contiguous dims instead of 8 scalar f16 loads, halves the load
*instructions* and shrinks the binary while keeping the same bytes; and/or
dropping the full unroll of the block loop (A8's 2.6 × binary vs the baseline).
If the limit is L2 bandwidth rather than instruction count, C cannot win, which
makes it a clean experiment; both change the per-lane dot order (C fully) and
need a parity run. A workgroup-shape change (more queries per workgroup, or
`nh`-major work sharing, in `src/vulkan/mod.rs`) is out of scope for this issue.

### Default re-validated after the A/B (same window)

`attn_prefill_matches_decode` ok (Vulkan vs decode rel err ≤ 4.42e-7 at
hd 256, Vulkan vs CUDA 6.14e-5); `kv_q8` 2/2 ok (f16 and `--kv q8`);
`tools/bench_v100.sh "$M" vulkan` twice, back to back:

| Run | pp 512 | pp 4096 | pp 32768 | decode @4096 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 768.5 | 1402.9 (1399.1–1406.5) | 520.8 | 117.5 |
| 2 | 768.6 | 1403.2 (1401.5–1405.8) | 520.3 | 117.6 |

Both samples are ~1.6 % under the M3-T1/M3-T2 controls (1 431.8 / 1 425.2) and
1.5 % under the roadmap's 1 423.9 floor, **with the unmodified kernel** (the
committed SPIR-V is what v100 ships), so this is the machine's state today, not
a regression of #51 — M3-T4 should re-check the floor against a same-session
control before failing a milestone on it.

Parity: `python3 -I tools/parity.py "$M" --extra "--gpu --backend vulkan"
--tag m3t3-vk` → **either 630/640, agree 626/640** (gate ≥ 629 / ≥ 625; the
M2-T5 baseline was 629/625), i.e. no parity change. No Xid: `dmesg | grep -ic
Xid` = 42 before and after, no device lost.
