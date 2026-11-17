# MACaQ on MLX: a native port and a W4A16 release for Qwen3.8-27B

## Summary

- We implemented MACaQ ([arXiv 2611.10649](https://arxiv.org/abs/2611.10649)), a calibrated MXFP4 post-training quantization method, natively in MLX. Every step runs on an Apple M5 Max: Hessians, exponent screening, and Flip-Refinement as a fused Metal kernel.
- **Release format.** Apple GPUs have no FP4 matmul, so the release is **W4A16**: MXFP4 weights, bf16 activations, and stock `mx.quantized_matmul`.
- **Quality.** On Qwen3.8-27B, MACaQ W4A16 at 4.25 bits per weight reaches **31% lower KLD (WikiText-2) and 24% lower KLD (C4)** than MLX's default affine 4-bit quantization, and is 0.8 GB smaller. It also has the smallest perplexity loss we measured: +1.46% (WikiText-2) and +1.22% (C4) over bf16.
- **Calibration data domain** affected quality more than any recipe parameter we tried. **The block Hadamard rotation** matters more than the per-channel scale.
- **Mixed precision.** Keeping 8 late blocks at 8-bit lowers KLD by another 33%, at +1.6 GB. We document it but don't release it (§6).

## 1. Method recap

MACaQ treats each linear $y = x W^\top$ on its own. It computes:

- **A per-input-channel scale**, $t_j = \sigma_j(W)^{\alpha} / \mathrm{rms}_j(x)^{1-\alpha}$.
- **A fixed block-diagonal Hadamard** $B$, with 32×32 blocks matching the MXFP4 block size.
- **The rewrite** $y = \big((x \odot t) B\big) \big((W \oslash t) B\big)^\top$. Only $W' = (W \oslash t) B$ is quantized; at inference the layer applies $x \odot t$ and $B$ online.

Codes are then chosen to minimize the Hessian-weighted output error $\sum_i r_i H_f r_i^\top$, where $H_f = B^\top D H D B + \lambda \cdot \mathrm{mean}(\mathrm{diag}(H)) \cdot I$, $D = \mathrm{diag}(t)$, and $H = X^\top X$ comes from calibration activations. This happens in two steps:

1. **Exponent screen.** For each 32-element block, try E8M0 exponents $e_0 + \lbrace +1, 0, -1 \rbrace$, where $e_0 = \lfloor \log_2 \mathrm{amax} \rfloor - 2$. Keep the one with the lowest $\mathrm{diag}(H_f)$-weighted round-to-nearest error.
2. **Flip-Refinement.** With exponents fixed, run up to $K$ iterations per row. Each iteration takes the single $\pm 1$ E2M1 level move with the best gain $2\Delta g - \Delta^2 H_{jj}$, then updates the gradient incrementally.

Nothing is propagated across layers. A layer's result depends only on its own $H$, so any subset of layers can be quantized and swapped independently. We rely on this for the sensitivity studies.

## 2. The MLX implementation

| Piece | Implementation |
|---|---|
| Hessians | fp32 $X^\top X$ per calibration sequence. Linears fed the same input (q/k/v, gate/up, GDN in_proj) share one accumulator, giving 4 per block (1.57 GB). **TF32 is disabled** (`MLX_ENABLE_TF32=0`); MLX otherwise runs fp32 matmuls as TF32 on Metal, with ~7e-4 relative error. |
| Calibration | Streaming. Hidden states for all calibration sequences are kept at the current block's input. Each block runs once in bf16, producing its Hessians and the next block's inputs. Hessians are cached to disk, so later configs reuse one calibration pass. |
| Exponent screen, rounding, packing | Pure MLX. E2M1 rounding is round-half-to-even, identical to MLX's own `mx.quantize`. Codes are packed into MLX's mxfp4 layout (8 nibbles per uint32, E8M0 uint8 scales); `mx.dequantize` reads them back bit-exactly. |
| Flip-Refinement | One `mx.fast.metal_kernel` per iteration, with one threadgroup per row: best-move scan, code update, rank-1 gradient update, accept flag. It is bit-identical to a pure-MLX reference version kept as a test oracle. It is **5–10× faster** than the reference: 4.5 s versus 32 s of flips per block at K = 128. |
| Inference | `MACaQLinear`: `x * t` (bf16), then `mx.hadamard_transform` on a `(…, n/32, 32)` view, then `mx.quantized_matmul(mode="mxfp4")`. |
| Checkpoint | Standard `mlx_lm` safetensors plus a `macaq` config section and a shipped loader `macaq_model.py`, loaded through `trust_remote_code`. See [`CHECKPOINT_FORMAT.md`](CHECKPOINT_FORMAT.md). |

Memory hygiene matters on a 128 GB machine running a 54 GB model. MLX's free-buffer cache grew to 39 GB in early runs. We cap it at 4 GB and clear it after each linear and each eval sequence, and abort cleanly above a set memory budget. Peak MLX memory is about 70 GB during calibration and quantization.

**Model coverage.** All 400 decoder linears are quantized:
- MLP gate/up/down in all 64 blocks;
- q/k/v/o in the 16 full-attention blocks;
- in_proj_qkv, in_proj_z and out_proj in the 48 Gated-DeltaNet blocks.

The 48-wide `in_proj_a` and `in_proj_b` gates stay bf16, as in the official code. Embeddings, lm_head, norms and conv1d also stay bf16. The vision tower is dropped.

## 3. Deviations from the paper and official code

| | Paper / official | Ours | Why |
|---|---|---|---|
| Activations | MXFP4 (W4A4) | **bf16 (W4A16)** for the release; W4A4 is emulated for research | No FP4 matmul on Apple GPUs, so A4 would add error without a speedup (§5). |
| $\alpha$ | 0.75 | **1.0** for W4A16 | With no activation quantization, only the weight-column balance matters; $\alpha = 1.0$ is best or tied (§4.1). |
| Calibration | Sequential: each block sees the outputs of already-quantized blocks | **Each block sees bf16-model activations** | Keeps every block independent, which the per-block and greedy studies need. |
| Calibration data | 2M tokens (CLI default); 200k in the paper's comparisons | **262k tokens**, half WikiText-2 train and half C4 train | §4.3 |
| Flip acceptance | $\mathrm{gain} > 10^{-12}$ | $\mathrm{gain} > 10^{-4} \cdot \Delta^2 H_{jj}$ | Scale-free guard against accepting noise-level moves. |
| E2M1 ties | away from zero | half to even | Matches MLX's own rounding. |
| Activation exponent (W4A4 only) | OCP, $\lfloor \log_2 \mathrm{amax} \rfloor - 2$ (can clip) | MLX, $\lceil \log_2(\mathrm{amax}/6) \rceil$ (never clips) by default; OCP selectable | MLX's convention beat OCP on 60 of 64 blocks (§5). |

## 4. W4A16 ablations

Setup for all tables:
- Full model, all 400 linears quantized.
- KLD is on 16×512 positions each of WikiText-2 test and C4 validation (seed 0 windows). Each reference is the full-vocab fp32 log-probs of bf16, cached once.
- Differences below about 3% are within noise at this eval size.

### 4.1 Which parts of the transform matter ($K = 128$, 65k WikiText calibration tokens)

| Config | KLD wiki | KLD C4 |
|---|---|---|
| $\alpha = 0.75$ (paper) | 3.00e-2 | 3.67e-2 |
| **$\alpha = 1.0$** | **2.92e-2** | **3.53e-2** |
| $\alpha = 0.5$ | 3.04e-2 | 4.06e-2 |
| no scale ($t = 1$), rotation only | 3.15e-2 | 3.54e-2 |
| $\alpha = 0.75$, no rotation | 3.37e-2 | 4.41e-2 |
| $\alpha = 1.0$, no rotation | 3.39e-2 | 4.15e-2 |
| no scale, no rotation (screen + flips only) | 3.51e-2 | 4.20e-2 |
| *mxfp4 RTN (`mlx_lm --q-mode mxfp4`)* | *6.06e-2* | *5.74e-2* |
| *affine 4-bit gs64 (`mlx_lm -q`)* | *4.02e-2* | *4.01e-2* |

- **Removing the rotation costs 11–15% KLD on WikiText and 13–20% on C4,** whatever the $\alpha$.
- **Removing the scale costs at most about 8%.**
- **The calibrated codes do the most work.** Even with no scale and no rotation, the screen and flips cut mxfp4 RTN's KLD by 42%.

### 4.2 Flip budget and calibration size ($\alpha = 1.0$, WikiText calibration)

| Calibration tokens | $K$ | KLD wiki | KLD C4 | Quantization time |
|---|---|---|---|---|
| 65k | 128 | 2.92e-2 | 3.53e-2 | 9.6 min |
| 262k | 128 | 2.92e-2 | 3.49e-2 | 8.5 min |
| 262k | 512 | **2.71e-2** | **3.33e-2** | 19.6 min |

- **$K = 512$ gives −7% on WikiText and −5% on C4** at 262k tokens.
- **On a single-block test at 65k tokens, $K = 512$ lowered the layer objective but not KLD.** That's consistent with more flips needing more calibration data, but we didn't run $K = 512$ at 65k on the full model.
- **4× more tokens alone, at $K = 128$, changed little.**

### 4.3 Calibration data ($\alpha = 1.0$, $K = 512$, 262k tokens)

| Calibration | KLD wiki | KLD C4 | PPL wiki | PPL C4 |
|---|---|---|---|---|
| WikiText-2 train | **2.71e-2** | 3.33e-2 | 7.1611 | 9.6243 |
| C4 train | 3.36e-2 | **2.92e-2** | – | – |
| **mix (½ + ½, interleaved)** | 2.77e-2 | 3.06e-2 | **7.1565** | **9.6032** |

- **Each calibration source wins its own domain by 12–24% KLD.** That's a larger effect than $\alpha$, $K$ or calibration size.
- **The mix keeps WikiText KLD within 2% (noise) and recovers two-thirds of the C4 gain.** It's also slightly better on both PPLs, so it is the release recipe.
- **Domain sensitivity probably extends to code, chat and multilingual text,** which we did not measure (§8).

## 5. W4A4 (research only)

We also emulated full W4A4, quantizing activations to MXFP4 as per the reference paper, using the same weight codes. Setup: $K = 128$, 65k WikiText calibration tokens, 8×512 WikiText eval positions.

| | KLD wiki | WikiText-2 PPL |
|---|---|---|
| MACaQ W4A16 | 3.15e-2 | 7.1612 |
| MACaQ W4A4, MLX activation exponent | 7.09e-2 | 7.3186 |
| MACaQ W4A4, OCP activation exponent | 7.38e-2 | – |
| mxfp4 RTN W4A4 | 1.57e-1 | 7.7059 |

- **MACaQ halves W4A4 KLD relative to RTN.**
- **Activation quantization accounts for 55% of the full model's W4A4 KLD** (a median 39% per block). It more than doubles the error of the W4A16 model built from the same codes.
- **On a Mac it buys nothing,** because MLX would dequantize the activations back to bf16 for the matmul anyway. W4A16 is the right deployment target on Apple Silicon. The same codes would still serve W4A4 on FP4 hardware.

Mixed activation precision on top of W4A4 (some blocks kept at A16) ranks blocks the same way the weight-sensitivity sweep does. A16 on the top 16 blocks recovers 25% of the KLD and matches W4A16's PPL (7.1685). These results are in `results/full_macaq_K128_c65536.json`.

## 6. Where the error lives: per-block sensitivity and mixed precision

**Single-block sweep.** Each of the 64 blocks was quantized alone, with the rest kept bf16 ($K = 128$, 65k tokens). The per-block KLD profile is U-shaped:
- low in blocks 11–15 (about 0.8e-3 at W4A16);
- bumps at 23–27 and 35–39;
- a steep rise from block 50 to block 63, the worst at 1.98e-3;
- block 0 also high at 1.74e-3.

Block type (full attention versus Gated DeltaNet) matters much less than depth. The data is in `results/sweep_macaq_K128_c65536.jsonl`.

**Greedy mixed precision.** Starting from the release recipe, each step tried keeping each remaining block at MLX affine 8-bit (group size 64). It kept the block with the best KLD reduction per byte, screened on 4 sequences per set and confirmed on the full eval. Scoring used the mean of WikiText and C4 KLD.

| 8-bit blocks | Disk | KLD wiki | KLD C4 | PPL wiki | PPL C4 | Decode |
|---|---|---|---|---|---|---|
| none (release) | 18.1 GB | 2.77e-2 | 3.06e-2 | 7.1565 | 9.6032 | 26–31 tok/s |
| 59, 61, 57, 63 (step 4) | ≈18.9 GB | 2.27e-2 | 2.72e-2 | – | – | – |
| 55–59, 61–63 (step 8) | 19.7 GB | 1.85e-2 | 2.44e-2 | 7.1066 | 9.5611 | 24–28 tok/s |

- **Every pick was one of the last 9 blocks.** A WikiText-only-scored run picked 56–63 instead, so the exact set depends on the scoring data, but always lies in the tail.
- **Each block costs about 200 MB and buys a 4–7% KLD reduction.**
- **The gain had not flattened after 8 steps.**
- **We didn't pursue this variant further.** The gain over uniform W4A16 was modest for its cost (+9% size, about 8% slower decode). Mixed precision may still be a useful direction on other models, particularly where, as here, sensitivity is concentrated in a few layers.

## 7. Release checkpoint

[`mlx-community/Qwen3.8-27B-MACaQ-W4A16-mlx`](https://huggingface.co/mlx-community/Qwen3.8-27B-MACaQ-W4A16-mlx):
- 18.10 GB, 4 safetensors shards.
- Recipe: $\alpha = 1.0$, $K = 512$, 32-block Hadamard, $\lambda = 1.0$, 262k mixed calibration tokens.

Verification, through `mlx_lm.load(path, trust_remote_code=True)`:

| Check | Result |
|---|---|
| Stock `mlx_lm.load` without `trust_remote_code` | Refused |
| Config with the loader entry removed | Refused ("parameters not in model"), so no silent misload as plain mxfp4 |
| Logits after reload vs the in-memory quantized model | Bitwise identical (max difference 0.0) |
| KLD after reload | Reproduces exactly |
| PPL | WikiText-2 7.1565 (bf16 7.0532); C4 9.6032 (bf16 9.4876) |
| Greedy chat, 6 prompts (arithmetic, time arithmetic, translation, facts, code, explanation) | All correct and fluent; 2 of 6 token-identical to bf16 |
| Decode on M5 Max (`stream_generate`, batch 1) | 26–31 tok/s vs about 9 for bf16; peak 23.6 GB |

**Online transform cost.** In a per-linear micro-benchmark against plain mxfp4 with identical packed weights:
- at batch 1 the scale and Hadamard add **3–9%** per linear;
- at 512-token prefill they add 9–21%.

Run alone, the transform takes about 140 µs. But at decode it mostly overlaps with the matmul, because decode is limited by kernel latency rather than bandwidth. We didn't write a fused kernel, since the end-to-end overhead didn't justify one.

**Provenance.** `macaq_provenance.json` in the checkpoint records:
- the full recipe;
- the calibration token-id sha256;
- base-model shard and code-file sha256s;
- MLX, mlx-lm and Python versions, macOS 27.2.1, and the chip.

## 8. Limitations and open questions

- **Eval size.** There are 8,192 positions per KLD set, so differences under about 3% are noise. PPL is coarser still: it could not separate configs that differ by 7% in KLD.
- **Calibration domain.** Only English web and Wikipedia text was used. Given §4.3, chat, code or multilingual calibration data would probably help those domains; this is untested.
- **No downstream benchmarks.** We ran no GSM8K, MMLU or IFEval, only the short generation check.
- **Not tested:**
  - the paper's own PPL protocol (official README: 2.8463 → 2.8994 for W4A4);
  - sequential (GPTQ-style) calibration;
  - 2M calibration tokens;
  - TF32 Hessians end to end. TF32 changes Hessians by about 1e-5 and is about 3× faster, but it flips up to 3.6% of packed codes on near-ties.
- **Other models.** The method is model-agnostic, but this code is validated only on `qwen3_5`.

## Reproducibility

Every number above comes from a file in `results/`; see the README for the commands.
