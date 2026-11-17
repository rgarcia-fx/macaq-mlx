# Raw results

Every table in `docs/WRITEUP.md` comes from these files. All were produced on an Apple M5 Max (128 GB) with MLX 0.32.4, mlx-lm 0.32.2 and TF32 off.

| File | Contents | Writeup |
|---|---|---|
| `sweep_macaq_K128_c65536.jsonl` | Per-block sweep: each block quantized alone (K=128, 65k WikiText tokens), with KLD at W4A4-mlx, W4A4-ocp and W4A16 | §5, §6 |
| `full_{macaq,rtn,affine}_K128_c65536.json` | Full model W4A4/W4A16 variants and A16 mixed precision (8×512 WikiText eval) | §5 |
| `a16_configs_c65536_K128.jsonl` | W4A16 α and rotation ablation, plus rtn and affine baselines (16×512 WikiText + C4 eval) | §4.1 |
| `a16_configs_c262144_K{128,512}.jsonl` | Calibration size and K, WikiText calibration | §4.2 |
| `a16_configs_c262144_{c4,mix}_K512.jsonl` | Calibration data source | §4.3 |
| `a16_baselines_ppl.jsonl` | rtn and affine 4-bit baselines with PPL | README, §7 |
| `a16_greedy_*_c262144.jsonl` | Greedy 8-bit blocks, WikiText calibration, WikiText screening | §6 |
| `a16_greedy_*_mix_screenboth.jsonl` | Greedy 8-bit blocks, mix calibration, screening on WikiText + C4 | §6 |
| `release_verify.json`, `release_provenance.json` | Release checkpoint verification and provenance | §7 |
| `q8tail_verify.json` | Unreleased 8-bit-tail variant: verification | §6 |
| `wikicalib_verify.json` | Superseded WikiText-calibrated candidate: verification | §4.3 |
| `bf16_generations.json` | bf16 greedy generations for the 6 sanity prompts | §7 |
| `ref_meta.json`, `ppl_bf16.json` | bf16 reference: eval token checksums and PPL | – |

KLD fields: `kld` is the mean, `p99` the 99th-percentile per-token KLD, and `top1` the argmax agreement with bf16.
