# macaq-mlx

A native [MLX](https://github.com/ml-explore/mlx) implementation of **MACaQ**, a training-free post-training quantization method for MXFP4, for Apple Silicon. The method is from
*Muller et al., "MACaQ: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations"*,
[arXiv 2611.10649](https://arxiv.org/abs/2611.10649).

- **Quantization runs entirely on the Mac.** That covers calibration, the Hessians, the exponent screen and Flip-Refinement (a fused Metal kernel).
- **The output is a standard `mlx_lm` checkpoint** plus a ~100-line loader shim. It loads with `mlx_lm.load(..., trust_remote_code=True)` and works with `mlx_lm.generate`, `mlx_lm.chat` and `mlx_lm.server`.
- **Released model:** [`mlx-community/Qwen3.8-27B-MACaQ-W4A16-mlx`](https://huggingface.co/mlx-community/Qwen3.8-27B-MACaQ-W4A16-mlx), 18.1 GB.

The [official code](https://github.com/huawei-csl/MACaQ) targets CUDA, vLLM and llama.cpp, and was used as a reading reference for the MLX implementation.

## Results: Qwen3.8-27B, W4A16

The comparison uses the same harness and eval tokens for every row. KLD is the mean token-level KL(bf16 ‖ quantized) over the full vocabulary, on 8,192 positions each of WikiText-2 test and C4 validation.

| Model | Size | KLD WikiText-2 | KLD C4 | PPL WikiText-2 | PPL C4 |
|---|---|---|---|---|---|
| bf16 | 54.7 GB | 0 | 0 | 7.0532 | 9.4876 |
| **MACaQ W4A16 (this repo)** | **18.1 GB** | **0.0277** | **0.0306** | **7.1565** | **9.6032** |
| `mlx_lm.convert -q` (affine 4-bit, group size 64) | ≈18.9 GB | 0.0402 | 0.0401 | 7.1690 | 9.7063 |
| `mlx_lm.convert -q --q-mode mxfp4` | ≈18.1 GB | 0.0606 | 0.0574 | 7.2023 | 9.8595 |

Against MLX's default 4-bit quantization, MACaQ has 31% lower KLD on WikiText and 24% lower on C4, with 0.8 GB less on disk. It also has the smallest perplexity gap to bf16: +1.46% and +1.22%. The full study is in [`docs/WRITEUP.md`](docs/WRITEUP.md).

## Install

Requirements: macOS on Apple Silicon, Python 3.12 or 3.13, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rgarcia-fx/macaq-mlx.git && cd macaq-mlx
uv sync
.venv/bin/python macaq_mlx.py --selftest     # random-matrix checks, incl. Metal kernel vs reference
.venv/bin/python tests/test_roundtrip.py     # tiny-model save/load round trip
```

## Reproducing the release checkpoint

```bash
export HF_DATASETS_OFFLINE=0                  # a16.py defaults to offline; allow the WikiText-2 download
uv run hf download mlx-community/Qwen3.8-27B-bf16 --local-dir models/Qwen3.8-27B-bf16
mkdir -p data && curl -L -o data/c4-train.00000-of-01024.json.gz \
  https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz
curl -L -o data/c4-validation.00000-of-00008.json.gz \
  https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz

# 1) bf16 reference for KLD/PPL (runs/ref, 15 GB; about 13 min)
.venv/bin/python a16.py ref
# 2) calibrate + quantize + write the checkpoint (about 45 min; Hessian cache 94 GB in runs/)
.venv/bin/python a16.py export --config a1.0 --flips 512 --calib 262144 --calib-data mix
# 3) reload through mlx_lm and check: refusal without trust_remote_code, bitwise round trip, KLD, PPL, generations
.venv/bin/python a16.py verify --out-dir runs/export/Qwen3.8-27B-MACaQ-W4A16-a1.0-K512-c262144-mix
```

You should get the same calibration-token sha256 as `results/release_provenance.json`. The provenance file also records sha256s of the code as it ran. `macaq_model.py` is unchanged since then. `macaq_mlx.py` and `a16.py` differ only in docstrings and one unused variable; the logic that produced the checkpoint is identical. The checkpoint is deterministic given the same MLX version; it was produced with MLX 0.32.4 and mlx-lm 0.32.2 on an M5 Max.

Resource use with 128 GB of unified memory:
- **Peak MLX memory** is about 70 GB.
- **Disk:** about 110 GB of scratch (Hessian cache plus reference), plus 18 GB for the output. Delete `runs/hess_*` afterwards.

## Repository layout

| Path | What it is |
|---|---|
| `macaq_mlx.py` | Core method: transforms, Hessian accumulation, exponent screen, Flip-Refinement (MLX reference version and fused Metal kernel), MXFP4 packing, `MACaQLinear`. Also has the research CLI: single-block runs, `--sweep` per-block sensitivity, and `--full` W4A4/W4A16 full-model variants. |
| `a16.py` | W4A16 pipeline: `ref` (cached bf16 reference), `configs` (α, rotation, K and calibration ablations), `greedy` (mixed-precision search), `export` (checkpoint writer) and `verify` (reload, round trip, KLD, PPL, generations). |
| `macaq_model.py` | Loader shim shipped inside every checkpoint. It builds the stock `mlx_lm` model and swaps in `MACaQLinear`. |
| `analyze_sweep.py` | Summarizes a per-block sweep JSONL. |
| `scripts/bench_linear.py` | Per-linear speed micro-benchmark: MACaQ vs mxfp4, affine and bf16. |
| `tests/test_roundtrip.py` | Checkpoint format test on a tiny random model. |
| `results/` | Raw results behind every table in the writeup (JSON and JSONL), plus the release provenance. |
| `docs/WRITEUP.md` | The study: what was implemented, deviations, ablations, results. |
| `docs/CHECKPOINT_FORMAT.md` | On-disk format and loader contract, for anyone writing another loader. |

## Using MACaQ on other models

The method is layer-local and model-agnostic, but the drivers here were written and validated only for the
`qwen3_5` architecture. Model-specific code is limited to a few places:
- `text_core()` and `get_layers()` locate the decoder stack;
- `SKIP_SUFFIXES` lists linears that are left unquantized;
- the per-layer mask selection in `calib_inputs()`.

Porting to another `mlx_lm` architecture should mostly mean adapting these. Contributions are welcome.

## License and citation

Apache-2.0 (see `LICENSE`). If you use this, please cite the MACaQ paper:

```bibtex
@misc{muller2026macaq,
  title         = {{MACaQ}: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations},
  author        = {Muller, Lorenz K. and Bich, Philippe and Boretti, Chiara and Chang, Hyun-Min and Zhuang, Jiawei and Cavigelli, Lukas},
  year          = {2026},
  eprint        = {2611.10649},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.10649}
}
```
