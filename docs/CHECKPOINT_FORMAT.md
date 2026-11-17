# MACaQ MLX checkpoint format (v1)

This document describes what a MACaQ MLX checkpoint contains, so that other loaders can be written.
The reference loader is `macaq_model.py`, which ships inside every checkpoint.

## Files

A standard `mlx_lm` model directory:
- `config.json`;
- `model-0000N-of-0000M.safetensors` and `model.safetensors.index.json`;
- tokenizer files and `chat_template.jinja`;
- `generation_config.json`.

It also contains:
- `macaq_model.py`, the loader shim;
- `macaq_provenance.json`, which records the recipe, calibration data hash, base-model, code and checkpoint sha256s, and software versions.

## config.json additions

```jsonc
{
  "model_type": "qwen3_5",                // unchanged: the base mlx_lm architecture
  "model_file": "macaq_model.py",         // mlx_lm imports this when trust_remote_code=True
  "quantization_config": {                // deliberately NOT "mxfp4": stock loaders must not
    "quant_method": "macaq_mxfp4",        // treat these weights as plain mxfp4
    "bits": 4, "group_size": 32, "activations": "bf16"
  },
  "macaq": {
    "format": "macaq-mxfp4-w4a16",
    "format_version": 1,
    "block": 32,                          // MXFP4 block == Hadamard block
    "has_t": true,                        // per-channel scale present
    "rotate": true,                       // block Hadamard present
    "recipe": { ... },                    // informational (alpha, K, lambda, calibration, ...)
    "modules": {                          // every quantized linear, by parameter path
      "language_model.model.layers.0.mlp.gate_proj": "macaq",
      // reserved for mixed precision (not used by the release):
      // "...": {"kind": "affine", "bits": 8, "group_size": 64}
    }
  }
}
```

Linears not listed in `modules` are bf16 `nn.Linear`. The config has no `quantization` key, so stock `mlx_lm` never runs `nn.quantize` on the model.

## Tensors of a `"macaq"` linear (in → out features)

| Name | dtype | Shape | Meaning |
|---|---|---|---|
| `<path>.weight` | uint32 | (out, in/8) | MXFP4 codes of `W' = (W / t) B`, in MLX's mxfp4 layout: 8 E2M1 nibbles per word, low nibble first, sign in bit 3 |
| `<path>.scales` | uint8 | (out, in/32) | E8M0 block exponents (biased by 127), one per 32 inputs |
| `<path>.t` | bfloat16 | (in,) | Per-input-channel scale. Present iff `has_t`. |

There is no bias; none of the Qwen3.5 linears have one.

## Forward pass

```python
x = x * t.astype(x.dtype)                                   # if has_t
x = mx.hadamard_transform(x.reshape(..., in // 32, 32))     # if rotate; normalized H32/sqrt(32)
x = x.reshape(..., in)
y = mx.quantized_matmul(x, weight, scales, transpose=True, group_size=32, bits=4, mode="mxfp4")
```

`B` is orthonormal and symmetric, so `y ≈ x Wᵀ`. A reimplementation must apply `t` and the Hadamard *before* the mxfp4 matmul. Skipping them gives fluent-looking garbage.

## Failure modes, by design

| Situation | Result |
|---|---|
| `mlx_lm.load(path)` without `trust_remote_code` | `ValueError: ... requires importing and running a custom module` |
| `model_file` removed | strict load fails: `Received N parameters not in model: ….t` |
| `format_version` newer than the loader | `ValueError` from `macaq_model.py` |

## Versioning

Any change to the tensor layout, the forward pass or module specs bumps `format_version`. Loaders refuse checkpoints with a newer version.
