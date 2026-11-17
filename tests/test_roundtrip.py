"""Checkpoint format test on a tiny random Qwen3.5-style model (no downloads, ~10 s).

Quantizes every decoder linear with MACaQ (one block in affine 8-bit to exercise the mixed path),
saves with mlx_lm.save_model + the MACaQ config section, reloads through stock
mlx_lm.utils.load_model(trust_remote_code=True) and checks:
  * stock loading without trust_remote_code is refused,
  * loading with the shim gives bitwise-identical logits,
  * removing model_file fails (no silent misload as plain mxfp4).

    .venv/bin/python -m pytest tests/  (or: .venv/bin/python tests/test_roundtrip.py)
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["MLX_ENABLE_TF32"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import _get_classes, load_model, save_model

import macaq_mlx as M

TEXT = dict(model_type="qwen3_5_text", hidden_size=256, intermediate_size=512, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=64, linear_value_head_dim=64,
            linear_conv_kernel_dim=4, full_attention_interval=4, vocab_size=1024, rms_norm_eps=1e-6,
            tie_word_embeddings=False, attn_output_gate=True, partial_rotary_factor=0.25,
            max_position_embeddings=4096,
            rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=0.25,
                                 mrope_interleaved=True, mrope_section=[11, 11, 10]),
            layer_types=["linear_attention"] * 3 + ["full_attention"])
CONFIG = dict(model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"],
              tie_word_embeddings=False, text_config=TEXT)


def build(dst):
    mx.random.seed(0)
    Model, Args = _get_classes(CONFIG)
    model = Model(Args.from_dict(CONFIG))
    model.update(tree_unflatten([(k, (mx.random.normal(v.shape) * 0.05).astype(mx.bfloat16))
                                 for k, v in tree_flatten(model.parameters())]))
    modules = {}
    for b, layer in enumerate(model.language_model.model.layers):
        new = []
        for n, lin in M.block_linears(layer):
            if not M.quantizable(n, lin):
                continue
            path = f"language_model.model.layers.{b}.{n}"
            if b == 3:
                q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
                modules[path] = dict(kind="affine", bits=8, group_size=64)
            else:
                X = mx.random.normal((512, lin.weight.shape[1]))
                q = M.quantize_linear(lin.weight, X.T @ X, 512, K=8, act_exp=None, verbose=False, alpha=1.0)
                modules[path] = "macaq"
            new.append((n, q))
        layer.update_modules(tree_unflatten(new))
    toks = mx.random.randint(0, TEXT["vocab_size"], (1, 64))
    ref = model(toks).astype(mx.float32)
    mx.eval(ref)
    save_model(dst, model, donate_model=True)
    cfg = dict(CONFIG, model_file="macaq_model.py",
               quantization_config=dict(quant_method="macaq_mxfp4", bits=4, group_size=32, activations="bf16"),
               macaq=dict(format="macaq-mxfp4-w4a16", format_version=1, block=32, has_t=True, rotate=True,
                          modules=modules))
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)
    shutil.copy(ROOT / "macaq_model.py", dst)
    return toks, ref, len(modules)


def test_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d)
        toks, ref, n = build(dst)
        try:
            load_model(dst)
            raise AssertionError("stock load without trust_remote_code accepted the checkpoint")
        except ValueError as e:
            assert "trust_remote_code" in str(e)
        model, _ = load_model(dst, trust_remote_code=True)
        kinds = [type(m).__name__ for _, m in model.named_modules()]
        assert kinds.count("MACaQLinear") + kinds.count("QuantizedLinear") == n
        assert mx.array_equal(model(toks).astype(mx.float32), ref).item(), "round-trip logits differ"
        cfg = json.load(open(dst / "config.json"))
        cfg.pop("model_file")
        json.dump(cfg, open(dst / "config.json", "w"))
        try:
            load_model(dst)
            raise AssertionError("checkpoint loaded without its MACaQ loader")
        except ValueError as e:
            assert "parameters not in model" in str(e)


if __name__ == "__main__":
    test_roundtrip()
    print("ok")
