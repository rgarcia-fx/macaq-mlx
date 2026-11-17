"""
macaq_model.py -- loader shim for MACaQ W4A16 MLX checkpoints.

Shipped next to config.json. mlx_lm loads it via `"model_file": "macaq_model.py"`
when called with trust_remote_code=True, e.g.

    mlx_lm.generate --model <path> --trust-remote-code --prompt "..."
    python -c "from mlx_lm import load; m, t = load('<path>', trust_remote_code=True)"

Without trust_remote_code stock mlx_lm refuses to load the checkpoint, which is
intended: the weights are MXFP4 codes of the *transformed* matrix (W / t) B, so
treating them as plain mxfp4 would load silently and produce garbage.

Each MACaQ linear computes  y = mxfp4_matmul( ((x * t) B), W_q^T )  where t is a bf16
per-input-channel scale and B = blockdiag(H_32 / sqrt(32)) is a 32-block Hadamard.
Activations stay bf16 (W4A16). See arXiv 2611.10649 for the method.

Self-contained: depends only on mlx and mlx_lm. The base architecture comes from the
stock mlx_lm model for config["model_type"]; this file only swaps the linear layers.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm.utils import _get_classes

FORMAT = "macaq-mxfp4-w4a16"
FORMAT_VERSION = 1


class MACaQLinear(nn.Module):
    def __init__(self, input_dims, output_dims, block=32, has_t=True, rotate=True):
        super().__init__()
        assert input_dims % block == 0
        self.block, self.rotate = block, rotate
        # placeholders with the stored shapes; real values come from load_weights
        self.weight = mx.zeros((output_dims, input_dims // 8), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // block), dtype=mx.uint8)
        if has_t:
            self.t = mx.ones((input_dims,), dtype=mx.bfloat16)
        self.freeze()

    def __call__(self, x):
        if "t" in self:
            x = x * self.t.astype(x.dtype)
        if self.rotate:
            shp = x.shape
            x = mx.hadamard_transform(x.reshape(*shp[:-1], shp[-1] // self.block, self.block))
            x = x.reshape(shp)
        return mx.quantized_matmul(x, self["weight"], self["scales"], transpose=True,
                                   group_size=self.block, bits=4, mode="mxfp4")


def _build_module(spec, lin, mc):
    out_d, in_d = lin.weight.shape
    kind = spec if isinstance(spec, str) else spec["kind"]
    if kind == "macaq":
        return MACaQLinear(in_d, out_d, block=mc["block"], has_t=mc["has_t"], rotate=mc["rotate"])
    if kind == "affine":
        return nn.QuantizedLinear(in_d, out_d, bias="bias" in lin, group_size=spec["group_size"],
                                  bits=spec["bits"], mode="affine")
    if kind == "bf16":
        return None
    raise ValueError(f"unknown MACaQ module kind {kind!r}")


class ModelArgs:
    """mlx_lm calls ModelArgs.from_dict(config) and passes the result to Model."""

    @classmethod
    def from_dict(cls, params):
        return params


class Model:
    """Factory: returns the stock mlx_lm model for config['model_type'] with MACaQ linears
    swapped in, so parameter paths, sanitize(), make_cache() etc. are all the stock ones."""

    def __new__(cls, config):
        mc = config.get("macaq")
        if not mc or mc.get("format") != FORMAT:
            raise ValueError(f"config has no '{FORMAT}' macaq section")
        if mc.get("format_version", 0) > FORMAT_VERSION:
            raise ValueError(f"checkpoint format v{mc['format_version']} is newer than this loader "
                             f"(v{FORMAT_VERSION})")
        base_cls, base_args = _get_classes(config)
        model = base_cls(base_args.from_dict(config))
        mods = dict(model.named_modules())
        new = []
        for path, spec in mc["modules"].items():
            if path not in mods or not isinstance(mods[path], nn.Linear):
                raise ValueError(f"MACaQ module {path} is not a Linear in the base model")
            m = _build_module(spec, mods[path], mc)
            if m is not None:
                new.append((path, m))
        model.update_modules(tree_unflatten(new))
        return model
