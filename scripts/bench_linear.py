"""Per-linear cost of MACaQ's online transform vs plain mxfp4 / affine4 / bf16 (decode & prefill shapes)."""
import os, sys, time; os.environ["MLX_ENABLE_TF32"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx, mlx.nn as nn
import macaq_model as MM

def bench(f, x, n=200):
    for _ in range(10): mx.eval(f(x))
    t = time.perf_counter()
    for _ in range(n): y = f(x); mx.eval(y)
    return (time.perf_counter() - t) / n * 1e6

shapes = [(5120, 17408), (17408, 5120), (5120, 12288), (6144, 5120)]   # (in, out): gate/up, down, q, o
for M_ in (1, 512):
    print(f"--- tokens={M_} (us/call)")
    for i, o in shapes:
        W = (mx.random.normal((o, i)) * 0.02).astype(mx.bfloat16)
        x = mx.random.normal((1, M_, i)).astype(mx.bfloat16)
        bf = nn.Linear(i, o, bias=False); bf.weight = W
        aff = nn.QuantizedLinear.from_linear(bf, group_size=64, bits=4)
        mxf = nn.QuantizedLinear.from_linear(bf, group_size=32, bits=4, mode="mxfp4")
        mq = MM.MACaQLinear(i, o); mq.weight, mq.scales = mxf.weight, mxf.scales
        mq.t = mx.ones((i,), mx.bfloat16)
        mx.eval(W, x, aff.parameters(), mxf.parameters(), mq.parameters())
        r = {k: bench(f, x) for k, f in [("bf16", bf), ("affine4", aff), ("mxfp4", mxf), ("macaq", mq)]}
        r["transform_only"] = bench(lambda z: mx.hadamard_transform((z * mq.t).reshape(1, M_, i // 32, 32)).reshape(z.shape), x)
        print(f"  {i:5d}->{o:5d}  " + "  ".join(f"{k} {v:7.1f}" for k, v in r.items())
              + f"   macaq/mxfp4 {r['macaq']/r['mxfp4']:.2f}x")
