"""
macaq_mlx.py -- core of a native MLX implementation of MACaQ (arXiv 2611.10649).

Method pieces: per-channel scale t (Eq. 9), 32-block Hadamard B (Eq. 3-7), transformed
Hessian (Eq. 5-6), exponent screen (Eq. 12), Flip-Refinement (Eq. 13-16, Alg. 1; fused
Metal kernel + pure-MLX reference), MXFP4 packing, and MACaQLinear for inference.

Research CLI (W4A4 / W4A16 emulation; the W4A16 release pipeline is in a16.py):

    python macaq_mlx.py --selftest                       # random-matrix checks
    python macaq_mlx.py --layer 3 [--act-exp ocp] [--w4a16]   # one block, KLD vs bf16
    python macaq_mlx.py --sweep --blocks all             # per-block sensitivity sweep
    python macaq_mlx.py --full                           # whole model + A16 variants
"""
import argparse
import os
import sys
import time

# MLX runs fp32 matmuls as TF32 on Metal by default (~7e-4 rel. error). Hessians,
# flip gradients and objectives need true fp32. bf16 matmuls are unaffected.
# Must be set before mlx is imported; pass --tf32 to opt back in.
if "--tf32" not in sys.argv:
    os.environ["MLX_ENABLE_TF32"] = "0"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

BLOCK = 32          # MXFP4 block == Hadamard block (paper default)
ALPHA = 0.75        # Eq. (9) balance
LAM = 1.0           # Eq. (6) ridge
T_CLAMP = (1e-4, 1e4)  # paper: "a finite range"; official code uses the same values
FLIP_RTOL = 1e-4    # accept a flip only if gain > FLIP_RTOL * delta^2 * Hf_jj (noise guard)
SKIP_SUFFIXES = ("in_proj_a", "in_proj_b")  # 48-wide GDN gates, left in bf16 as in official code
DATASET = "Salesforce/wikitext"

# Signed E2M1 grid as 15 levels; index 7 == 0. A "flip" is level +/- 1.
_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
SGRID = mx.array([-v for v in _POS[:0:-1]] + _POS)
ZERO, LMAX = 7, 14


# ---------------------------------------------------------------- transforms
def block_hadamard(x):
    """x @ B, B = blockdiag(H32/sqrt(32)). Orthonormal and symmetric."""
    shp = x.shape
    y = mx.hadamard_transform(x.reshape(*shp[:-1], shp[-1] // BLOCK, BLOCK))
    return y.reshape(shp)


def channel_scales(W, H, n_tok, alpha=ALPHA):
    """Eq. (9): t_j = sigma_j(W)^a / rms_j(x)^(1-a), rms from diag(H). alpha=None: t = 1."""
    if alpha is None:
        return mx.ones((W.shape[1],), dtype=mx.float32)
    sigma = mx.std(W.astype(mx.float32), axis=0)
    rms = mx.sqrt(mx.diag(H) / n_tok)
    t = sigma**alpha / mx.maximum(rms, 1e-8) ** (1 - alpha)
    return mx.clip(t, *T_CLAMP)


def transform_hessian(H, t, lam=LAM, rotate=True):
    """Eq. (5)-(6): Hf = B^T D H D B + lam * mean(diag) * I (B = I if not rotate)."""
    Hp = H * t[:, None] * t[None, :]
    if rotate:
        Hp = block_hadamard(Hp)            # (DHD) B
        Hp = block_hadamard(Hp.T).T        # B^T (DHD B)
    return Hp + lam * mx.mean(mx.diag(Hp)) * mx.eye(Hp.shape[0])


# ---------------------------------------------------------------- FP4 helpers
def dequant(L, e):
    """L: signed level idx (m, nb, 32); e: block exponent (m, nb)."""
    return mx.take(SGRID, L) * mx.power(2.0, e.astype(mx.float32))[..., None]


# E2M1 RTN boundaries with round-half-to-even (ties go to the even mantissa: 0, 1, 2, 4).
# Identical to MLX's Metal fp4_e2m1 conversion; the official Triton kernel rounds ties away.
_BOUNDS = [(0.25, False), (0.75, True), (1.25, False), (1.75, True),
           (2.5, False), (3.5, True), (5.0, False)]  # (boundary, tie rounds up)


def rtn_levels(Wb, e):
    """Round-to-nearest-even onto the E2M1 grid at block scale 2^e (saturating at +/-6)."""
    x = Wb / mx.power(2.0, e.astype(mx.float32))[..., None]
    a = mx.abs(x)
    mag = mx.zeros(a.shape, dtype=mx.int32)
    for b, tie_up in _BOUNDS:
        mag = mag + ((a >= b) if tie_up else (a > b)).astype(mx.int32)
    return ZERO + mx.where(x < 0, -mag, mag)


def block_exponent(amax, conv):
    """Shared E8M0 exponent of a block from its abs-max.
    ocp: floor(log2 amax) - 2   (paper / OCP spec / official kernel; may clip to 6*2^e)
    mlx: ceil(log2(amax / 6))   (mx.quantize mode="mxfp4"; never clips)"""
    amax = mx.maximum(amax.astype(mx.float32), 2.0**-126)
    if conv == "ocp":
        e = mx.floor(mx.log2(amax)) - 2
    elif conv == "mlx":
        e = mx.ceil(mx.log2(amax / 6.0))
    else:
        raise ValueError(conv)
    return mx.clip(e, -127, 127).astype(mx.int32)


def fake_quant_act(x, conv):
    """MXFP4 quantize->dequantize along the last axis; result in x.dtype (exact in bf16)."""
    if conv == "mlx":
        q, s = mx.quantize(x, group_size=BLOCK, bits=4, mode="mxfp4")
        return mx.dequantize(q, s, group_size=BLOCK, bits=4, mode="mxfp4", dtype=x.dtype)
    shp = x.shape
    xb = x.reshape(*shp[:-1], shp[-1] // BLOCK, BLOCK).astype(mx.float32)
    e = block_exponent(mx.max(mx.abs(xb), axis=-1), conv)
    return dequant(rtn_levels(xb, e), e).astype(x.dtype).reshape(shp)


def screen_exponents(Wp, hdiag, offsets=(1, 0, -1)):
    """Eq. (12): per row & block, pick e in e0+{-1,0,+1} minimising sum_j Hf_jj err_j^2."""
    m, n = Wp.shape
    Wb = Wp.reshape(m, n // BLOCK, BLOCK)
    hd = hdiag.reshape(1, n // BLOCK, BLOCK)
    e0 = block_exponent(mx.max(mx.abs(Wb), axis=-1), "ocp")  # paper's base exponent
    best = None
    for o in offsets:  # first-listed offset wins ties
        e = mx.clip(e0 + o, -127, 127)
        L = rtn_levels(Wb, e)
        err = mx.sum(hd * (Wb - dequant(L, e)) ** 2, axis=-1)
        if best is None:
            best = [e, L, err]
        else:
            b = err < best[2]
            best = [mx.where(b, e, best[0]), mx.where(b[..., None], L, best[1]), mx.minimum(err, best[2])]
        mx.eval(*best)
    return best[1].reshape(m, n), best[0]


def flip_refine_ref(Wp, Hf, L, e, K=128, recompute_every=32, rtol=FLIP_RTOL):
    """Alg. 1, pure-MLX reference: per row, take the single best +/-1 level move;
    g_i -= delta * Hf[:, j]. A move is accepted only if gain > rtol * delta^2 * Hf_jj,
    i.e. it is not fp32 noise relative to its own quadratic term (scale-invariant; the
    official code uses 1e-12 abs). Kept as the oracle for the Metal kernel below."""
    m, n = Wp.shape
    s = mx.repeat(mx.power(2.0, e.astype(mx.float32)), BLOCK, axis=1)  # (m, n)
    hdiag = mx.diag(Hf)[None, :]
    rows = mx.arange(m)
    grad = lambda L: (Wp - mx.take(SGRID, L) * s) @ Hf    # rows of Hf r^T (Hf symmetric)
    g = grad(L)
    k = -1
    for k in range(K):
        cur = mx.take(SGRID, L)
        up = mx.where(L < LMAX, (mx.take(SGRID, mx.minimum(L + 1, LMAX)) - cur) * s, 0.0)
        dn = mx.where(L > 0, (mx.take(SGRID, mx.maximum(L - 1, 0)) - cur) * s, 0.0)
        gu = 2 * up * g - up * up * hdiag            # Eq. (15)
        gd = 2 * dn * g - dn * dn * hdiag
        use_up = gu >= gd
        gain = mx.where(use_up, gu, gd)
        j = mx.argmax(gain, axis=1)[:, None]
        pick = lambda a: mx.take_along_axis(a, j, axis=1)[:, 0]
        dsel = pick(mx.where(use_up, up, dn))
        ok = pick(gain) > rtol * dsel * dsel * hdiag[0, j[:, 0]]
        ok = ok & (dsel != 0)
        delta = mx.where(ok, dsel, 0.0)
        step = mx.where(ok, mx.where(pick(use_up), 1, -1), 0)
        L = L.at[rows, j[:, 0]].add(step)
        g = g - delta[:, None] * Hf[j[:, 0]]         # Eq. (16)
        if (k + 1) % recompute_every == 0:
            g = grad(L)                              # limit fp32 drift
        mx.eval(L, g, ok)
        if not ok.any().item():
            break
    return L, k + 1


# One Flip-Refinement iteration as a single Metal kernel: one threadgroup per output row
# scans the row for the best move (first index wins ties, like mx.argmax), applies it to
# L, and does the rank-1 gradient update. Same arithmetic, in the same order, as
# flip_refine_ref, so the codes are bit-identical (checked by --selftest).
# 5-10x faster than the reference: one pass over memory instead of ~15.
_FLIP_TG = 256
_FLIP_HEADER = """
constant float SGV[15] = {-6.f,-4.f,-3.f,-2.f,-1.5f,-1.f,-0.5f,0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
"""
_FLIP_SRC = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    const uint TGS = threads_per_threadgroup.x;
    const int n = params[0];
    const float rtol = rtolp[0];
    const device uchar* Lr = L + (size_t)row * n;
    const device float* gr = g + (size_t)row * n;
    const device float* sr = sb + (size_t)row * (n / 32);

    float best = -INFINITY; int bj = 0x7fffffff; float bd = 0.f; int bdir = 0;
    for (int c = tid; c < n; c += TGS) {
        int l = Lr[c];
        float s = sr[c >> 5], gc = gr[c], h = hd[c];
        float up = (l < 14) ? (SGV[min(l + 1, 14)] - SGV[l]) * s : 0.f;
        float dn = (l > 0)  ? (SGV[max(l - 1, 0)] - SGV[l]) * s : 0.f;
        float gu = 2.f * up * gc - up * up * h;
        float gd = 2.f * dn * gc - dn * dn * h;
        bool u = gu >= gd;
        float gain = u ? gu : gd;
        if (gain > best) { best = gain; bj = c; bd = u ? up : dn; bdir = u ? 1 : -1; }
    }
    threadgroup float sg[256]; threadgroup int sj[256]; threadgroup float sd[256]; threadgroup int sdir[256];
    sg[tid] = best; sj[tid] = bj; sd[tid] = bd; sdir[tid] = bdir;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = TGS / 2; w > 0; w >>= 1) {
        if (tid < w) {
            float og = sg[tid + w]; int oj = sj[tid + w];
            if (og > sg[tid] || (og == sg[tid] && oj < sj[tid])) {
                sg[tid] = og; sj[tid] = oj; sd[tid] = sd[tid + w]; sdir[tid] = sdir[tid + w];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    int j = sj[0]; float delta = sd[0]; int dir = sdir[0];
    bool accept = (sg[0] > rtol * delta * delta * hd[j]) && (delta != 0.f);
    if (!accept) { delta = 0.f; dir = 0; }
    device uchar* Lo = L_out + (size_t)row * n;
    device float* go = g_out + (size_t)row * n;
    const device float* Hj = Hf + (size_t)j * n;   // row j == column j (Hf symmetric)
    for (int c = tid; c < n; c += TGS) {
        go[c] = accept ? gr[c] - delta * Hj[c] : gr[c];
        Lo[c] = (uchar)((int)Lr[c] + ((c == j) ? dir : 0));
    }
    if (tid == 0) ok[row] = accept ? 1 : 0;
"""
_flip_kernel = None


def _flip_step(L8, g, sb, hd, Hf, rtol):
    global _flip_kernel
    if _flip_kernel is None:
        _flip_kernel = mx.fast.metal_kernel(
            name="macaq_flip_step", input_names=["L", "g", "sb", "hd", "Hf", "params", "rtolp"],
            output_names=["L_out", "g_out", "ok"], source=_FLIP_SRC, header=_FLIP_HEADER)
    m, n = g.shape
    return _flip_kernel(inputs=[L8, g, sb, hd, Hf, mx.array([n], mx.int32), mx.array([rtol], mx.float32)],
                        grid=(m * _FLIP_TG, 1, 1), threadgroup=(_FLIP_TG, 1, 1),
                        output_shapes=[L8.shape, g.shape, (m,)],
                        output_dtypes=[mx.uint8, mx.float32, mx.uint8])


def flip_refine_metal(Wp, Hf, L, e, K=128, recompute_every=32, rtol=FLIP_RTOL):
    """Alg. 1 with the fused Metal step. Same semantics and stopping rule as flip_refine_ref."""
    s_blk = mx.power(2.0, e.astype(mx.float32))
    s = mx.repeat(s_blk, BLOCK, axis=1)
    hd = mx.diag(Hf)
    grad = lambda L8: (Wp - mx.take(SGRID, L8.astype(mx.int32)) * s) @ Hf
    L8 = L.astype(mx.uint8)
    g = grad(L8)
    mx.eval(L8, g)
    k = -1
    for k in range(K):
        L8, g, ok = _flip_step(L8, g, s_blk, hd, Hf, rtol)
        if (k + 1) % recompute_every == 0:
            g = grad(L8)
        mx.eval(L8, g, ok)
        if not ok.any().item():
            break
    return L8.astype(mx.int32), k + 1


FLIP_IMPL = "metal"


def flip_refine(*args, **kw):
    return (flip_refine_metal if FLIP_IMPL == "metal" else flip_refine_ref)(*args, **kw)


def objective(Wp, Hf, L, e):
    """Eq. (1) in transformed basis: sum_i r_i Hf r_i^T."""
    m, n = Wp.shape
    r = Wp - dequant(L.reshape(m, n // BLOCK, BLOCK), e).reshape(m, n)
    return mx.sum((r @ Hf) * r).item()


def pack_mxfp4(L, e):
    """Pack to MLX's mxfp4 layout: nibble = sign<<3 | mag, 8 per uint32 (low first); E8M0 = e+127."""
    code = ((L < ZERO).astype(mx.uint32) << 3) | mx.abs(L - ZERO).astype(mx.uint32)
    m, n = code.shape
    shifts = mx.arange(8, dtype=mx.uint32) * 4
    packed = mx.sum(code.reshape(m, n // 8, 8) << shifts, axis=-1).astype(mx.uint32)
    return packed, (e + 127).astype(mx.uint8)


# ---------------------------------------------------------------- modules
class HessAcc:
    """H = sum X^T X (fp32) and the row count N; shared by all linears that read the same input."""

    def __init__(self, n):
        self.H, self.N = mx.zeros((n, n), dtype=mx.float32), 0


class HessianCatcher(nn.Module):
    """Wraps a Linear and feeds a HessAcc. Optionally aborts the forward.
    `shared` is a per-forward dict {id(x): (x, acc)}. A linear fed the same x object as an
    earlier one in the same forward (q/k/v, gate/up, GDN in_proj_*) joins that
    accumulator: X^T X is computed and stored once per distinct input. The dict holds a
    reference to x, so ids can't be recycled while it lives; clear it after every forward."""

    def __init__(self, inner, shared, stop=False):
        super().__init__()
        self.inner, self.stop, self._shared, self.acc = inner, stop, shared, None

    def __call__(self, x):
        hit = self._shared.get(id(x))
        if hit is not None and hit[0] is x:
            if self.acc is None:
                self.acc = hit[1]
            assert self.acc is hit[1], "input sharing pattern changed between forwards"
        else:
            if self.acc is None:
                self.acc = HessAcc(x.shape[-1])
            xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
            self.acc.H = self.acc.H + xf.T @ xf
            self.acc.N += xf.shape[0]
            self._shared[id(x)] = (x, self.acc)
        if self.stop:
            raise StopForward
        return self.inner(x)


class StopForward(Exception):
    pass


class MACaQLinear(nn.Module):
    """y = [ (x*t) B ]_MXFP4 @ W'^T  (Eq. 7).
    act_exp: "mlx" | "ocp" activation exponent convention, or None for W4A16.
    t must be the same bf16 values the weights were divided by; t=None means no scale.
    rotate=False drops B (ablation: then W' = W / t)."""

    def __init__(self, wq, scales, t, act_exp="mlx", rotate=True):
        super().__init__()
        assert t is None or t.dtype == mx.bfloat16
        self.weight, self.scales, self.act_exp, self.rotate = wq, scales, act_exp, rotate
        if t is not None:
            self.t = t
        self.freeze()

    def __call__(self, x):
        xp = x * self.t.astype(x.dtype) if "t" in self else x
        if self.rotate:
            xp = block_hadamard(xp)
        if self.act_exp is not None:  # exact emulation; mx.qqmm is the fused (mlx-convention) path
            xp = fake_quant_act(xp, self.act_exp)
        return mx.quantized_matmul(xp, self.weight, self.scales, transpose=True,
                                   group_size=BLOCK, bits=4, mode="mxfp4")


class PlainMXFP4Linear(nn.Module):
    """RTN baseline: stock mx.quantize(W, mode="mxfp4") (what `mlx_lm.convert --q-mode mxfp4`
    produces), no scale, no rotation, no calibration. Optional activation fake-quant."""

    def __init__(self, W, act_exp="mlx"):
        super().__init__()
        self.weight, self.scales = mx.quantize(W, group_size=BLOCK, bits=4, mode="mxfp4")
        self.act_exp = act_exp
        self.freeze()

    def __call__(self, x):
        if self.act_exp is not None:
            x = fake_quant_act(x, self.act_exp)
        return mx.quantized_matmul(x, self.weight, self.scales, transpose=True,
                                   group_size=BLOCK, bits=4, mode="mxfp4")


def quantize_linear(W, H, n_tok, K=128, act_exp="mlx", verbose=True, alpha=ALPHA, rotate=True):
    tic = time.time()
    # Round t to bf16 *first*: inference multiplies x by bf16 t, so W must be divided
    # by exactly the same values or the transform no longer cancels.
    t16 = channel_scales(W, H, n_tok, alpha).astype(mx.bfloat16)
    t = t16.astype(mx.float32)
    if verbose and alpha is not None:
        lo, hi = T_CLAMP
        nclamp = mx.sum((t <= lo * 1.01) | (t >= hi * 0.99)).item()
        print(f"    t: min {t.min().item():.3e} median {mx.sort(t)[t.size // 2].item():.3e} "
              f"max {t.max().item():.3e} clamped {nclamp}")
    Wp = W.astype(mx.float32) / t                             # W' = (W / t) B
    if rotate:
        Wp = block_hadamard(Wp)
    Hf = transform_hessian(H, t, rotate=rotate)
    mx.eval(t, Wp, Hf)
    t_prep = time.time() - tic; tic = time.time()
    L, e = screen_exponents(Wp, mx.diag(Hf))
    mx.eval(L, e)
    t_screen = time.time() - tic
    obj_rtn = objective(Wp, Hf, L, e)   # one (m,n)x(n,n) matmul; cheap next to the flips
    tic = time.time()
    L, iters = flip_refine(Wp, Hf, L, e, K=K)
    t_flip = time.time() - tic
    obj = objective(Wp, Hf, L, e)
    ref = mx.sum((Wp @ Hf) * Wp).item()
    if verbose:
        print(f"    rel.err RTN {obj_rtn/ref:.3e} -> flips {obj/ref:.3e}  ({iters} iters)")
        print(f"    time prep {t_prep:.2f}s  screen {t_screen:.2f}s  flips {t_flip:.2f}s "
              f"({1e3 * t_flip / max(iters, 1):.1f} ms/iter)")
    wq, sc = pack_mxfp4(L, e)
    q = MACaQLinear(wq, sc, None if alpha is None else t16, act_exp=act_exp, rotate=rotate)
    if True:
        # plain attribute (not a Module child), so it never enters the parameter tree
        object.__setattr__(q, "stats", dict(rel_rtn=obj_rtn / ref, rel_flip=obj / ref,
                                            iters=iters, flip_secs=t_flip))
    return q


# ---------------------------------------------------------------- driver
GB = 1e9


def mem(tag):
    """One-line MLX memory report (active / allocator cache / peak), plus the budget guard."""
    act, cache = mx.get_active_memory(), mx.get_cache_memory()
    print(f"  [mem] {tag:24s} active {act/GB:6.1f} GB  cache {cache/GB:5.1f} GB  "
          f"peak {mx.get_peak_memory()/GB:6.1f} GB", flush=True)
    if MEM_BUDGET and act + cache > MEM_BUDGET:
        raise MemoryError(f"MLX memory {(act+cache)/GB:.1f} GB > budget {MEM_BUDGET/GB:.0f} GB at '{tag}'")


MEM_BUDGET = 0


def setup_memory(budget_gb, cache_gb):
    """Keep the process from hogging unified memory:
    - cap MLX's free-buffer cache (default would grow to the memory limit; we saw 39 GB),
    - make MLX's memory limit the budget so it reclaims cache / waits before exceeding it,
    - abort cleanly from mem() if active+cache passes the budget anyway.
    The wired limit is left at 0 deliberately: wiring the model would make the OS
    unable to page us out, so a co-running GPU job would more likely take the machine down."""
    global MEM_BUDGET
    MEM_BUDGET = int(budget_gb * GB)
    mx.set_memory_limit(MEM_BUDGET)
    mx.set_cache_limit(int(cache_gb * GB))
    print(f"memory: budget {budget_gb:.0f} GB, cache cap {cache_gb:.0f} GB "
          f"(system {mx.device_info()['memory_size']/GB:.0f} GB)")


def get_layers(model):
    for path in ("layers", "model.layers", "language_model.model.layers", "language_model.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not find decoder layers; inspect the model and add the path")


def token_seqs(tok, split, n_tok, seqlen):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset(DATASET, "wikitext-2-raw-v1", split=split)["text"])
    ids = mx.array(tok.encode(text))
    n = min(n_tok // seqlen, ids.size // seqlen)
    return ids[: n * seqlen].reshape(n, seqlen)


def logprobs(model, s):
    """(L, V) fp32 log-probs for one sequence."""
    return nn.log_softmax(model(s[None]).astype(mx.float32), axis=-1)[0]


def ref_topk(model, seqs, k):
    """Reference distribution kept as top-k (idx, logp) per position. Memory: T*k*8 B
    instead of T*V*4 B (4096 x 248k = 4 GB). k=0 keeps the full fp32 rows."""
    out = []
    for s in seqs:
        lp = logprobs(model, s)
        if k:
            # contiguous: a slice view would keep the whole (L, V) argpartition buffer alive
            idx = mx.contiguous(mx.argpartition(-lp, k - 1, axis=-1)[:, :k].astype(mx.int32))
            out.append((idx, mx.contiguous(mx.take_along_axis(lp, idx, axis=-1))))
        else:
            out.append((None, lp))
        mx.eval(out[-1])
        del lp
        mx.clear_cache()
    return out


def kld_vs_ref(model, seqs, ref):
    """Per-position KL(P_ref || Q). With top-k refs this is the KL of the (k+1)-bin
    coarsening {top-k tokens, rest}: exact when the tail is negligible, and a lower
    bound on the full KL otherwise (data-processing inequality)."""
    klds, agree = [], []
    for s, (idx, lp_r) in zip(seqs, ref):
        lp_q = logprobs(model, s)
        top_q = mx.argmax(lp_q, axis=-1)
        if idx is None:
            kl = mx.sum(mx.exp(lp_r) * (lp_r - lp_q), axis=-1)
            top_r = mx.argmax(lp_r, axis=-1)
        else:
            lq = mx.take_along_axis(lp_q, idx, axis=-1)
            pr = mx.exp(lp_r)
            tail_r = mx.maximum(1.0 - mx.sum(pr, -1), 1e-12)
            tail_q = mx.maximum(1.0 - mx.sum(mx.exp(lq), -1), 1e-12)
            kl = mx.sum(pr * (lp_r - lq), -1) + tail_r * (mx.log(tail_r) - mx.log(tail_q))
            top_r = mx.take_along_axis(idx, mx.argmax(lp_r, -1, keepdims=True), -1)[:, 0]
        klds.append(kl)
        agree.append(top_r == top_q)
        mx.eval(klds[-1], agree[-1])
        del lp_q
        mx.clear_cache()
    return mx.concatenate(klds), mx.concatenate(agree)


def report_kld(kld, agree, tag=""):
    ks = mx.sort(kld)
    p99 = ks[int(0.99 * (ks.size - 1))].item()
    r = dict(kld=kld.mean().item(), p99=p99, top1=agree.astype(mx.float32).mean().item())
    print(f"KLD{tag} mean {r['kld']:.4e}  p99 {r['p99']:.4e}  top1-agree {r['top1']:.4f}", flush=True)
    return r


def quantizable(name, lin):
    return not (lin.weight.shape[1] % BLOCK or name.endswith(SKIP_SUFFIXES)
                or getattr(lin, "bias", None) is not None)


def block_linears(block):
    return [(n, m) for n, m in block.named_modules() if isinstance(m, nn.Linear)]


def calibrate(model, layers, block_ids, calib_seqs):
    """One pass over the calibration set collecting Hessians for every quantizable linear
    of the blocks in `block_ids`, on the *unmodified* model's activations. The forward is
    aborted after the last block's down_proj. The original modules are put back before
    returning. Returns {block_id: [(name, linear, HessAcc)]}."""
    shared, catchers = {}, {}
    last = max(block_ids)
    for b in block_ids:
        lins = [(n, m) for n, m in block_linears(layers[b]) if quantizable(n, m)]
        assert lins, f"no quantizable nn.Linear in block {b} (already quantized?)"
        stop = next((n for n, _ in lins if n.endswith("down_proj")), lins[-1][0]) if b == last else None
        catchers[b] = {n: HessianCatcher(m, shared, stop=(n == stop)) for n, m in lins}
        layers[b].update_modules(tree_unflatten(list(catchers[b].items())))
    try:
        for s in calib_seqs:
            try:
                model(s[None])
            except StopForward:
                pass
            mx.eval([c.acc.H for cs in catchers.values() for c in cs.values()])
            shared.clear()
            mx.clear_cache()
    finally:
        for b, cs in catchers.items():
            layers[b].update_modules(tree_unflatten([(n, c.inner) for n, c in cs.items()]))
    return {b: [(n, c.inner, c.acc) for n, c in cs.items()] for b, cs in catchers.items()}


def quantize_block(lins, method, K, act_exp, verbose=True, alpha=ALPHA, rotate=True):
    """Quantize one block's linears from its Hessians. Frees each HessAcc once every linear
    reading it is done. Returns [(name, new_module)] and per-linear stats."""
    left = {}
    for _, _, acc in lins:
        left[id(acc)] = left.get(id(acc), 0) + 1
    new, stats = [], []
    for n, lin, acc in lins:
        t0 = time.time()
        if verbose:
            print(f"  {n} {tuple(lin.weight.shape)}")
        if method == "rtn":
            q = PlainMXFP4Linear(lin.weight, act_exp=act_exp)
        else:
            q = quantize_linear(lin.weight, acc.H, acc.N, K=K, act_exp=act_exp, verbose=verbose,
                                alpha=alpha, rotate=rotate)
        mx.eval(q.parameters())
        new.append((n, q))
        stats.append(dict(name=n, shape=list(lin.weight.shape), secs=time.time() - t0,
                          **getattr(q, "stats", {})))
        left[id(acc)] -= 1
        if left[id(acc)] == 0 and acc is not None:
            acc.H = None
        mx.clear_cache()
        if verbose:
            print(f"    {time.time()-t0:.1f}s")
            mem(n)
    return new, stats


def main(a):
    from mlx_lm import load
    setup_memory(a.mem_budget_gb, a.cache_gb)
    act_exp = None if a.w4a16 else a.act_exp
    print(f"mode: {'W4A16' if act_exp is None else 'W4A4 act-exp=' + act_exp}  method {a.method}  "
          f"TF32={'on' if a.tf32 else 'off'}  layer {a.layer}  K={a.flips}  calib={a.calib_tokens}")
    t0 = time.time()
    model, tok = load(a.model)
    print(f"loaded {a.model} in {time.time()-t0:.1f}s")
    mem("model loaded")
    layers = get_layers(model)
    block = layers[a.layer]

    eval_seqs = token_seqs(tok, "test", a.eval_seqs * a.eval_len, a.eval_len)
    t0 = time.time()
    lp_ref = ref_topk(model, eval_seqs, a.ref_topk)
    print(f"reference: {eval_seqs.size} positions, top-k={a.ref_topk or 'full'} in {time.time()-t0:.1f}s")
    mem("reference logprobs")
    if a.no_quant:  # noise floor: unmodified model vs its own reference
        report_kld(*kld_vs_ref(model, eval_seqs, lp_ref), tag=" (bf16 vs bf16)")
        return

    # 1) Hessians for this block, on the unmodified model's activations
    t0 = time.time()
    lins = calibrate(model, layers, [a.layer], token_seqs(tok, "train", a.calib_tokens, a.seqlen))[a.layer]
    accs = {id(acc): acc for _, _, acc in lins}
    print(f"hessians: {len(lins)} linears / {len(accs)} distinct inputs, "
          f"{lins[0][2].N} tokens, {time.time()-t0:.1f}s")
    mem("hessians")

    # 2) quantize each linear independently and swap it in
    for n, m in block_linears(block):
        if not quantizable(n, m):
            print(f"  skip {n} {tuple(m.weight.shape)}")
    new, _ = quantize_block(lins, a.method, a.flips, act_exp)
    block.update_modules(tree_unflatten(new))
    del lins, accs, new
    mx.clear_cache()

    # 3) KLD vs bf16 (one sequence at a time; never materialise (T, V) for the quantized model)
    mx.reset_peak_memory()
    report_kld(*kld_vs_ref(model, eval_seqs, lp_ref))
    mem("kld")


def parse_blocks(spec, n):
    """'all' | '0-15' | '3,7,11' | '0-7,32'  ->  sorted list of block ids."""
    if spec == "all":
        return list(range(n))
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(b for b in out if 0 <= b < n)


def sweep(a):
    """Per-block sensitivity: quantize each block *alone*, measure KLD, restore it.
    One process: model + reference loaded once, Hessians collected for `--group` blocks
    per calibration pass (always on the unmodified bf16 model, so they are valid for any
    subset). Each block's codes are computed once and evaluated under every mode in
    --modes (weight codes don't depend on activation quantization). Results are appended
    to a JSONL file as each block finishes; rerunning with the same --out resumes."""
    import json
    from mlx_lm import load
    setup_memory(a.mem_budget_gb, a.cache_gb)
    modes = [None if m == "none" else m for m in a.modes.split(",")]
    label = lambda m: "w4a16" if m is None else f"w4a4_{m}"
    out = a.out or f"runs/sweep_{a.method}_K{a.flips}_c{a.calib_tokens}.jsonl"
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {json.loads(l)["block"] for l in f if l.strip()}
    t_start = time.time()
    model, tok = load(a.model)
    layers = get_layers(model)
    blocks = [b for b in parse_blocks(a.blocks, len(layers)) if b not in done]
    print(f"sweep -> {out}: {len(blocks)} blocks to do ({len(done)} already done), "
          f"modes {[label(m) for m in modes]}, method {a.method}, K={a.flips}, "
          f"calib {a.calib_tokens}, group {a.group}", flush=True)
    if not blocks:
        return
    eval_seqs = token_seqs(tok, "test", a.eval_seqs * a.eval_len, a.eval_len)
    lp_ref = ref_topk(model, eval_seqs, a.ref_topk)
    calib = token_seqs(tok, "train", a.calib_tokens, a.seqlen)
    mem("model + reference")

    for gi in range(0, len(blocks), a.group):
        group = blocks[gi:gi + a.group]
        t0 = time.time()
        hess = calibrate(model, layers, group, calib) if a.method == "macaq" else {
            b: [(n, m, None) for n, m in block_linears(layers[b]) if quantizable(n, m)] for b in group}
        t_cal = time.time() - t0
        print(f"\n== calibrated blocks {group[0]}..{group[-1]} in {t_cal:.1f}s", flush=True)
        mem("hessians")
        for b in group:
            t0 = time.time()
            print(f"-- block {b} ({'linear' if getattr(layers[b], 'is_linear', False) else 'full'} attention)")
            new, stats = quantize_block(hess.pop(b), a.method, a.flips, modes[0], verbose=a.verbose)
            t_q = time.time() - t0
            orig = [(n, m) for n, m in block_linears(layers[b]) if n in dict(new)]
            layers[b].update_modules(tree_unflatten(new))
            rec = dict(block=b, kind="linear" if getattr(layers[b], "is_linear", False) else "full",
                       method=a.method, K=a.flips, calib_tokens=a.calib_tokens,
                       quant_secs=round(t_q, 2), calib_secs_group=round(t_cal, 2), linears=stats)
            try:
                for m in modes:
                    for _, q in new:
                        q.act_exp = m
                    t1 = time.time()
                    rec[label(m)] = report_kld(*kld_vs_ref(model, eval_seqs, lp_ref), tag=f" [{label(m)}]")
                    rec[label(m)]["secs"] = round(time.time() - t1, 2)
            finally:
                layers[b].update_modules(tree_unflatten(orig))   # back to bf16
            del new, orig
            mx.clear_cache()
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            el = time.time() - t_start
            print(f"   block {b} done in {time.time()-t0:.1f}s (quant {t_q:.1f}s); "
                  f"elapsed {el/60:.1f} min", flush=True)
            mem(f"after block {b}")
        del hess
        mx.clear_cache()
    print(f"\nsweep finished in {(time.time()-t_start)/60:.1f} min -> {out}")


# ---------------------------------------------------------------- full model
def text_core(model):
    """The decoder stack object (embed_tokens, layers, norm) and the head, for qwen3_5."""
    lm = model.language_model
    assert hasattr(lm.model, "embed_tokens") and not lm.args.tie_word_embeddings
    return lm.model


def stream_quantize(model, calib, K, act_exp, verbose=False):
    """Layer-wise streaming calibration + MACaQ quantization of every block.
    Each calibration sequence's hidden state at the current block's input is cached
    (bf16, from the *unmodified* model). For each block we wrap its linears in
    HessianCatchers and run the bf16 block on every cached hidden state. That one call
    yields both the Hessians and the next block's inputs. Then we quantize and swap in.
    This is the same arithmetic as calibrate() on the bf16 model (the manual layer loop
    matches model() exactly), but costs one forward in total instead of one per group.
    Returns {(block, name): quantized module}, per-block stats."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    core = text_core(model)
    layers = core.layers
    hs = [core.embed_tokens(s[None]) for s in calib]
    mx.eval(hs)
    fa, ssm = create_attention_mask(hs[0], None), create_ssm_mask(hs[0], None)
    qmods, stats = {}, []
    for b, layer in enumerate(layers):
        t0 = time.time()
        shared = {}
        cat = {n: HessianCatcher(m, shared) for n, m in block_linears(layer) if quantizable(n, m)}
        layer.update_modules(tree_unflatten(list(cat.items())))
        try:
            nxt = []
            for h in hs:
                o = layer(h, mask=ssm if layer.is_linear else fa)
                mx.eval(o, [c.acc.H for c in cat.values()])
                shared.clear()
                nxt.append(o)
        finally:
            layer.update_modules(tree_unflatten([(n, c.inner) for n, c in cat.items()]))
        hs = nxt
        t_cal = time.time() - t0
        new, st = quantize_block([(n, c.inner, c.acc) for n, c in cat.items()], "macaq", K, act_exp,
                                 verbose=verbose)
        layer.update_modules(tree_unflatten(new))
        qmods.update({(b, n): q for n, q in new})
        del cat, new
        mx.clear_cache()
        stats.append(dict(block=b, calib_secs=round(t_cal, 2), quant_secs=round(time.time() - t0 - t_cal, 2),
                          linears=st))
        print(f"  block {b:2d} calib {t_cal:4.1f}s quant {time.time()-t0-t_cal:4.1f}s  "
              f"active {mx.get_active_memory()/GB:5.1f} GB", flush=True)
        mem(f"block {b}") if b % 16 == 15 else None
    del hs
    mx.clear_cache()
    return qmods, stats


def wikitext_ppl(model, tok, seqlen=2048):
    """WikiText-2 test perplexity over non-overlapping seqlen windows plus the tail
    (if >= 10 tokens), like MACaQ/quantize/mxfp4_helper.evaluate_wikitext2. Each window
    scores L-1 next-token predictions; the official code weights each window's mean
    loss by L instead, which is a negligible difference at L = 2048."""
    from datasets import load_dataset
    ids = mx.array(tok.encode("\n\n".join(load_dataset(DATASET, "wikitext-2-raw-v1", split="test")["text"])))
    n = ids.size // seqlen
    wins = [ids[i * seqlen:(i + 1) * seqlen] for i in range(n)]
    if ids.size - n * seqlen >= 10:
        wins.append(ids[n * seqlen:])
    nll, cnt = 0.0, 0
    for w in wins:
        logits = model(w[None])[0, :-1].astype(mx.float32)
        tgt = w[1:]
        l = mx.sum(mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, tgt[:, None], -1)[:, 0])
        nll += l.item()
        cnt += tgt.size
        del logits
        mx.clear_cache()
    return float(mx.exp(mx.array(nll / cnt)).item())


def full(a):
    """Quantize every block, then evaluate a list of activation-precision variants of
    the same codes. Weight codes don't depend on activation quantization, so every
    variant is just a different act_exp per linear: no requantization."""
    import json
    from mlx_lm import load
    setup_memory(a.mem_budget_gb, a.cache_gb)
    t_start = time.time()
    model, tok = load(a.model)
    core = text_core(model)
    layers = core.layers
    eval_seqs = token_seqs(tok, "test", a.eval_seqs * a.eval_len, a.eval_len)
    lp_ref = ref_topk(model, eval_seqs, a.ref_topk)
    mem("model + reference")
    res = dict(method=a.method, K=a.flips, calib_tokens=a.calib_tokens, eval_positions=int(eval_seqs.size),
               variants={}, ppl={})
    out = a.out or f"runs/full_{a.method}_K{a.flips}_c{a.calib_tokens}.json"
    want_ppl = [v for v in a.ppl.split(",") if v]

    ppl_cache = "runs/ppl_bf16.json"
    if "bf16" in want_ppl:
        if os.path.exists(ppl_cache):
            res["ppl"]["bf16"] = json.load(open(ppl_cache))["ppl"]
        else:
            t0 = time.time()
            res["ppl"]["bf16"] = wikitext_ppl(model, tok, a.seqlen)
            json.dump(dict(ppl=res["ppl"]["bf16"], seqlen=a.seqlen), open(ppl_cache, "w"))
            print(f"bf16 PPL {res['ppl']['bf16']:.4f} ({time.time()-t0:.0f}s)")
        print(f"bf16 wikitext2 PPL: {res['ppl']['bf16']:.4f}", flush=True)

    # ---- quantize everything
    t0 = time.time()
    if a.method == "macaq":
        calib = token_seqs(tok, "train", a.calib_tokens, a.seqlen)
        qmods, res["blocks"] = stream_quantize(model, calib, a.flips, "mlx", verbose=a.verbose)
    else:
        qmods = {}
        for b, layer in enumerate(layers):
            new = []
            for n, m in block_linears(layer):
                if not quantizable(n, m):
                    continue
                q = (PlainMXFP4Linear(m.weight) if a.method == "rtn"
                     else nn.QuantizedLinear.from_linear(m, group_size=64, bits=4))  # affine 4-bit gs64
                mx.eval(q.parameters())
                new.append((n, q))
            layer.update_modules(tree_unflatten(new))
            qmods.update({(b, n): q for n, q in new})
            mx.clear_cache()
    res["quant_secs"] = round(time.time() - t0, 1)
    nbytes = sum(v.nbytes for q in qmods.values() for _, v in tree_flatten(q.parameters()))
    print(f"quantized {len(qmods)} linears in {res['quant_secs']/60:.1f} min; "
          f"quantized-linear storage {nbytes/GB:.2f} GB", flush=True)
    mem("quantized model")

    # ---- variants: name -> (default act, set of blocks or (block, name) at A16)
    can_a4 = a.method in ("macaq", "rtn")
    variants = {"w4a16": (None, set())}
    if can_a4:
        variants["w4a4_mlx"] = ("mlx", set())
        variants["w4a4_ocp"] = ("ocp", set())
    if a.method == "macaq":
        nb = len(layers)
        last = nb - 1
        variants[f"w4a4_mlx+A16@{last}"] = ("mlx", {last})
        if a.sweep_ref and os.path.exists(a.sweep_ref):
            S = [json.loads(l) for l in open(a.sweep_ref) if l.strip()]
            by_gain = [r["block"] for r in sorted(S, key=lambda r: r["w4a16"]["kld"] - r["w4a4_mlx"]["kld"])]
            res["a16_rank"] = by_gain
            for k in (2, 4, 8, 16):
                variants[f"w4a4_mlx+A16@top{k}"] = ("mlx", set(by_gain[:k]))
        for (b, n) in sorted(k for k in qmods if k[0] == last):
            variants[f"w4a4_mlx+A16@{last}.{n}"] = ("mlx", {(b, n)})
    params = {k: q.weight.shape[0] * q.weight.shape[1] * 8 if isinstance(q, (MACaQLinear, PlainMXFP4Linear)) else 0 for k, q in qmods.items()}
    tot_p = sum(params.values()) or 1

    def apply(base, a16):
        for (b, n), q in qmods.items():
            if hasattr(q, "act_exp"):
                q.act_exp = None if (base is None or b in a16 or (b, n) in a16) else base

    for name, (base, a16) in variants.items():
        apply(base, a16)
        t1 = time.time()
        r = report_kld(*kld_vs_ref(model, eval_seqs, lp_ref), tag=f" [{name}]")
        sel = [k for k in qmods if base is None or k[0] in a16 or k in a16]
        r.update(secs=round(time.time() - t1, 1), a16_linears=len(sel),
                 a16_param_frac=round(sum(params[k] for k in sel) / tot_p, 4))
        res["variants"][name] = r
        json.dump(res, open(out, "w"), indent=1)

    # ---- perplexity for selected variants ("best" = lowest-KLD mixed variant)
    mixed = {k: v for k, v in res["variants"].items() if "+A16" in k}
    for name in want_ppl:
        if name == "bf16":
            continue
        if name == "best":
            if not mixed:
                continue
            name = min(mixed, key=lambda k: mixed[k]["kld"])
        if name not in variants or name in res["ppl"]:
            continue
        apply(*variants[name])
        t1 = time.time()
        res["ppl"][name] = wikitext_ppl(model, tok, a.seqlen)
        print(f"PPL [{name}] {res['ppl'][name]:.4f}  ({time.time()-t1:.0f}s)", flush=True)
        json.dump(res, open(out, "w"), indent=1)
    res["total_secs"] = round(time.time() - t_start, 1)
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nfull run finished in {res['total_secs']/60:.1f} min -> {out}")


def selftest():
    """Random-matrix smoke check, not a test suite."""
    mx.random.seed(0)
    m, n, T = 96, 128, 4096
    X = mx.random.normal((T, n)) * mx.exp(mx.random.normal((n,)))   # uneven channels
    W = mx.random.normal((m, n)) * 0.02
    H = X.T @ X
    t = channel_scales(W, H, T)
    Xp, Wp = block_hadamard(X * t), block_hadamard(W / t)
    print("transform identity err:", mx.abs(Xp @ Wp.T - X @ W.T).max().item())
    print("hessian transform err:", mx.abs(transform_hessian(H, t, lam=0) - Xp.T @ Xp).max().item()
          / mx.abs(Xp.T @ Xp).max().item())
    t16 = t.astype(mx.bfloat16)
    Xb = X.astype(mx.bfloat16)
    y_ref = Xb.astype(mx.float32) @ W.T
    rel = lambda y: (mx.linalg.norm(y.astype(mx.float32) - y_ref) / mx.linalg.norm(y_ref)).item()
    for name, tw in (("fp32 t", t), ("bf16 t", t16.astype(mx.float32))):  # fix: W/t uses the bf16 t
        y = block_hadamard(Xb * t16) @ block_hadamard(W / tw).astype(mx.bfloat16).T
        print(f"bf16 inference transform, W divided by {name}: rel.err {rel(y):.3e}")

    L, e = screen_exponents(Wp, mx.diag(transform_hessian(H, t)))
    wq, sc = pack_mxfp4(L, e)
    mine = dequant(L.reshape(m, -1, BLOCK), e).reshape(m, n)
    theirs = mx.dequantize(wq, sc, group_size=BLOCK, bits=4, mode="mxfp4", dtype=mx.float32)
    print("pack round-trip err:", mx.abs(mine - theirs).max().item())

    # activation quantizer: our RTN with the mlx exponent rule must reproduce mx.quantize bit-exactly
    xa = (block_hadamard(Xb * t16) * 3.0).astype(mx.bfloat16)
    xa = mx.concatenate([xa, mx.zeros((1, n), mx.bfloat16)])                 # all-zero block
    xb = xa.reshape(-1, n // BLOCK, BLOCK).astype(mx.float32)
    amax = mx.max(mx.abs(xb), axis=-1)
    e_mlx = block_exponent(amax, "mlx")
    emu = dequant(rtn_levels(xb, e_mlx), e_mlx).reshape(xa.shape).astype(mx.bfloat16)
    print("mlx-convention emulation vs mx.quantize: max err",
          mx.abs(emu.astype(mx.float32) - fake_quant_act(xa, "mlx").astype(mx.float32)).max().item())
    diff = e_mlx - block_exponent(amax, "ocp")
    print(f"act exponent mlx-ocp: +1 on {mx.mean(diff == 1).item():.1%} of blocks, "
          f"0 on {mx.mean(diff == 0).item():.1%}")
    for conv in ("mlx", "ocp"):
        r = mx.linalg.norm((fake_quant_act(xa, conv) - xa).astype(mx.float32)) / mx.linalg.norm(xa.astype(mx.float32))
        print(f"act fake-quant rel.err ({conv}): {r.item():.4f}")

    # Metal flip kernel must reproduce the pure-MLX reference exactly (codes and iterations)
    Hf = transform_hessian(H, t)
    Lr, kr = flip_refine_ref(Wp, Hf, L, e, K=64)
    Lm, km = flip_refine_metal(Wp, Hf, L, e, K=64)
    print(f"flip metal vs ref: codes equal {mx.array_equal(Lr, Lm).item()}  iters {km} vs {kr}")

    lin = quantize_linear(W, H, T, K=64, act_exp=None)
    print(f"W4A16 output rel.err: {rel(lin(Xb)):.4f}")
    for conv in ("mlx", "ocp"):
        lin.act_exp = conv
        print(f"W4A4 ({conv}) output rel.err: {rel(lin(Xb)):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3.8-27B-bf16")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--calib-tokens", type=int, default=65536)   # paper: 2e5
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--flips", type=int, default=128)            # paper: K=128
    p.add_argument("--eval-seqs", type=int, default=8)
    p.add_argument("--eval-len", type=int, default=512)
    p.add_argument("--w4a16", action="store_true", help="skip activation quant")
    p.add_argument("--act-exp", choices=("mlx", "ocp"), default="mlx",
                   help="activation E8M0 rule: mlx=ceil(log2(amax/6)) (mx.quantize), "
                        "ocp=floor(log2 amax)-2 (paper/official)")
    p.add_argument("--tf32", action="store_true", help="allow TF32 for fp32 matmuls (faster, ~7e-4 rel. err)")
    p.add_argument("--ref-topk", type=int, default=0,
                   help="0 = exact full-vocab KLD (4 GB at 8x512); k>0 keeps top-k + tail bin (lower bound; k=256 read ~4%% low)")
    p.add_argument("--mem-budget-gb", type=float, default=80,
                   help="abort cleanly if MLX active+cache exceeds this (0 = off)")
    p.add_argument("--cache-gb", type=float, default=4, help="cap on MLX's free-buffer cache")
    p.add_argument("--method", choices=("macaq", "rtn", "affine"), default="macaq",
                   help="rtn = stock mx.quantize mxfp4 baseline (no t, no B, no calibration); "
                        "affine = nn.QuantizedLinear 4-bit gs64 (mlx_lm.convert default, W4A16; --full only)")
    p.add_argument("--no-quant", action="store_true",
                   help="noise floor: KLD of the unmodified model against its own reference")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--sweep", action="store_true", help="per-block sensitivity sweep (see sweep())")
    p.add_argument("--blocks", default="all", help="sweep: 'all' | '0-15' | '3,7,11'")
    p.add_argument("--group", type=int, default=6,
                   help="sweep: blocks per calibration pass (~1.6 GB of Hessians per block)")
    p.add_argument("--modes", default="mlx,ocp,none",
                   help="sweep: comma list of act modes to evaluate: mlx, ocp, none (=W4A16)")
    p.add_argument("--out", default=None, help="sweep: JSONL output (resumes if it exists)")
    p.add_argument("--verbose", action="store_true", help="sweep: per-linear prints")
    p.add_argument("--full", action="store_true", help="quantize all blocks and evaluate variants (see full())")
    p.add_argument("--ppl", default="bf16,w4a16,w4a4_mlx,best",
                   help="full: variants to also score with WikiText-2 PPL (~5 min each); '' = none")
    p.add_argument("--sweep-ref", default="runs/sweep_macaq_K128_c65536.jsonl",
                   help="full: per-block sweep used to rank blocks for A16 variants")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.full:
        full(a)
    elif a.method == "affine":
        p.error("--method affine is only supported with --full")
    elif a.sweep:
        sweep(a)
    else:
        main(a)
