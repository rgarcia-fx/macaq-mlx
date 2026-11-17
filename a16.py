"""
a16.py -- W4A16 pipeline (the deployable MLX path) on top of macaq_mlx.py.

    python a16.py ref                     # cache bf16 reference logprobs + PPL (runs/ref)
    python a16.py configs --calib 262144 --flips 512 --configs a1.0,a0.75,anone,a1.0-norot
    python a16.py greedy  --calib 262144 --flips 512 --config a1.0 --keep q8 --screen-set both
    python a16.py export  --calib 262144 --flips 512 --config a1.0 --calib-data mix
    python a16.py verify  --out-dir runs/export/<checkpoint>

Calibration is streamed once per (tokens, source) and the per-block Hessians are cached on
disk (runs/hess_*, 94 GB at any token count), so further configs reuse it.
Config syntax: a<alpha> | anone (t = 1) | <...>-norot (no Hadamard) | rtn | affine | affine8.
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

if "--tf32" not in sys.argv:
    os.environ["MLX_ENABLE_TF32"] = "0"
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

import macaq_mlx as M

REF_DIR = "runs/ref"
C4_FILE = "data/c4-validation.00000-of-00008.json.gz"
GB = 1e9


# ---------------------------------------------------------------- data
def c4_seqs(tok, n_seqs, seqlen, seed=0):
    """GPTQ-style C4 eval: random seqlen windows from validation docs longer than seqlen."""
    import random
    rng = random.Random(seed)
    docs = [json.loads(l)["text"] for l in gzip.open(C4_FILE, "rt")]
    out = []
    while len(out) < n_seqs:
        ids = tok.encode(docs[rng.randrange(len(docs))])
        if len(ids) <= seqlen:
            continue
        i = rng.randrange(len(ids) - seqlen)
        out.append(ids[i:i + seqlen])
    return mx.array(out)


def eval_sets(tok, a):
    return {"wiki": M.token_seqs(tok, "test", a.eval_seqs * a.eval_len, a.eval_len),
            "c4": c4_seqs(tok, a.eval_seqs, a.eval_len)}


# ---------------------------------------------------------------- reference cache
def build_ref(model, tok, a):
    """Full-vocab fp32 reference logprobs per eval sequence (on disk), and bf16 PPL."""
    os.makedirs(REF_DIR, exist_ok=True)
    meta_path = f"{REF_DIR}/meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    key = f"{a.eval_seqs}x{a.eval_len}"
    for name, seqs in eval_sets(tok, a).items():
        tag = f"{name}_{key}"
        if tag in meta:
            continue
        t0 = time.time()
        for i, s in enumerate(seqs):
            lp = M.logprobs(model, s)
            mx.save(f"{REF_DIR}/{tag}_{i}.npy", lp)
            del lp
            mx.clear_cache()
        meta[tag] = dict(n=int(seqs.shape[0]), len=int(seqs.shape[1]),
                         tokens_sha=int(mx.sum(seqs.astype(mx.int64) * (mx.arange(seqs.size).reshape(seqs.shape) + 1)).item()))
        print(f"ref {tag}: {seqs.shape[0]} seqs in {time.time()-t0:.0f}s", flush=True)
        json.dump(meta, open(meta_path, "w"), indent=1)
    if "ppl_wiki" not in meta:
        meta["ppl_wiki"] = ppl(model, tok, a.ppl_len)
        print(f"bf16 wikitext2 PPL {meta['ppl_wiki']:.4f}")
        json.dump(meta, open(meta_path, "w"), indent=1)
    if "ppl_c4" not in meta:
        meta["ppl_c4"] = ppl_c4(model, tok, a.ppl_len)
        print(f"bf16 C4 PPL {meta['ppl_c4']:.4f}")
        json.dump(meta, open(meta_path, "w"), indent=1)
    return meta


class Ref:
    """Streams reference logprobs from disk, one sequence at a time."""

    def __init__(self, tok, a):
        meta = json.load(open(f"{REF_DIR}/meta.json"))
        self.key = f"{a.eval_seqs}x{a.eval_len}"
        self.sets = eval_sets(tok, a)
        for name, seqs in self.sets.items():
            m = meta[f"{name}_{self.key}"]
            chk = int(mx.sum(seqs.astype(mx.int64) * (mx.arange(seqs.size).reshape(seqs.shape) + 1)).item())
            assert m["tokens_sha"] == chk, f"eval tokens for {name} changed; rebuild the reference"
        self.meta = meta

    def kld(self, model, name):
        klds, agree = [], []
        for i, s in enumerate(self.sets[name]):
            lp_r = mx.load(f"{REF_DIR}/{name}_{self.key}_{i}.npy")
            lp_q = M.logprobs(model, s)
            kl = mx.sum(mx.exp(lp_r) * (lp_r - lp_q), axis=-1)
            ag = mx.argmax(lp_r, -1) == mx.argmax(lp_q, -1)
            mx.eval(kl, ag)
            klds.append(kl); agree.append(ag)
            del lp_r, lp_q
            mx.clear_cache()
        kl, ag = mx.concatenate(klds), mx.concatenate(agree)
        ks = mx.sort(kl)
        return dict(kld=kl.mean().item(), p99=ks[int(0.99 * (ks.size - 1))].item(),
                    top1=ag.astype(mx.float32).mean().item())

    def score(self, model, tag=""):
        r = {n: self.kld(model, n) for n in self.sets}
        print(f"  [{tag}] KLD wiki {r['wiki']['kld']:.4e} (top1 {r['wiki']['top1']:.4f})  "
              f"c4 {r['c4']['kld']:.4e} (top1 {r['c4']['top1']:.4f})", flush=True)
        return r


def _nll(model, wins):
    """Sum of next-token NLL over windows (one window per forward)."""
    nll, cnt = 0.0, 0
    for w in wins:
        logits = model(w[None])[0, :-1].astype(mx.float32)
        tgt = w[1:]
        l = mx.sum(mx.logsumexp(logits, -1) - mx.take_along_axis(logits, tgt[:, None], -1)[:, 0])
        nll += l.item(); cnt += tgt.size
        del logits
        mx.clear_cache()
    return nll, cnt


def ppl(model, tok, seqlen):
    from datasets import load_dataset
    ids = mx.array(tok.encode("\n\n".join(load_dataset(M.DATASET, "wikitext-2-raw-v1", split="test")["text"])))
    n = ids.size // seqlen
    wins = [ids[i * seqlen:(i + 1) * seqlen] for i in range(n)]
    if ids.size - n * seqlen >= 10:
        wins.append(ids[n * seqlen:])
    nll, cnt = _nll(model, wins)
    return float(mx.exp(mx.array(nll / cnt)).item())


def ppl_c4(model, tok, seqlen, n=128):
    """C4 PPL on 128 random validation windows (GPTQ/AWQ convention uses 256 x 2048)."""
    nll, cnt = _nll(model, list(c4_seqs(tok, n, seqlen, seed=1)))
    return float(mx.exp(mx.array(nll / cnt)).item())


# ---------------------------------------------------------------- configs
def parse_config(s):
    """'a0.75' | 'a1.0' | 'anone' | 'a0.75-norot' | 'rtn' | 'affine' | 'affine8'  (K/calib from CLI)"""
    if s in ("rtn", "affine", "affine8"):
        return dict(name=s, method=s)
    rotate = not s.endswith("-norot")
    a = s.removesuffix("-norot")
    assert a.startswith("a"), s
    alpha = None if a == "anone" else float(a[1:])
    return dict(name=s, method="macaq", alpha=alpha, rotate=rotate)


def stock_quant(lin, method):
    if method == "rtn":
        return M.PlainMXFP4Linear(lin.weight, act_exp=None)
    bits = 8 if method == "affine8" else 4
    return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)


C4_TRAIN = "data/c4-train.00000-of-01024.json.gz"


def c4_train_seqs(tok, n_tok, seqlen, seed=0):
    """C4 *train* calibration text: shuffled docs joined with '\n\n', cut into seqlen rows.
    (Disjoint from the C4 validation eval set.)"""
    import random
    docs = [json.loads(l)["text"] for l in gzip.open(C4_TRAIN, "rt")]
    random.Random(seed).shuffle(docs)
    ids, need = [], (n_tok // seqlen) * seqlen
    for d in docs:
        ids.extend(tok.encode(d + "\n\n"))
        if len(ids) >= need:
            break
    return mx.array(ids[:need]).reshape(-1, seqlen)


def calib_seqs(tok, source, n_tok, seqlen):
    """'wiki' (WikiText-2 train, the default), 'c4' (C4 train), 'mix' (half each, interleaved)."""
    if source == "wiki":
        return M.token_seqs(tok, "train", n_tok, seqlen)
    if source == "c4":
        return c4_train_seqs(tok, n_tok, seqlen)
    if source == "mix":
        w = M.token_seqs(tok, "train", n_tok // 2, seqlen)
        c = c4_train_seqs(tok, n_tok // 2, seqlen)
        return mx.stack([r for pair in zip(w, c) for r in pair])
    raise ValueError(source)


def calib_inputs(model, tok, calib_tokens, seqlen, source="wiki"):
    """Per-block Hessians for the bf16 model, computed once and cached on disk.
    Returns a loader: block -> [(name, HessAcc)] with shared accumulators."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    d = f"runs/hess_c{calib_tokens}" + ("" if source == "wiki" else f"_{source}")
    os.makedirs(d, exist_ok=True)
    core = M.text_core(model)
    layers = core.layers
    done = os.path.exists(f"{d}/done")
    if not done:
        t0 = time.time()
        calib = calib_seqs(tok, source, calib_tokens, seqlen)
        assert calib.size == calib_tokens, (calib.shape, calib_tokens)
        hs = [core.embed_tokens(s[None]) for s in calib]
        mx.eval(hs)
        fa, ssm = create_attention_mask(hs[0], None), create_ssm_mask(hs[0], None)
        for b, layer in enumerate(layers):
            shared = {}
            cat = {n: M.HessianCatcher(m, shared) for n, m in M.block_linears(layer) if M.quantizable(n, m)}
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
            accs, names = {}, {}
            for n, c in cat.items():
                accs.setdefault(id(c.acc), c.acc)
                names[n] = id(c.acc)
            ids = {k: i for i, k in enumerate(accs)}
            mx.save_safetensors(f"{d}/block{b:02d}.safetensors",
                                {f"H{ids[k]}": acc.H for k, acc in accs.items()},
                                metadata={"names": json.dumps({n: ids[k] for n, k in names.items()}),
                                          "N": json.dumps({ids[k]: acc.N for k, acc in accs.items()})})
            del cat, accs
            mx.clear_cache()
            if b % 8 == 7:
                print(f"  calib block {b} ({time.time()-t0:.0f}s)", flush=True)
        del hs
        mx.clear_cache()
        open(f"{d}/done", "w").write(str(time.time() - t0))
        print(f"calibration ({calib_tokens} tokens) cached in {time.time()-t0:.0f}s -> {d}")

    def load_block(b):
        H, md = mx.load(f"{d}/block{b:02d}.safetensors", return_metadata=True)
        names, N = json.loads(md["names"]), json.loads(md["N"])
        accs = {}
        for i in set(names.values()):
            acc = M.HessAcc.__new__(M.HessAcc)
            acc.H, acc.N = H[f"H{i}"], N[str(i)]
            accs[i] = acc
        lin = dict(M.block_linears(layers[b]))
        return [(n, lin[n], accs[i]) for n, i in names.items()]
    return load_block


def quantize_model(model, cfg, load_block, K, verbose=False):
    """Replace every quantizable linear in place. Returns {(b, name): (orig, quant)}."""
    layers = M.text_core(model).layers
    swaps, t0 = {}, time.time()
    for b, layer in enumerate(layers):
        if cfg["method"] == "macaq":
            new, _ = M.quantize_block(load_block(b), "macaq", K, None, verbose=verbose,
                                      alpha=cfg["alpha"], rotate=cfg["rotate"])
        else:
            new = [(n, stock_quant(m, cfg["method"])) for n, m in M.block_linears(layer) if M.quantizable(n, m)]
            mx.eval([q.parameters() for _, q in new])
        orig = dict(M.block_linears(layer))
        for n, q in new:
            swaps[(b, n)] = (orig[n], q)
        layer.update_modules(tree_unflatten(new))
        mx.clear_cache()
    return swaps, time.time() - t0


def restore(model, swaps, keys=None):
    layers = M.text_core(model).layers
    by_block = {}
    for (b, n), (o, q) in swaps.items():
        if keys is None or (b, n) in keys:
            by_block.setdefault(b, []).append((n, o))
    for b, mods in by_block.items():
        layers[b].update_modules(tree_unflatten(mods))


def apply(model, swaps, keep):
    """Set each linear to quantized, or kept (orig bf16 / 8-bit) if in `keep` dict."""
    layers = M.text_core(model).layers
    by_block = {}
    for (b, n), (o, q) in swaps.items():
        by_block.setdefault(b, []).append((n, keep.get((b, n), q)))
    for b, mods in by_block.items():
        layers[b].update_modules(tree_unflatten(mods))


def nbytes(mod):
    return sum(v.nbytes for _, v in tree_flatten(mod.parameters()))


def cmd_configs(a):
    from mlx_lm import load
    M.setup_memory(a.mem_budget_gb, a.cache_gb)
    model, tok = load(a.model)
    build_ref(model, tok, a)
    ref = Ref(tok, a)
    load_block = calib_inputs(model, tok, a.calib, a.seqlen, a.calib_data)
    tag = "" if a.calib_data == "wiki" else f"_{a.calib_data}"
    out = a.out or f"runs/a16_configs_c{a.calib}{tag}_K{a.flips}.jsonl"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["config"] for l in open(out) if l.strip()}
    for cs in a.configs.split(","):
        if cs in done:
            print(f"skip {cs} (done)"); continue
        cfg = parse_config(cs)
        print(f"\n== {cs}: {cfg}", flush=True)
        swaps, tq = quantize_model(model, cfg, load_block, a.flips)
        size = sum(nbytes(q) for _, q in swaps.values())
        rec = dict(config=cs, K=a.flips, calib=a.calib, calib_data=a.calib_data, quant_secs=round(tq, 1), linear_bytes=size,
                   bpw=round(8 * size / sum(o.weight.size for o, _ in swaps.values()), 3))
        rec.update(ref.score(model, cs))
        if a.ppl:
            t0 = time.time()
            rec["ppl_wiki"] = ppl(model, tok, a.ppl_len)
            rec["ppl_c4"] = ppl_c4(model, tok, a.ppl_len)
            print(f"  PPL wiki {rec['ppl_wiki']:.4f} (bf16 {ref.meta['ppl_wiki']:.4f})  "
                  f"c4 {rec['ppl_c4']:.4f} (bf16 {ref.meta['ppl_c4']:.4f})  [{time.time()-t0:.0f}s]", flush=True)
        print(f"  quant {tq:.0f}s  {size/GB:.2f} GB  {rec['bpw']} bpw", flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        restore(model, swaps)
        del swaps
        mx.clear_cache()


def cmd_greedy(a):
    """Greedy mixed precision on the W4A16 model: at each step, try keeping each candidate
    unit (block or linear) at `--keep` precision and pick the one with the best KLD drop
    per extra byte. Scored on the WikiText eval set; C4 is reported for the chosen set."""
    from mlx_lm import load
    M.setup_memory(a.mem_budget_gb, a.cache_gb)
    model, tok = load(a.model)
    build_ref(model, tok, a)
    ref = Ref(tok, a)
    cfg = parse_config(a.config)
    load_block = calib_inputs(model, tok, a.calib, a.seqlen, a.calib_data) if cfg["method"] == "macaq" else None
    swaps, _ = quantize_model(model, cfg, load_block, a.flips)
    class Hi(dict):
        # kept-precision modules, built on demand (8-bit for all linears would be +27 GB)
        def __missing__(self, k):
            o = swaps[k][0]
            if a.keep == "bf16":
                return o
            m = stock_quant(o, "affine8"); mx.eval(m.parameters())
            return m
    hi = Hi()
    bpw_hi = 16 if a.keep == "bf16" else 8.5      # affine 8-bit gs64: 8 + (16+16)/64
    extra = {k: int(o.weight.size * bpw_hi / 8) - nbytes(q) for k, (o, q) in swaps.items()}
    base_bytes = sum(nbytes(q) for _, q in swaps.values())
    units = sorted({k[0] for k in swaps}) if a.unit == "block" else sorted(swaps)
    members = (lambda u: [k for k in swaps if k[0] == u]) if a.unit == "block" else (lambda u: [u])
    out = a.out or (f"runs/a16_greedy_{a.config}_K{a.flips}_{a.keep}_{a.unit}_c{a.calib}"
                    + ("" if a.calib_data == "wiki" else f"_{a.calib_data}")
                    + ("" if a.screen_set == "wiki" else f"_screen{a.screen_set}") + ".jsonl")
    kept, hist = {}, []
    if os.path.exists(out):
        hist = [json.loads(l) for l in open(out) if l.strip()]
        for h in hist[1:]:
            for k in members(tuple(h["pick"]) if a.unit == "linear" else h["pick"]):
                kept[k] = hi[k]
    apply(model, swaps, kept)
    if not hist:
        r = ref.score(model, "start")
        hist.append(dict(step=0, pick=None, extra_bytes=0, total_bytes=base_bytes, **r))
        open(out, "w").write(json.dumps(hist[0]) + "\n")
    # pre-screen on a subset: wiki only, or the mean of wiki and C4 (avoids in-domain selection)
    n_screen = a.screen_seqs
    sets = ("wiki",) if a.screen_set == "wiki" else ("wiki", "c4")

    def screen(m):
        return sum(ref.kld_subset(m, n, n_screen) if n_screen else ref.kld(m, n)["kld"] for n in sets) / len(sets)
    for step in range(len(hist), a.steps + 1):
        t0 = time.time()
        cand = [u for u in units if not all(k in kept for k in members(u))]
        scores = {}
        for u in cand:
            trial = dict(kept); trial.update({k: hi[k] for k in members(u)})
            apply(model, swaps, trial)
            scores[u] = screen(model)
        apply(model, swaps, kept)
        b0 = screen(model)
        eff = {u: (b0 - scores[u]) / max(sum(extra[k] for k in members(u)), 1) for u in cand}
        pick = max(eff, key=eff.get)
        kept.update({k: hi[k] for k in members(pick)})   # only kept units stay materialised
        apply(model, swaps, kept)
        r = ref.score(model, f"step {step}: +{pick}")
        tot = base_bytes + sum(extra[k] for k in kept)
        rec = dict(step=step, pick=pick, extra_bytes=sum(extra[k] for k in members(pick)), total_bytes=tot,
                   secs=round(time.time() - t0, 1), screen=dict(sorted(((str(u), scores[u]) for u in cand),
                                                                        key=lambda x: x[1])[:8]), **r)
        hist.append(rec)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  step {step}: keep {pick} @{a.keep}  +{rec['extra_bytes']/1e6:.0f} MB  total {tot/GB:.2f} GB  "
              f"({time.time()-t0:.0f}s)", flush=True)


def _kld_subset(self, model, name, n):
    kl = []
    for i, s in enumerate(self.sets[name][:n]):
        lp_r = mx.load(f"{REF_DIR}/{name}_{self.key}_{i}.npy")
        lp_q = M.logprobs(model, s)
        k = mx.sum(mx.exp(lp_r) * (lp_r - lp_q), -1)
        mx.eval(k); kl.append(k)
        del lp_r, lp_q
        mx.clear_cache()
    return mx.concatenate(kl).mean().item()


Ref.kld_subset = _kld_subset


# ---------------------------------------------------------------- export / verify
EXPORT_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "chat_template.jinja", "generation_config.json")


def sha256_file(path, chunk=1 << 24):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_array(x):
    import hashlib
    import numpy as np
    return hashlib.sha256(np.array(x).tobytes()).hexdigest()


def provenance(a, cfg, tok):
    import platform
    import subprocess
    from importlib.metadata import version
    here = os.path.dirname(os.path.abspath(__file__))
    calib = calib_seqs(tok, a.calib_data, a.calib, a.seqlen)
    base = {f: sha256_file(os.path.join(a.model, f)) for f in sorted(os.listdir(a.model))
            if f.endswith(".safetensors") or f == "config.json"}
    return dict(
        method="MACaQ (arXiv 2611.10649), independent MLX port, W4A16",
        recipe=dict(config=cfg["name"], alpha=cfg.get("alpha"), rotate=cfg.get("rotate"),
                    hadamard_block=M.BLOCK, flips_K=a.flips, ridge_lambda=M.LAM, t_clamp=list(M.T_CLAMP),
                    flip_rtol=M.FLIP_RTOL, exponent_screen=[1, 0, -1], recompute_every=32,
                    e2m1_ties="half-to-even", calibration="per block on bf16-model activations (not sequential)",
                    hessian_dtype="float32", tf32=os.environ.get("MLX_ENABLE_TF32") != "0",
                    skipped_linears=list(M.SKIP_SUFFIXES)),
        calibration_data=dict(source=a.calib_data,
                              wiki=f"{M.DATASET} wikitext-2-raw-v1 train: '\\n\\n'.join(text), first n*seqlen tokens",
                              c4=f"{C4_TRAIN}: docs shuffled (seed 0), joined with '\\n\\n'",
                              mix="half wiki, half c4, rows interleaved",
                              tokens=int(calib.size), seqlen=a.seqlen, token_ids_sha256=sha256_array(calib)),
        base_model=dict(repo="mlx-community/Qwen3.8-27B-bf16", upstream="Qwen/Qwen3.8-27B",
                        local_path=a.model, sha256=base),
        code={f: sha256_file(os.path.join(here, f)) for f in ("macaq_mlx.py", "a16.py", "macaq_model.py")},
        software=dict(python=platform.python_version(), mlx=version("mlx"), mlx_lm=version("mlx-lm"),
                      datasets=version("datasets"), transformers=version("transformers"),
                      macos=platform.mac_ver()[0],
                      chip=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          capture_output=True, text=True).stdout.strip()),
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


def cmd_export(a):
    """Quantize with one config and write a loadable MLX checkpoint to --out-dir."""
    import shutil
    from mlx_lm import load
    from mlx_lm.utils import save_model, load_config
    M.setup_memory(a.mem_budget_gb, a.cache_gb)
    cfg = parse_config(a.config)
    assert cfg["method"] == "macaq", "export is for MACaQ configs"
    dst = a.out_dir or (f"runs/export/Qwen3.8-27B-MACaQ-W4A16-{a.config}-K{a.flips}-c{a.calib}"
                        + ("" if a.calib_data == "wiki" else f"-{a.calib_data}")
                        + (f"-q8b{a.q8_blocks}" if a.q8_blocks else ""))
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists; remove it first")
    model, tok = load(a.model)
    load_block = calib_inputs(model, tok, a.calib, a.seqlen, a.calib_data)
    prov = provenance(a, cfg, tok)
    swaps, tq = quantize_model(model, cfg, load_block, a.flips)
    print(f"quantized in {tq:.0f}s", flush=True)
    q8 = set(M.parse_blocks(a.q8_blocks, len(M.text_core(model).layers))) if a.q8_blocks else set()
    keep = {k: stock_quant(o, "affine8") for k, (o, q) in swaps.items() if k[0] in q8}
    apply(model, swaps, keep)
    core_prefix = "language_model.model.layers"
    modules = {}
    for (b, n), (o, q) in swaps.items():
        if (b, n) in keep:
            modules[f"{core_prefix}.{b}.{n}"] = dict(kind="affine", bits=8, group_size=64)
        else:
            assert isinstance(q, M.MACaQLinear) and q.act_exp is None
            modules[f"{core_prefix}.{b}.{n}"] = "macaq"
    prov["recipe"]["affine8_blocks"] = sorted(q8)
    print(f"blocks at affine 8-bit: {sorted(q8) or 'none'}", flush=True)
    # reference logits of the in-memory model, for the round-trip check
    probe = M.token_seqs(tok, "test", 2 * 512, 512)
    ref_logits = [model(s[None])[0].astype(mx.float32) for s in probe]
    mx.eval(ref_logits)
    del swaps                          # drop the bf16 originals before writing
    mx.clear_cache()
    os.makedirs(dst)
    save_model(dst, model, donate_model=True)
    config = load_config(Path(a.model))
    config.pop("vision_config", None)
    config.pop("quantization", None)
    config["model_file"] = "macaq_model.py"
    config["quantization_config"] = dict(quant_method="macaq_mxfp4", bits=4, group_size=M.BLOCK,
                                         activations="bf16")
    config["macaq"] = dict(format="macaq-mxfp4-w4a16", format_version=1, block=M.BLOCK,
                           has_t=cfg["alpha"] is not None, rotate=cfg["rotate"],
                           recipe=prov["recipe"], modules=modules)
    with open(f"{dst}/config.json", "w") as f:
        json.dump(dict(sorted(config.items())), f, indent=2)
    for fn in EXPORT_FILES:
        if os.path.exists(os.path.join(a.model, fn)):
            shutil.copy(os.path.join(a.model, fn), dst)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "macaq_model.py"), dst)
    prov["checkpoint"] = {f: sha256_file(os.path.join(dst, f)) for f in sorted(os.listdir(dst))
                          if f.endswith(".safetensors")}
    json.dump(prov, open(f"{dst}/macaq_provenance.json", "w"), indent=2)
    mx.save_safetensors(f"{dst}.roundtrip.safetensors",
                        {f"logits{i}": l for i, l in enumerate(ref_logits)} | {"tokens": probe})
    print(f"exported -> {dst}  ({sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst)) / GB:.2f} GB)")


def cmd_verify(a):
    """Load an exported checkpoint the way users will (mlx_lm.load, trust_remote_code) and check
    it: bitwise round-trip vs the in-memory model, KLD vs bf16, optional PPL and generation."""
    from mlx_lm import load
    M.setup_memory(a.mem_budget_gb, a.cache_gb)
    src = a.out_dir
    try:
        load(src)
        raise SystemExit("FAIL: stock mlx_lm.load without trust_remote_code accepted the checkpoint")
    except ValueError as e:
        print(f"ok: stock load refused ({str(e)[:80]}...)")
    model, tok = load(src, trust_remote_code=True)
    nq = sum(type(m).__name__ == "MACaQLinear" for _, m in model.named_modules())
    n8 = sum(isinstance(m, nn.QuantizedLinear) and m.bits == 8 for _, m in model.named_modules())
    print(f"loaded: {nq} MACaQ linears, {n8} affine 8-bit linears, active {mx.get_active_memory() / GB:.2f} GB")
    rt = mx.load(f"{src.rstrip('/')}.roundtrip.safetensors")
    for i, s in enumerate(rt["tokens"]):
        l = model(s[None])[0].astype(mx.float32)
        d = mx.abs(l - rt[f"logits{i}"]).max().item()
        print(f"round-trip seq {i}: max |logit diff| = {d}")
        assert d == 0.0, "round-trip mismatch"
    rec = dict(checkpoint=src)
    if a.kld:
        rec.update(Ref(tok, a).score(model, "export"))
    if a.ppl:
        meta = json.load(open(f"{REF_DIR}/meta.json"))
        t0 = time.time()
        rec["ppl_wiki"] = ppl(model, tok, a.ppl_len)
        rec["ppl_c4"] = ppl_c4(model, tok, a.ppl_len)
        print(f"PPL wiki {rec['ppl_wiki']:.4f} (bf16 {meta['ppl_wiki']:.4f})  c4 {rec['ppl_c4']:.4f} "
              f"(bf16 {meta['ppl_c4']:.4f})  [{time.time() - t0:.0f}s]", flush=True)
    if a.generate:
        rec["generations"] = generations(model, tok)
    json.dump(rec, open(f"{src.rstrip('/')}.verify.json", "w"), indent=1)


GEN_PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive? Answer briefly.",
    "Translate to French: 'The meeting has been moved to Thursday morning.'",
    "List the planets of the solar system in order from the Sun.",
    "What is 17 * 23? Show your work briefly.",
]


def generations(model, tok, max_tokens=256):
    """Greedy chat generations (thinking off) with mlx_lm's own prompt/decode tok/s."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    out = []
    for p in GEN_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        text, last = "", None
        for r in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
            text += r.text; last = r
        print(f"\n### {p}\n{text}\n[{last.generation_tokens} tok, decode {last.generation_tps:.1f} tok/s, "
              f"peak {last.peak_memory:.1f} GB]", flush=True)
        out.append(dict(prompt=p, output=text, tokens=last.generation_tokens,
                        decode_tps=round(last.generation_tps, 2), prompt_tps=round(last.prompt_tps, 1),
                        peak_gb=round(last.peak_memory, 2)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("ref", "configs", "greedy", "export", "verify", "bf16gen"))
    p.add_argument("--model", default="models/Qwen3.8-27B-bf16")
    p.add_argument("--calib", type=int, default=65536)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--calib-data", choices=("wiki", "c4", "mix"), default="wiki")
    p.add_argument("--flips", type=int, default=128)
    p.add_argument("--eval-seqs", type=int, default=16, help="per eval set (wiki, c4)")
    p.add_argument("--eval-len", type=int, default=512)
    p.add_argument("--ppl-len", type=int, default=2048)
    p.add_argument("--ppl", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--configs", default="a0.75")
    p.add_argument("--config", default="a0.75", help="greedy: base config")
    p.add_argument("--keep", choices=("bf16", "q8"), default="bf16", help="greedy: precision of kept units")
    p.add_argument("--unit", choices=("block", "linear"), default="block")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--screen-seqs", type=int, default=4, help="greedy: wiki seqs used to screen candidates (0=all)")
    p.add_argument("--out", default=None)
    p.add_argument("--screen-set", choices=("wiki", "both"), default="wiki", help="greedy: screening eval set(s)")
    p.add_argument("--q8-blocks", default="", help="export: blocks kept at affine 8-bit, e.g. 56-63")
    p.add_argument("--out-dir", default=None, help="export/verify: checkpoint dir")
    p.add_argument("--kld", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--generate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--mem-budget-gb", type=float, default=90)
    p.add_argument("--cache-gb", type=float, default=4)
    a = p.parse_args()
    if a.cmd == "ref":
        from mlx_lm import load
        M.setup_memory(a.mem_budget_gb, a.cache_gb)
        model, tok = load(a.model)
        print(json.dumps(build_ref(model, tok, a), indent=1))
    elif a.cmd == "configs":
        cmd_configs(a)
    elif a.cmd == "export":
        cmd_export(a)
    elif a.cmd == "verify":
        cmd_verify(a)
    elif a.cmd == "bf16gen":
        from mlx_lm import load
        M.setup_memory(a.mem_budget_gb, a.cache_gb)
        model, tok = load(a.model)
        json.dump(generations(model, tok), open("runs/bf16_generations.json", "w"), indent=1)
    else:
        cmd_greedy(a)


if __name__ == "__main__":
    main()
