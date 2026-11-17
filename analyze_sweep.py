"""Summarise a per-block sweep JSONL: ranking, by-kind stats, timing."""
import json, sys, statistics as st
path = sys.argv[1] if len(sys.argv) > 1 else "runs/sweep_macaq_K128_c65536.jsonl"
R = sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda r: r["block"])
modes = [k for k in ("w4a4_mlx", "w4a4_ocp", "w4a16") if k in R[0]]
print(f"{len(R)} blocks from {path}\n")
print("blk kind   " + "  ".join(f"{m:>9s}" for m in modes) + "   top1(mlx)  quant_s  worst-linear(rel_flip)")
for r in R:
    wl = max(r["linears"], key=lambda x: x.get("rel_flip", 0))
    wls = f"{wl['name']} {wl['rel_flip']:.2e}" if "rel_flip" in wl else "-"
    print(f"{r['block']:3d} {r['kind']:6s} " + "  ".join(f"{r[m]['kld']:9.3e}" for m in modes)
          + f"   {r[modes[0]]['top1']:.4f}    {r['quant_secs']:5.1f}   {wls}")
print()
for kind in ("full", "linear"):
    sub = [r for r in R if r["kind"] == kind]
    if not sub:
        continue
    print(f"{kind:6s} n={len(sub):2d}  " + "  ".join(
        f"{m}: mean {st.mean(r[m]['kld'] for r in sub):.3e} median {st.median(r[m]['kld'] for r in sub):.3e}"
        for m in modes))
print()
for m in modes:
    tot = sum(r[m]["kld"] for r in R)
    print(f"sum of single-block KLDs [{m}]: {tot:.3e}  (NOT the full-model KLD; ranking only)")
if "w4a4_ocp" in modes:
    w = sum(r["w4a4_mlx"]["kld"] < r["w4a4_ocp"]["kld"] for r in R)
    print(f"\nmlx act-exp beats ocp on {w}/{len(R)} blocks; mean ratio ocp/mlx "
          f"{st.mean(r['w4a4_ocp']['kld']/r['w4a4_mlx']['kld'] for r in R):.2f}")
if "w4a16" in modes:
    print(f"act-quant share of W4A4(mlx) KLD: median "
          f"{st.median(1 - r['w4a16']['kld']/r['w4a4_mlx']['kld'] for r in R):.0%}")
top = sorted(R, key=lambda r: -r[modes[0]]["kld"])
print(f"\nmost sensitive [{modes[0]}]: " + ", ".join(f"{r['block']}({r['kind'][0]}) {r[modes[0]]['kld']:.2e}" for r in top[:10]))
print(f"least sensitive: " + ", ".join(f"{r['block']}({r['kind'][0]}) {r[modes[0]]['kld']:.2e}" for r in top[-5:]))
if "iters" in R[0]["linears"][0]:
    c = sum(1 for r in R for l in r["linears"] if l["iters"] < r["K"])
    n = sum(len(r["linears"]) for r in R)
    print(f"\nlinears converged before K: {c}/{n}")
else:
    print("\n(per-linear stats not recorded in this run)")
print(f"quant time: total {sum(r['quant_secs'] for r in R)/60:.1f} min, mean {st.mean(r['quant_secs'] for r in R):.1f}s/block")
cal = {}
for r in R: cal[r["calib_secs_group"]] = r["block"]
print(f"calibration: {len(cal)} passes, total {sum(cal)/60:.1f} min: " + ", ".join(f"{v}:{k:.0f}s" for k, v in sorted(cal.items(), key=lambda x: x[1])))
