#!/usr/bin/env python3
"""Analyse E5 results. Unit of inference = training seed (datasets are shared across seeds).
usage: python analyze_e5.py E5_RESULTS"""
import sys, json, glob
import numpy as np, pandas as pd
from scipy import stats

folder = sys.argv[1] if len(sys.argv) > 1 else "E5_RESULTS"
rows = []
for f in sorted(glob.glob(f"{folder}/res_*.json")):
    rows += json.load(open(f))["rows"]
df = pd.DataFrame(rows)
df["diff"] = df.W_calib - df.W_fine_half          # >0: calibration is worse (primary endpoint)
df["improved"] = df["diff"] < 0
rng = np.random.default_rng(0)

def two_level_boot(sub, B=5000):
    seeds = sub.seed.unique(); g = {s: sub[sub.seed == s]["diff"].to_numpy() for s in seeds}
    out = []
    for _ in range(B):
        ss = rng.choice(seeds, len(seeds))
        out.append(np.mean([rng.choice(g[s], len(g[s])).mean() for s in ss]))
    return np.percentile(out, [2.5, 97.5])

units = df.groupby(["pair", "seed"]).agg(
    W_fine_half=("W_fine_half", "mean"), W_calib=("W_calib", "mean"), diff=("diff", "mean"),
    improved=("improved", "mean"), E_SBI=("E_SBI_fine", "mean"), E_disc=("O_fine", "mean"),
    O_rich=("O_rich", "mean"), ESS=("ESS", "mean"), cov_fine=("cov_fine", "mean"), cov_calib=("cov_calib", "mean")).reset_index()
units["ratio"] = units.E_SBI / units.E_disc
units.to_csv(f"{folder}/summary_by_seed.csv", index=False)

print("=" * 70)
for pair, u in units.groupby("pair"):
    n = len(u); d = u["diff"].to_numpy()
    t = stats.ttest_1samp(d, 0.0); 
    try: wp = stats.wilcoxon(d).pvalue
    except ValueError: wp = float("nan")
    lo, hi = two_level_boot(df[df.pair == pair])
    print(f"pair {pair}: seeds={n}  W_fine(half)={u.W_fine_half.mean():.4f}  W_calib={u.W_calib.mean():.4f}  "
          f"oracle fine/rich={u.E_disc.mean():.4f}/{u.O_rich.mean():.4f}")
    print(f"   paired diff (calib - fine_half) = {d.mean():+.4f}  [two-level bootstrap 95%: {lo:+.4f}, {hi:+.4f}]")
    print(f"   seed-level t-test p={t.pvalue:.4f} (df={n-1}); Wilcoxon p={wp:.4f}; seeds with diff<0: {(d<0).sum()}/{n}")
    print(f"   engine error E_SBI={u.E_SBI.mean():.4f} (= {u.ratio.mean():.1f} x E_disc); coverage {u.cov_fine.mean():.3f} -> {u.cov_calib.mean():.3f}; ESS/N={u.ESS.mean():.2f}")
print("=" * 70)
if len(units) >= 5:
    rho = stats.spearmanr(units.ratio, units["diff"])
    print(f"decision rule: Spearman(E_SBI/E_disc, harm) over {len(units)} (pair,seed) units: rho={rho.statistic:+.2f}, p={rho.pvalue:.4f}")
    lo_ = units[units.ratio < 1]; hi_ = units[units.ratio >= 1]
    print(f"   units with E_SBI<E_disc: n={len(lo_)}, mean diff={lo_['diff'].mean() if len(lo_) else float('nan'):+.4f};  "
          f"E_SBI>=E_disc: n={len(hi_)}, mean diff={hi_['diff'].mean() if len(hi_) else float('nan'):+.4f}")