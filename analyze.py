"""Paired comparisons between results files, as reported: Wilcoxon signed-rank tests, Holm correction over every
test printed, and 95% bootstrap confidence intervals of the mean paired difference. Only patients present in both
files are compared.

python analyze.py results/zia_rp30_150.csv results/lora_encoder_v2_rp30_150.csv      each later file vs the first
python analyze.py --all results/zia_rp30_150.csv results/lora_encoder_rp30_150.csv results/lora_encoder_v2_rp30_150.csv results/zia_ceiling_150.csv
"""
import argparse
import csv
import os

import numpy as np
from scipy.stats import wilcoxon

METRICS = {"dice": 1, "nsd": 1, "hd95": -1}  # 1: higher is better, -1: lower is better


def holm(p):
    """Holm-adjusted p-values, returned in the input order."""
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, p[i] * (len(p) - rank)))
        adj[i] = run
    return adj


assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def load(path, region):
    return {r["case"]: r for r in csv.DictReader(open(path)) if r["region"] == region}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="results CSVs from run.py eval")
    ap.add_argument("--all", action="store_true", help="compare every pair of files, not each file with the first")
    ap.add_argument("--region", default="WT")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    a = ap.parse_args()
    data = {f: load(f, a.region) for f in a.files}
    pairs = ([(x, y) for i, x in enumerate(a.files) for y in a.files[i + 1:]] if a.all
             else [(a.files[0], y) for y in a.files[1:]])
    name = lambda f: os.path.splitext(os.path.basename(f))[0]
    rng = np.random.default_rng(0)
    rows = []
    for x, y in pairs:
        cases = sorted(set(data[x]) & set(data[y]))
        for m, sign in METRICS.items():
            u = np.array([float(data[x][c][m]) for c in cases])
            v = np.array([float(data[y][c][m]) for c in cases])
            ok = np.isfinite(u) & np.isfinite(v)
            u, v = u[ok], v[ok]
            d = v - u
            boot = d[rng.integers(0, len(d), (a.boot, len(d)))].mean(axis=1)
            p = wilcoxon(v, u).pvalue if d.any() else 1.0
            rows.append((name(y), name(x), m, len(d), u.mean(), v.mean(), d.mean(), *np.percentile(boot, [2.5, 97.5]),
                         int((sign * d > 0).sum()), p))
    print(f"{'B':28s} {'A':28s} {'metric':6s} {'n':>4s} {'mean A':>7s} {'mean B':>7s} {'B-A':>7s} {'95% CI':>17s} "
          f"{'B better':>9s} {'p':>8s} {'p Holm':>8s}")
    for r, q in zip(rows, holm([r[-1] for r in rows])):
        b, a_, m, n, ma, mb, dm, lo, hi, w, p = r
        print(f"{b:28s} {a_:28s} {m:6s} {n:4d} {ma:7.3f} {mb:7.3f} {dm:+7.3f} [{lo:+7.3f},{hi:+7.3f}] {w:4d}/{n:<4d} {p:8.2g} {q:8.2g}")


if __name__ == "__main__":
    main()
