"""H3 tables and tests, as Dr. Zia asked on Nov 14 (whole tumour, re-prompting every 30 slices vs a box on every slice).

python analyze_h3.py            uses the 150-patient files (falls back to the 30-patient ones that exist)
python analyze_h3.py --n 30     the first 30 patients only

1. Dice (and NSD) for tight, medium and loose boxes in each mode, on the patients every file shares.
2. Does a loose box cost more with propagation than per slice? Per patient, cost = tight - loose in each mode;
   paired Wilcoxon on (cost with re-prompting - cost per slice), Holm-corrected, with a bootstrap 95% CI.
3. How much each adapter loses with loose boxes compared with zero-shot re-prompting (same test on the costs).
4. Where over-propagation lands, from the one-box runs: share of spilled voxels by distance past the tumour's ends.
"""
import argparse
import csv
import os

import numpy as np
from scipy.stats import wilcoxon

from analyze import holm

MODES = {  # mode -> level -> results file stem ({n} = 150 or 30)
    "Zero-shot, box on every slice": {"tight": "ceiling_{n}", "medium": "h3_ceiling_medium_{n}", "loose": "h3_ceiling_low_{n}"},
    "Zero-shot, re-prompt every 30": {"tight": "rp30_{n}", "medium": "h3_rp30_medium_{n}", "loose": "h3_rp30_low_{n}"},
    "LoRA encoder v2, re-prompt every 30": {"tight": "lora_encoder_v2_rp30_150", "loose": "h3_lora_v2_rp30_low_{n}"},
    "LoRA memory v2, re-prompt every 30": {"tight": "lora_memory_v2_rp30_150", "loose": "h3_lora_mem_v2_rp30_low_{n}"},
}
SPILL_BINS = ["spill_1_3", "spill_4_10", "spill_11_20", "spill_21p"]


def load(results, stem, n):
    """WT rows of results/<stem>.csv, falling back from the 150-patient file to the 30-patient one."""
    for s in dict.fromkeys([stem.format(n=n), stem.format(n=30)]):
        path = os.path.join(results, s + ".csv")
        if os.path.exists(path):
            return s, {r["case"]: r for r in csv.DictReader(open(path)) if r["region"] == "WT"}
    return None, None


def paired(d, rng, boot=10000):
    d = d[np.isfinite(d)]
    b = d[rng.integers(0, len(d), (boot, len(d)))].mean(axis=1)
    return d.mean(), *np.percentile(b, [2.5, 97.5]), (wilcoxon(d).pvalue if d.any() else 1.0), len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--n", type=int, default=150, help="150 or 30 test patients")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    data, used = {}, []
    for mode, levels in MODES.items():
        for level, stem in levels.items():
            name, rows = load(a.results, stem, a.n)
            if rows:
                data[mode, level] = rows
                used.append(name)
    cases = sorted(set.intersection(*(set(r) for r in data.values())))
    get =lambda key, m: np.array([float(data[key][c][m]) for c in cases])
    print(f"files: {', '.join(used)}\npatients in every file: {len(cases)}\n")

    print("1. Mean Dice (NSD), whole tumour")
    print(f"{'Mode':38s} {'tight':>14s} {'medium':>14s} {'loose':>14s}")
    for mode in MODES:
        cells = [f"{get((mode, l), 'dice').mean():.3f} ({get((mode, l), 'nsd').mean():.3f})" if (mode, l) in data else "-"
                 for l in ["tight", "medium", "loose"]]
        if any(c != "-" for c in cells):
            print(f"{mode:38s} " + " ".join(f"{c:>14s}" for c in cells))

    tests = []
    ps, rp = "Zero-shot, box on every slice", "Zero-shot, re-prompt every 30"
    cost = lambda mode, level, m: get((mode, "tight"), m) - get((mode, level), m)
    for m in ["dice", "nsd"]:
        for level in ["loose", "medium"]:
            if all((k, l) in data for k in (ps, rp) for l in ("tight", level)):
                tests.append((f"2. {level} box costs more with re-prompting than per slice ({m})",
                              cost(rp, level, m) - cost(ps, level, m), cost(rp, level, m).mean(), cost(ps, level, m).mean()))
        for ad in [k for k in MODES if k.startswith("LoRA")]:
            if all((k, l) in data for k in (ad, rp) for l in ("tight", "loose")):
                tests.append((f"3. {ad.split(',')[0]} loses more than zero-shot with loose boxes ({m})",
                              cost(ad, "loose", m) - cost(rp, "loose", m), cost(ad, "loose", m).mean(), cost(rp, "loose", m).mean()))
    if tests:
        res = [paired(d, rng) for _, d, _, _ in tests]
        print("\nPaired tests on per-patient costs (cost = tight - loose); a positive difference means the first loses more")
        for (label, _, c1, c2), (dm, lo, hi, p, n), q in zip(tests, res, holm([r[3] for r in res])):
            print(f"{label}\n   costs {c1:.3f} vs {c2:.3f}   difference {dm:+.3f} [{lo:+.3f}, {hi:+.3f}]   n={n}   p={p:.2g}   p Holm={q:.2g}")

    print("\n4. Over-propagation, one box carried through the whole scan (first 30 patients)")
    for level in ["none", "low"]:
        name, rows = load(a.results, "h3_onebox_spill_" + level + "_{n}", a.n)
        name, rows = (name, rows) if rows else load(a.results, "h3_onebox_" + level + "_{n}", a.n)
        if not rows:
            continue
        r = list(rows.values())
        op = np.array([float(x["over_prop"]) for x in r])
        line = f"   {level:5s} boxes, {name}: spill in {(op > 0).sum()} of {len(op)}; tumour-free slices within 10 of the ends with a mask {np.nanmean(op):.0%}"
        if SPILL_BINS[0] in r[0]:
            sh = np.array([[float(x[b]) for b in SPILL_BINS] for x in r])
            far = np.array([float(x["spill_max"]) for x in r])
            line += ("\n      share of spilled voxels by slices past the end (mean over patients with spill): "
                     + ", ".join(f"{b[6:].replace('_', '-').replace('21p', '21+')}: {v:.0%}" for b, v in zip(SPILL_BINS, np.nanmean(sh, axis=0)))
                     + f"\n      furthest spill: median {np.median(far[far > 0]):.0f} slices past the end")
        elif "fp_near" in r[0]:
            fn = np.array([float(x["fp_near"]) for x in r])
            line += f"\n      share of spilled voxels within 3 slices of the ends: mean {np.nanmean(fn):.0%}, median {np.nanmedian(fn):.0%}"
        print(line)


if __name__ == "__main__":
    main()
