"""
python run.py smoke                  check SAM 2 runs (no data needed)
python run.py split --data DATA      fixed dev/test split
python run.py split --data members.txt --sample 150 --seed 42 --splits splits_base.json   Dr. Zia's 150 patients
python run.py eval  --data DATA ...  run one setting, write results to CSV
python run.py summary results/*.csv  mean ± std per file and region (like Table 2 in Dr. Zia's paper)
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import zlib

import numpy as np

import brats


def fake_case(D=40, H=128, W=128):
    """A synthetic 'tumour' (nested ellipsoids) for checking the pipeline."""
    z, y, x = np.ogrid[:D, :H, :W]
    r = ((z - 20) / 10) ** 2 + ((y - H / 2) / (H / 4)) ** 2 + ((x - W / 2) / (W / 4)) ** 2
    seg = np.zeros((D, H, W), np.uint8)
    seg[r < 1] = 2  # oedema
    seg[r < 0.5] = 4  # enhancing
    seg[r < 0.25] = 1  # necrotic core
    case = {s: 1 + seg * (i + 1) + 0.1 * np.random.default_rng(i).standard_normal(seg.shape) for i, s in enumerate(brats.SEQUENCES)}
    case["seg"] = seg
    return case


def smoke(args):
    case = fake_case()
    gt = np.isin(case["seg"], brats.REGIONS["WT"])
    # quick checks on my own functions first
    assert brats.dice(gt, gt) == 1.0 and brats.hd95(gt, gt) == 0.0
    assert 0 < brats.hd95(np.roll(gt, 2, axis=2), gt) <= 2.0  # shifted 2 voxels -> at most 2 mm
    assert brats.over_propagation(gt, gt) == 0.0
    spill = gt.copy(); z = brats.tumour_slices(gt); spill[z[-1] + 1] = spill[z[-1] + 6] = gt[z[-1]]
    assert np.isnan(brats.fp_near_ends(gt, gt)) and brats.fp_near_ends(spill, gt) == 0.5  # 1 and 6 slices past the end
    gappy = gt.copy(); gappy[22] = False  # a tumour with a gap must never get a box on an empty slice
    assert all(gappy[z].any() for z in brats.key_slices(gappy, "first-every", 2))
    assert brats.nsd(gt, gt) == 1.0 and brats.nsd(np.roll(gt, 3, axis=2), gt) < 1.0
    always = lambda z: True
    assert brats.reprompt_plan(always, 10, 0, 20, 5) == [(10, 14, False), (15, 19, False), (20, 20, False),
                                                       (10, 6, True), (5, 1, True), (0, 0, True)]
    assert brats.reprompt_plan(always, 10, 0, 20, 0) == [(10, 20, False), (10, 0, True)]
    assert brats.reprompt_plan(lambda z: z != 15, 10, 0, 20, 5, both=False) == [(10, 14, False), (16, 20, False)]
    print("metric checks passed")

    from sam2_runner import Sam2Runner

    runner = Sam2Runner(args.model)
    frames = brats.make_frames(case, "flair3")
    boxes = {z: brats.box_from_mask(gt[z]) for z in brats.tumour_slices(gt)}
    key = brats.key_slices(gt, "largest")[0]
    per_slice = runner.per_slice(frames, boxes)
    propagated = runner.propagate(runner.load_volume(frames), {key: boxes[key]})
    print(f"per-slice Dice {brats.dice(per_slice, gt):.3f}, propagated from slice {key} Dice {brats.dice(propagated, gt):.3f}")


def split(args):
    if os.path.isfile(args.data):  # a listing like `tar -t` output, so the split can be made before unpacking
        cases = sorted(set(re.findall(r"BraTS2021_\d{5}", open(args.data).read())))
    else:
        cases = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.data, "BraTS2021_*")) if os.path.isdir(p))
    if args.sample:  # Dr. Zia: n patients drawn at random with a fixed seed; the rest can train LoRA
        # shortcut: assumes random.Random(seed).sample on the sorted IDs; switch if his code used numpy
        test = random.Random(args.seed).sample(cases, args.sample)
        out = {"test": test, "train": [c for c in cases if c not in set(test)]}
    else:
        order = [cases[i] for i in np.random.default_rng(0).permutation(len(cases))]
        out = {"dev": order[: args.dev], "test": order[args.dev : args.dev + args.test]}
    json.dump(out, open(args.splits, "w"), indent=1)
    print(f"{len(cases)} cases -> " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f" ({args.splits})")


def evaluate(args):
    from sam2_runner import Sam2Runner

    cases = json.load(open(args.splits))[args.subset][: args.limit or None]
    runner = Sam2Runner(args.model, args.lora)
    setting = dict(lora=os.path.basename(args.lora or "none"), mode=args.mode, placement=args.placement, step=args.step, direction=args.direction,
                   channels=args.channels, norm=args.norm, noise=args.noise)
    os.makedirs("results", exist_ok=True)
    path = f"results/{args.name}.csv"
    fields = ["case", "region", *setting, "n_boxes", "dice", "hd95", "nsd", "over_prop", "fp_near"]
    old = list(csv.DictReader(open(path))) if args.resume and os.path.exists(path) else []
    for r in old:  # resume only the same setting, so rows from different runs never mix in one file
        assert all(r[k] == str(v) for k, v in setting.items()), f"{path} holds a different setting, use another --name"
    if old:
        fields = list(old[0])  # keep the file's own columns (older files have no fp_near)
    done = {r["case"] for r in old}
    with open(path, "a" if old else "w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        if not old:
            writer.writeheader()
        for name in cases:
            if name in done:
                continue
            case = brats.load_case(os.path.join(args.data, name))
            frames = brats.make_frames(case, args.channels, args.norm)
            state = runner.load_volume(frames) if args.mode != "per-slice" else None
            rng = np.random.default_rng(zlib.crc32(name.encode()))  # same noise every run
            for region in args.regions.split(","):
                labels = brats.REGIONS[region]
                gt = np.isin(case["seg"], labels)
                if not gt.any():
                    continue
                z = brats.tumour_slices(gt)
                if args.mode == "per-slice":
                    keys = z.tolist()
                elif args.mode == "reprompt":
                    seed = brats.key_slices(gt, args.placement)[0]
                    plan = brats.reprompt_plan(lambda s: gt[s].any(), seed, z[0], z[-1], args.step, args.direction == "both")
                    keys = sorted({start for start, _, _ in plan})
                else:
                    keys = brats.key_slices(gt, args.placement, args.step)
                boxes = {z: brats.noisy_box(brats.box_from_mask(gt[z]), args.noise, rng, gt.shape[1:]) for z in keys}
                if args.mode == "per-slice":
                    pred = runner.per_slice(frames, boxes)
                elif args.mode == "reprompt":
                    pred = runner.reprompt(state, boxes, plan)
                else:
                    pred = runner.propagate(state, boxes, args.direction)
                writer.writerow({"case": name, "region": region, **setting, "n_boxes": len(keys),
                                 "dice": brats.dice(pred, gt), "hd95": brats.hd95(pred, gt), "nsd": brats.nsd(pred, gt),
                                 "over_prop": brats.over_propagation(pred, gt), "fp_near": brats.fp_near_ends(pred, gt)})
            f.flush()
            print("done", name)


def summary(args):
    for path in args.files:
        rows = list(csv.DictReader(open(path)))
        for region in sorted({r["region"] for r in rows}):
            x = [r for r in rows if r["region"] == region]
            cols = [k for k in ["dice", "hd95", "nsd", "n_boxes"] if k in x[0]]
            v = {k: np.array([float(r[k]) for r in x]) for k in cols}
            sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0
            print(f"{os.path.basename(path)} {region} n={len(x)}  " + "  ".join(f"{k} {v[k].mean():.3f} ± {sd(v[k]):.3f}" for k in cols))


p = argparse.ArgumentParser()
p.add_argument("command", choices=["smoke", "split", "eval", "summary"])
p.add_argument("files", nargs="*", help="summary: result CSVs")
p.add_argument("--data", help="folder with the BraTS2021_xxxxx case folders")
p.add_argument("--model", default="facebook/sam2.1-hiera-small")
p.add_argument("--lora", help="eval: LoRA weights from train_lora.py")
p.add_argument("--dev", type=int, default=50)
p.add_argument("--test", type=int, default=200)
p.add_argument("--subset", default="dev", choices=["dev", "test", "val", "train"])
p.add_argument("--splits", default="splits.json", help="split file to write (split) or read (eval)")
p.add_argument("--sample", type=int, default=0, help="split: draw this many patients at random (Dr. Zia: 150)")
p.add_argument("--seed", type=int, default=42, help="split: seed for --sample")
p.add_argument("--limit", type=int, default=0, help="only the first N cases")
p.add_argument("--mode", default="video", choices=["per-slice", "video", "reprompt"])
p.add_argument("--placement", default="largest", choices=["first-every", "largest", "middle"])
p.add_argument("--step", type=int, default=10, help="first-every spacing; reprompt interval (0 = one box)")
p.add_argument("--direction", default="both", choices=["forward", "both"])
p.add_argument("--channels", default="flair3", choices=list(brats.CHANNELS))
p.add_argument("--norm", default="volume", choices=["volume", "slice"], help="slice = Dr. Zia's 1st-99th per slice")
p.add_argument("--noise", default="none", choices=list(brats.NOISE))
p.add_argument("--regions", default="WT,TC,ET", help="comma-separated, e.g. WT")
p.add_argument("--name", default="run", help="results/<name>.csv")
p.add_argument("--resume", action="store_true", help="eval: keep results/<name>.csv and skip the cases already in it")

if __name__ == "__main__":
    args = p.parse_args()
    {"smoke": smoke, "split": split, "eval": evaluate, "summary": summary}[args.command](args)
