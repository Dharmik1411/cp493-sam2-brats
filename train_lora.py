"""H4: train LoRA adapters on SAM 2.1 for BraTS whole tumour. Base weights stay frozen.

Each step takes `--clip` consecutive slices from inside a training patient's tumour, puts a ground-truth
box on the first slice only, and runs SAM 2's own track_step frame by frame, so the memory carries the
mask forward the same way it does at test time. Loss = 20 x focal + Dice per frame (as in SAM 2).

python train_lora.py --data data --splits splits_zia.json --part memory_attention --steps 300 --out lora_memory.pt
python train_lora.py --data data --splits splits_zia.json --part image_encoder   --steps 300 --out lora_encoder.pt
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import brats
import lora
from sam2_runner import pick_device

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_input(frame, size):
    """uint8 (H, W, 3) slice -> normalized (3, size, size) tensor, resized like SAM 2's frame loader."""
    img = np.asarray(Image.fromarray(frame).resize((size, size)), dtype=np.float32) / 255.0
    return (torch.from_numpy(img).permute(2, 0, 1) - MEAN) / STD


def sample_clip(case_dirs, length, norm, rng):
    """Random run of `length` slices inside one patient's whole-tumour extent, first slice non-empty."""
    while True:
        case = brats.load_case(rng.choice(case_dirs))  # shortcut: reloads the case every step, cache if I/O-bound
        gt = np.isin(case["seg"], brats.REGIONS["WT"])
        z = brats.tumour_slices(gt)
        if len(z) < length:
            continue
        s = rng.randrange(int(z[0]), int(z[-1]) - length + 2)
        if gt[s].any():
            return brats.make_frames(case, "flair3", norm)[s : s + length], gt[s : s + length]


def sigmoid_focal(logits, target, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    return ((alpha * target + (1 - alpha) * (1 - target)) * ce * (1 - pt) ** gamma).mean()


def soft_dice(logits, target, eps=1.0):
    p = logits.sigmoid()
    return 1 - (2 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps)


def _detach(x):
    return x.detach() if torch.is_tensor(x) else [_detach(v) for v in x] if isinstance(x, (list, tuple)) else x


def train_clip(model, frames, gt, device, train_encoder):
    """Forward + backward one slice at a time; gradients accumulate, returns the mean loss."""
    # shortcut: memory from earlier slices is detached (truncated backprop) so peak memory is one slice;
    # the LoRA still learns how to read the memory, not how its own earlier outputs shaped it
    size, (H, W) = model.image_size, gt.shape[1:]
    memory = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    total = 0.0
    for t in range(len(frames)):
        with torch.set_grad_enabled(train_encoder):  # frozen encoder: no need to keep its activations
            backbone = model.forward_image(to_input(frames[t], size).unsqueeze(0).to(device))
        _, feats, pos, sizes = model._prepare_backbone_features(backbone)
        points = None
        if t == 0:  # box as two corner points, scaled to SAM 2's 1024 input, like add_new_points_or_box
            box = torch.tensor(brats.box_from_mask(gt[0]), dtype=torch.float32).view(1, 2, 2)
            points = {"point_coords": (box / torch.tensor([W, H]) * size).to(device),
                      "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=device)}
        out = model.track_step(frame_idx=t, is_init_cond_frame=t == 0, current_vision_feats=feats,
                               current_vision_pos_embeds=pos, feat_sizes=sizes, point_inputs=points,
                               mask_inputs=None, output_dict=memory, num_frames=len(frames))
        logits = F.interpolate(out["pred_masks_high_res"], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        target = torch.from_numpy(gt[t]).float().to(device)
        loss = (20 * sigmoid_focal(logits, target) + soft_dice(logits, target)) / len(frames)
        if loss.requires_grad:  # the first slice never reads memory, so memory-attention LoRA gets no gradient there
            loss.backward()
        total += loss.item()
        memory["cond_frame_outputs" if t == 0 else "non_cond_frame_outputs"][t] = {k: _detach(v) for k, v in out.items()}
    return total


def main():
    from sam2.build_sam import build_sam2_hf

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="splits_zia.json", help="trains only on its 'train' patients")
    ap.add_argument("--part", required=True, choices=list(lora.TARGETS))
    ap.add_argument("--rank", type=int, default=0, help="0 = rank 8 for memory, size-matched rank for encoder")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--clip", type=int, default=8, help="slices per training clip (SAM 2 used 8 frames)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--norm", default="slice", choices=["volume", "slice"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="where to save the LoRA weights (.pt)")
    args = ap.parse_args()

    splits = json.load(open(args.splits))
    test = set(splits.get("test", [])) | set(splits.get("val", []))  # never train on test or validation patients
    complete = lambda d: os.path.isdir(d) and sum(f.endswith(".nii.gz") for f in os.listdir(d)) == 5
    case_dirs = [os.path.join(args.data, c) for c in splits["train"] if c not in test and complete(os.path.join(args.data, c))]
    assert case_dirs, "no unpacked training patients in --data"
    device = pick_device()
    model = build_sam2_hf(args.model, device=device)
    rank = args.rank or (8 if args.part == "memory_attention" else
                         lora.matched_rank(model, "image_encoder", 8 * lora.per_rank(model, "memory_attention")))
    n_train = lora.add_lora(model, args.part, rank)
    print(f"{len(case_dirs)} training patients, LoRA on {args.part} rank {rank}: {n_train:,} trainable, on {device}")

    # recompute each memory-attention layer in backprop instead of storing it: attention over 7 memories
    # of 4096 tokens is several GB per slice otherwise
    from torch.utils.checkpoint import checkpoint

    layers = list(model.memory_attention.layers)
    if args.part == "image_encoder":  # same for the Hiera blocks: without it encoder training swaps on the M2 (~3 h/step)
        layers += list(model.image_encoder.trunk.blocks)
    for layer in layers:
        layer.forward = lambda *a, _f=layer.forward, **k: checkpoint(_f, *a, use_reentrant=False, **k)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    log = open(os.path.splitext(args.out)[0] + "_log.csv", "w")
    log.write("step,loss,seconds\n")
    start = time.time()
    for step in range(1, args.steps + 1):
        frames, gt = sample_clip(case_dirs, args.clip, args.norm, rng)
        opt.zero_grad()
        loss = train_clip(model, frames, gt, device, args.part == "image_encoder")
        opt.step()
        if device == "mps":  # give cached GPU memory back each step so it can't creep into swap
            torch.mps.empty_cache()
        log.write(f"{step},{loss:.4f},{time.time() - start:.0f}\n")
        log.flush()
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{args.steps} loss {loss:.4f} ({time.time() - start:.0f}s)", flush=True)

    weights = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"part": args.part, "rank": rank, "model": args.model, "trainable": n_train,
                "patients": [os.path.basename(c) for c in case_dirs], "args": vars(args), "weights": weights}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
