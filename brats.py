"""BraTS 2021 glioma: load a case, make box prompts, and score predictions."""
import glob
import os

import nibabel as nib
import numpy as np

SEQUENCES = ["t1", "t1ce", "t2", "flair"]
REGIONS = {"WT": [1, 2, 4], "TC": [1, 4], "ET": [4]}  # labels: 1 necrotic core, 2 oedema, 4 enhancing
CHANNELS = {  # what goes into SAM's 3 input channels
    "flair3": ["flair", "flair", "flair"],  # same as Ludwig et al.
    "t1ce3": ["t1ce", "t1ce", "t1ce"],
    "flair_t1ce_t2": ["flair", "t1ce", "t2"],
}
NOISE = {"none": (0, 0), "medium": (0.10, 0.10), "low": (0.30, 0.20)}  # (box growth, max shift), Ludwig et al.


def find_file(case_dir, key):
    hits = glob.glob(os.path.join(case_dir, f"*_{key}.nii*"))  # e.g. BraTS2021_00000_flair.nii.gz
    if not hits:
        raise FileNotFoundError(f"no {key} file in {case_dir}")
    return hits[0]


def load_case(case_dir):
    """Returns the 4 sequences and 'seg' as (slices, height, width) arrays."""
    case = {}
    for key in SEQUENCES + ["seg"]:
        vol = np.asarray(nib.load(find_file(case_dir, key)).dataobj)
        case[key] = vol.transpose(2, 0, 1)
    case["seg"] = case["seg"].astype(np.uint8)
    return case


def to_uint8(vol):
    # clip to the 0.5-99.5th percentile of brain voxels, scale to 0-255
    brain = vol[vol > 0]
    if brain.size == 0:
        return np.zeros(vol.shape, np.uint8)
    lo, hi = np.percentile(brain, [0.5, 99.5])
    return (np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def slice_norm(vol):
    """Dr. Zia's preprocessing: each axial slice rescaled from its own 1st-99th percentile to 0-255."""
    lo, hi = np.percentile(vol, [1, 99], axis=(1, 2), keepdims=True)
    return (np.clip((vol - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def make_frames(case, channels, norm="volume"):
    """(slices, H, W, 3) uint8 images for SAM 2. norm='slice' matches Dr. Zia's paper."""
    f = slice_norm if norm == "slice" else to_uint8
    done = {s: f(case[s]) for s in set(CHANNELS[channels])}
    return np.stack([done[s] for s in CHANNELS[channels]], axis=-1)


def reprompt_plan(has_box, seed, lo, hi, interval, both=True):
    """Dr. Zia's protocol: from `seed`, propagate toward each end of [lo, hi] (the tumour's z-extent).
    A fresh box and a memory reset every `interval` slices (0 = never). A segment always starts on a
    slice with tumour; empty slices are skipped in the propagation direction.
    Returns [(start, last, reverse)] with `last` inclusive."""
    plan = []
    for reverse in ([False, True] if both else [False]):
        step, end = (-1, lo) if reverse else (1, hi)
        z = seed
        while (z - end) * step <= 0:
            if not has_box(z):
                z += step
                continue
            last = end if interval == 0 else z + step * (interval - 1)
            if (last - end) * step > 0:
                last = end
            plan.append((z, last, reverse))
            z = last + step
    return plan


def tumour_slices(mask):
    return np.flatnonzero(mask.any(axis=(1, 2)))


def key_slices(mask, placement, step=10):
    """Which slices get a box prompt."""
    z = tumour_slices(mask)
    if placement == "first-every":  # Ludwig et al.: first tumour slice, then every `step`-th tumour slice
        return z[::step].tolist()
    if placement == "largest":
        return [int(mask.sum(axis=(1, 2)).argmax())]
    if placement == "middle":
        return [int(z[len(z) // 2])]
    raise ValueError(placement)


def box_from_mask(mask2d):
    ys, xs = np.nonzero(mask2d)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)  # x0, y0, x1, y1


def noisy_box(box, level, rng, shape):
    # grow each side and shift the centre (my reading of Ludwig et al.'s description)
    growth, shift = NOISE[level]
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2 + rng.uniform(-shift, shift) * w
    cy = (y0 + y1) / 2 + rng.uniform(-shift, shift) * h
    w, h = w * (1 + growth), h * (1 + growth)
    H, W = shape
    return np.clip([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0, [W - 1, H - 1, W - 1, H - 1])


def dice(pred, gt):
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else float(2 * (pred & gt).sum() / total)


def _surface_distances(pred, gt):
    """Distances (mm, 1 mm voxels) from each surface voxel of pred to gt's surface, and back."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    sp, sg = pred & ~binary_erosion(pred), gt & ~binary_erosion(gt)
    return distance_transform_edt(~sg)[sp], distance_transform_edt(~sp)[sg]


def hd95(pred, gt):
    """95th percentile surface distance in mm (1 mm voxels). Empty prediction -> volume diagonal."""
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(np.linalg.norm(gt.shape))
    d1, d2 = _surface_distances(pred, gt)
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def nsd(pred, gt, tol=1.0):
    """Normalized surface Dice: share of both surfaces lying within `tol` mm of the other."""
    # shortcut: voxel-surface version; a surface-element library (e.g. MONAI) can differ slightly
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    d1, d2 = _surface_distances(pred, gt)
    return float(((d1 <= tol).sum() + (d2 <= tol).sum()) / (len(d1) + len(d2)))


def fp_near_ends(pred, gt, k=3):
    """H3: of the voxels predicted on slices past the tumour's ends, the share within k slices of an end (nan if none)."""
    z, s = tumour_slices(gt), np.arange(gt.shape[0])
    past = np.maximum(z[0] - s, s - z[-1])  # slices past the nearer end; <= 0 inside the tumour's extent
    fp = pred.sum(axis=(1, 2)) * (past > 0)
    return float(fp[past <= k].sum() / fp.sum()) if fp.sum() else float("nan")


def over_propagation(pred, gt, margin=10):
    """Share of tumour-free slices (within `margin` of the tumour's ends) that still got a mask."""
    z = tumour_slices(gt)
    outside = [s for s in range(max(z[0] - margin, 0), min(z[-1] + margin + 1, gt.shape[0])) if s < z[0] or s > z[-1]]
    return float(pred[outside].any(axis=(1, 2)).mean()) if outside else float("nan")
