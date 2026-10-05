#!/usr/bin/env python
"""One case, three panels: the fused airway projected coronally and coloured
by its confidence, the same projection coloured by what the ground truth says
(correct / false positive), and one axial slice of the CT with the confidence
overlaid. The point of the figure is that the two projections agree: where the
map is red, the voxels are wrong.

    python fig_example.py --split val --case ATM_043 --out fig_example_ATM_043.png
"""
import argparse, glob, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import SimpleITK as sitk

import common

CMAP = LinearSegmentedColormap.from_list("conf", [(0, "#d62728"), (0.5, "#ffc800"), (1, "#2ca02c")])


def ct_path(split, cid):
    if split == "val":
        return os.path.join(common.VAL_RUN, "data/ATM26/imagesTr", f"{cid}_0000.mha")
    if split == "pediatric":
        return os.path.join(common.ped_case_dir(cid), "image.nii.gz")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper", action="store_true", help="no in-figure titles (the caption carries them)")
    a = ap.parse_args()
    c = common.load_cached(a.split, a.case)
    table = common.pattern_table("val")
    conf = common.confidence_from_pattern(c["pattern"], "calibrated", table)
    M = c["M"]; gt = c["gt"].astype(bool)
    sp = c["spacing"]                                      # z, y, x
    sl = ndimage_bbox = None
    zs, ys, xs = np.nonzero(M)
    z0, z1, x0, x1 = zs.min(), zs.max() + 1, xs.min(), xs.max() + 1
    # coronal projections (collapse y): least confident voxel along the ray;
    # and error status: red if any FP on the ray and no TP, grey if TP
    # mean confidence along the ray (the minimum is pure boundary speckle)
    cnt = M.sum(axis=1).astype(np.float32)
    proj_conf = (np.where(M, conf, 0.0).sum(axis=1) / np.maximum(cnt, 1))[z0:z1, x0:x1]
    any_m = M.any(axis=1)[z0:z1, x0:x1]
    fp = (M & ~gt); tp = (M & gt)
    proj_fp = fp.any(axis=1)[z0:z1, x0:x1]; proj_tp = tp.any(axis=1)[z0:z1, x0:x1]
    aspect = sp[0] / sp[2]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6 if a.paper else 5.2))
    ax = axes[0]
    img = np.ma.masked_where(~any_m, proj_conf)
    ax.imshow(np.zeros_like(proj_conf), cmap="gray", vmin=0, vmax=1, aspect=aspect)
    im = ax.imshow(img, cmap=CMAP, vmin=0, vmax=1, aspect=aspect, origin="upper")
    ax.invert_yaxis(); ax.set_title("" if a.paper else "confidence, mean along each coronal ray", fontsize=9); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="P(airway)")
    ax = axes[1]
    rgb = np.zeros(proj_conf.shape + (3,), float)
    rgb[proj_tp] = (0.75, 0.75, 0.75)
    rgb[proj_fp & ~proj_tp] = (0.84, 0.15, 0.16)
    rgb[proj_fp & proj_tp] = (0.95, 0.55, 0.2)
    ax.imshow(rgb, aspect=aspect, origin="upper"); ax.invert_yaxis(); ax.axis("off")
    ax.set_title("" if a.paper else "ground truth: grey correct, red false positive only, orange mixed ray", fontsize=9)
    n_fp = int(fp.sum()); n_m = int(M.sum())
    low = conf < 0.5
    stats = (f"false-positive voxels: {100*n_fp/n_m:.1f}% of the mask; {100*(fp & low).sum()/max(n_fp,1):.0f}% of them have P<0.5, "
             f"while P<0.5 covers only {100*(M & low).sum()/n_m:.1f}% of the mask")
    ax = axes[2]
    p = ct_path(a.split, a.case)
    if p and os.path.exists(p):
        ct = sitk.GetArrayFromImage(sitk.ReadImage(p))
        if a.split == "pediatric":
            pass
        # slice with the most low-confidence voxels
        z = int(np.argmax((M & low).sum(axis=(1, 2))))
        ys_, xs_ = np.nonzero(M[z]); pad = 25
        yy0, yy1 = max(0, ys_.min() - pad), min(M.shape[1], ys_.max() + pad); xx0, xx1 = max(0, xs_.min() - pad), min(M.shape[2], xs_.max() + pad)
        ax.imshow(np.clip(ct[z, yy0:yy1, xx0:xx1], -1000, 400), cmap="gray", vmin=-1000, vmax=400, aspect=sp[1] / sp[2])
        ov = np.ma.masked_where(~M[z, yy0:yy1, xx0:xx1], conf[z, yy0:yy1, xx0:xx1])
        ax.imshow(ov, cmap=CMAP, vmin=0, vmax=1, alpha=0.85, aspect=sp[1] / sp[2])
        ax.contour(gt[z, yy0:yy1, xx0:xx1].astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
        ax.set_title("" if a.paper else f"axial slice {z}: CT, confidence overlay, GT outline (cyan)", fontsize=9)
    else:
        ax.text(0.5, 0.5, "CT not on disk", ha="center"); 
    ax.axis("off")
    if not a.paper:
        fig.suptitle(f"{a.split} {a.case}: union + seeded component, coloured by calibrated vote-pattern confidence\n{stats}", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out, dpi=160); print("wrote", a.out)


if __name__ == "__main__":
    main()
