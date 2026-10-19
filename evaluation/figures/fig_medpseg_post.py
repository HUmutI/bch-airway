#!/usr/bin/env python
"""MEDPSeg raw output against MEDPSeg with its own --post step (largest
26-connected component, poly_pipeline.py), beside the reference.

One row per case; the crop is taken from the raw mask so that what --post removes
stays in frame. Same depth-shaded raycast as the other figures.

    FIG_CASES='[["pediatric_v3","sub-0012_acc-0295",0.2,"infant, 0.9 y"],["test","ATM_xxx",0.4,"adult"]]' \
    FIG_OUT=.../medpseg_post python fig_medpseg_post.py
"""
import json, os, sys
import numpy as np
import cc3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "fusion_uncertainty"))
import airway_surface_render as sr
import common
from fig_confidence_panels import to_lps, crop_box, field_and_labels, render_plain, dice, content_box
import fig_confidence_panels as fcp

CASES = [tuple(c) for c in json.loads(os.environ["FIG_CASES"])]
OUT = os.environ["FIG_OUT"]
BLUE = mc.to_rgb("#347ac2"); RED = np.array([0.84, 0.15, 0.16])


def lcc26(m):
    lab = cc3d.connected_components(m.astype(np.uint8), connectivity=26)
    n = np.bincount(lab.ravel()); n[0] = 0
    return lab == n.argmax() if n.max() else m


def panels(split, cid, mm):
    fcp.USE_SDF = split.startswith("pediatric")
    c = common.load_cached(split, cid)
    sp = np.asarray(c["spacing"], float)[::-1]
    raw, gt = to_lps(c["medpseg"].astype(bool)), to_lps(c["gt"].astype(bool))
    post = lcc26(raw)
    sl = crop_box(raw | gt, sp)
    sr.RENDER_MM = mm
    out = []
    for m, tint in ((raw, BLUE), (post, BLUE), (gt, RED)):
        f, _ = field_and_labels(m, None, sl, sp, mm)
        out.append(render_plain(f, tint=np.asarray(tint)))
    return out, [dice(raw, gt), dice(post, gt)], [m.sum() * np.prod(sp) / 1000 for m in (raw, post)]


rows, stats = [], []
for split, cid, mm, _ in CASES:
    print("rendering", cid, flush=True)
    p, d, v = panels(split, cid, mm); rows.append(p); stats.append((d, v))
rows = [[im[int(r0 * im.shape[0]):int(r1 * im.shape[0]), int(c0 * im.shape[1]):int(c1 * im.shape[1])] for im in row]
        for row, (r0, r1, c0, c1) in zip(rows, [content_box(r) for r in rows])]
asp = [r[0].shape[0] / r[0].shape[1] for r in rows]
fig, axes = plt.subplots(len(rows), 3, squeeze=False, figsize=(12, 4 * sum(asp) + 1.2), gridspec_kw={"height_ratios": asp})
for r, (row, (d, v), (_, _, _, desc)) in enumerate(zip(rows, stats, CASES)):
    labs = [f"MEDPSeg raw (default)\nDice {d[0]:.1f}, {v[0]:.1f} mL", f"MEDPSeg --post\nDice {d[1]:.1f}, {v[1]:.1f} mL", "Reference"]
    for c in range(3):
        im = row[c]; im = im[..., :3] * im[..., 3:] + (1 - im[..., 3:])
        ax = axes[r][c]; ax.set_axis_off(); ax.imshow(np.clip(im, 0, 1), interpolation="lanczos")
        ax.text(0.5, -0.02, f"({'abcdefghi'[3 * r + c]}) {labs[c]}", transform=ax.transAxes, ha="center", va="top", fontsize=15)
    axes[r][0].text(-0.03, 0.5, desc, transform=axes[r][0].transAxes, rotation=90, ha="right", va="center", fontsize=14, color="#555")
fig.subplots_adjust(wspace=0.02, hspace=0.25, left=0.04, right=0.995, top=0.99, bottom=0.08)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", OUT + ".pdf", stats)
