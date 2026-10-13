#!/usr/bin/env python
"""Four-panel 3D figure of the uncertainty-aware ensemble, one row per case.

(a) the fused mask (union, seeded component) with its surface coloured by the
    calibrated vote-pattern confidence, red (one model alone) to green (all
    seven predictors agree);
(b) the same surface coloured by the reference: blue where the voxel is
    airway, red where it is a false positive;
(c) the refined mask (generation-specific consensus: G1-3 confidence >= 0.4,
    G4-6 >= 5 of 7 predictors, G7+ confidence >= 0.2, 1-voxel core kept),
    green as the selected component is drawn elsewhere in the paper;
(d) the reference annotation, grey.
Dice against the reference is printed under (a)/(b) and (c). Rendering is the
same depth-shaded raycast as the other figures (airway_surface_render), on one
shared crop and one isotropic grid per row.

Reads the study cache (fusion_uncertainty_run/cache/<split>/<case>/volumes.npz)
and the refined masks
(fusion_uncertainty_run/ablation/generation_specific/masks/<split>/).

    AIRWAY_ROOT=/fileserver/external/quin/ege python fig_confidence_panels.py
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "fusion_uncertainty"))
import airway_surface_render as sr
import common

ROOT = os.environ.get("AIRWAY_ROOT", "/fileserver/external/quin/ege")
STUDY = os.path.join(ROOT, "fusion_uncertainty_run")
OUT = os.environ.get("FIG_OUT", os.path.join(ROOT, "paper", "figures", "08_confidence_panels"))
REFINED = os.path.join(STUDY, "ablation", "generation_specific", "masks")
# Pediatric references: the final radiologist-reviewed set (split "pediatric_v3"), see common.py.
PED_SPLIT = os.environ.get("AIRWAY_PED_SPLIT", "pediatric_v3")
CASES = [("test", "ATM_078", 0.5, "adult, ATM'22 test"), (PED_SPLIT, "sub-0012_acc-0295", 0.2, "infant, 0.9 years")]
# FIG_CASES='[["test","ATM_610",0.3,"row label"], ...]' replaces the default rows.
if os.environ.get("FIG_CASES"):
    CASES = [tuple(c) for c in json.loads(os.environ["FIG_CASES"])]
DPI = int(os.environ.get("FIG_DPI", "300"))
# FIG_FORMATS="pdf,png,eps,tiff"; EPS has no alpha, so panels are flattened on white.
FORMATS = os.environ.get("FIG_FORMATS", "pdf,png").split(",")
# Cases whose branches are only 1-2 voxels wide: a blurred binary field drops their
# centres below the 0.5 level and the render breaks connected branches apart, so
# the surface is taken from a signed distance field instead.
SDF_CASES = {"sub-0012_acc-0295"}
if os.environ.get("SDF_ALL_PED"):
    SDF_CASES |= {cid for split, cid, *_ in CASES if split.startswith("pediatric")}
USE_SDF = False
MARGIN_MM = 4.0
RAMP = LinearSegmentedColormap.from_list("conf", [(0, "#d62728"), (0.5, "#ffc800"), (1, "#2ca02c")])
BLUE = np.array([0.204, 0.478, 0.761]); RED = np.array([0.84, 0.15, 0.16]); GREY = np.array([0.62, 0.62, 0.60])
# REF_TINT="#rrggbb" colours the reference panel (default grey).
REF_TINT = np.array(mc.to_rgb(os.environ["REF_TINT"])) if os.environ.get("REF_TINT") else GREY * 1.15


def retint(panel, old, new):
    """Re-colour a render_plain panel: recover the shading luminance from `old`, apply `new`."""
    lum = panel[..., 0:1] / (old[0] * 1.45)
    return np.dstack([np.clip(lum * new * 1.45, 0, 1), panel[..., 3]])


def to_lps(a):
    """SimpleITK (z, y, x) array on an LPS grid -> (x, y, z) as the renderer wants."""
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))


def crop_box(mask, spacing):
    pad = np.ceil(MARGIN_MM / spacing).astype(int)
    sl = []
    for axis, extent in enumerate(mask.shape):
        present = np.where(mask.any(axis=tuple(i for i in range(3) if i != axis)))[0]
        sl.append(slice(max(int(present.min()) - pad[axis], 0), min(int(present.max()) + pad[axis] + 1, extent)))
    return tuple(sl)


def field_and_labels(mask, labels, sl, spacing, render_mm):
    """Isosurface field of `mask` and nearest-filled `labels` on the render grid."""
    zoom = spacing / render_mm
    if USE_SDF:
        # 0.5 at the voxel faces, > 0.5 at every inside voxel centre, so thin
        # 6-connected branches stay connected after interpolation
        m = mask[sl].astype(bool)
        sd = ndimage.distance_transform_edt(m, sampling=spacing) - ndimage.distance_transform_edt(~m, sampling=spacing)
        f0 = np.clip(0.5 + sd / (2.0 * spacing.max()), 0.0, 1.0).astype(np.float32)
        field = ndimage.zoom(f0, zoom, order=1, prefilter=False)
        field = ndimage.gaussian_filter(field, 0.3 / render_mm, output=np.float32)
    else:
        field = ndimage.zoom(mask[sl].astype(np.float32), zoom, order=1, prefilter=False)
        field = ndimage.gaussian_filter(field, 0.45 / render_mm, output=np.float32)
    if labels is None:
        return field, None
    lab = ndimage.zoom(np.where(mask[sl], labels[sl], 0).astype(np.int16), zoom, order=0, prefilter=False)
    grown = np.zeros(field.shape, np.int16)
    tr = tuple(slice(0, min(a, b)) for a, b in zip(lab.shape, field.shape))
    grown[tr] = lab[tr]
    _, near = ndimage.distance_transform_edt(grown == 0, return_indices=True)
    return field, grown[tuple(near)]


def render_plain(field, tint=None):
    rgb, alpha = sr.shade(field)
    if tint is not None:
        lum = rgb.mean(axis=2, keepdims=True)
        rgb = np.clip(lum * np.asarray(tint) * 1.45, 0, 1)
    return np.dstack([rgb, alpha])


def render_labeled(field, labels, lut):
    """lut: (n, 3) array indexed by label."""
    rgb, alpha, _ = sr.shade_labeled(field, labels, lambda lab: lut[np.clip(lab, 0, len(lut) - 1)])
    return np.dstack([rgb, alpha])


def dice(a, b):
    return 200.0 * float((a & b).sum()) / max(1, int(a.sum()) + int(b.sum()))


def panels_for(split, cid, render_mm):
    global USE_SDF
    USE_SDF = cid in SDF_CASES
    c = common.load_cached(split, cid)
    table = common.pattern_table("val")
    conf = common.confidence_from_pattern(c["pattern"], "calibrated", table)
    M, gt = c["M"], c["gt"].astype(bool)
    R = np.load(os.path.join(REFINED, split.replace("_v3", ""), f"ATM26_{cid}.npz"))["mask"].astype(bool)
    spacing = np.asarray(c["spacing"], float)[::-1]           # (z,y,x) -> (x,y,z)
    M, gt, R, conf = to_lps(M), to_lps(gt), to_lps(R), to_lps(conf)
    sl = crop_box(M | gt, spacing)
    sr.RENDER_MM = render_mm

    # (a) confidence: 20 bins
    cbin = np.clip(np.ceil(conf * 20).astype(np.int16), 1, 20)
    lut_c = np.vstack([[0, 0, 0], RAMP(np.linspace(0.025, 0.975, 20))[:, :3]])
    f, lab = field_and_labels(M, cbin, sl, spacing, render_mm)
    pa = render_labeled(f, lab, lut_c)
    # (b) reference status on the same surface
    status = np.where(gt, 1, 2).astype(np.int16)
    _, lab = field_and_labels(M, status, sl, spacing, render_mm)
    pb = render_labeled(f, lab, np.vstack([[0, 0, 0], BLUE, RED]))
    # (c) refined, (d) reference
    fr, _ = field_and_labels(R, None, sl, spacing, render_mm)
    pc = render_plain(fr, tint=mc.to_rgb("#1baf7a"))
    fg, _ = field_and_labels(gt, None, sl, spacing, render_mm)
    pd_ = render_plain(fg, tint=REF_TINT)
    stats = {"dice_union": dice(M, gt), "dice_refined": dice(R, gt),
             "fp_union": 100 * float((M & ~gt).sum()) / M.sum(), "fp_refined": 100 * float((R & ~gt).sum()) / R.sum(),
             "low": 100 * float((M & (conf < 0.5)).sum()) / M.sum()}
    return [pa, pb, pc, pd_], stats


def missed_panel(split, cid, render_mm):
    """(b'), OVERLAY=missed: the fused mask in blue and, in red, every reference
    branch it misses by the BND rule (< 80 % of the branch centreline inside the
    mask), drawn whole, including the part the mask does cover; so the red
    branches are exactly the undetected branches counted by BND."""
    global USE_SDF
    USE_SDF = cid in SDF_CASES
    c = common.load_cached(split, cid)
    M, gt = c["M"].astype(bool), c["gt"].astype(bool)
    par, skel = c["gt_parsing"].astype(np.int64), c["gt_skeleton"].astype(bool)
    lab_b = np.where(skel, par, 0)
    total = np.bincount(lab_b.ravel())
    inside = np.bincount(np.where(M, lab_b, 0).ravel(), minlength=total.size)
    ids = np.nonzero(total[1:])[0] + 1
    missed_ids = ids[inside[ids] < 0.8 * total[ids]]
    missed = np.isin(par, missed_ids) & gt
    bnd = 100.0 * (ids.size - missed_ids.size) / ids.size
    spacing = np.asarray(c["spacing"], float)[::-1]
    M, gt, missed = to_lps(M), to_lps(gt), to_lps(missed)
    sl = crop_box(M | gt, spacing)
    sr.RENDER_MM = render_mm
    status = np.where(missed, 2, 1).astype(np.int16)
    f, lab = field_and_labels(M | missed, status, sl, spacing, render_mm)
    return render_labeled(f, lab, np.vstack([[0, 0, 0], BLUE, RED])), bnd, int(missed_ids.size), int(ids.size)


def content_box(row, m=0.02):
    r0, r1, c0, c1 = 1.0, 0.0, 1.0, 0.0
    for im in row:
        a = im[..., 3] > 0.01
        ri, ci = np.where(a.any(axis=1))[0], np.where(a.any(axis=0))[0]
        if ri.size:
            r0 = min(r0, ri.min() / im.shape[0]); r1 = max(r1, (ri.max() + 1) / im.shape[0])
            c0 = min(c0, ci.min() / im.shape[1]); c1 = max(c1, (ci.max() + 1) / im.shape[1])
    return max(r0 - m, 0.0), min(r1 + m, 1.0), max(c0 - m, 0.0), min(c1 + m, 1.0)


def main():
    tag = "" if not os.environ.get("FIG_CASES") else "_" + "_".join(f"{c[1]}@{c[2]}" for c in CASES)
    cache = OUT + f"_genspec_{PED_SPLIT}_sdf_panels{tag}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        rows = [[z[f"p{r}{c}"] for c in range(4)] for r in range(len(CASES))]
        if os.environ.get("REF_TINT"):                 # cached panels were rendered grey
            for row in rows:
                row[3] = retint(row[3], GREY * 1.15, REF_TINT)
        stats = list(z["stats"])
        print("loaded cached panels")
    else:
        rows, stats = [], []
        for split, cid, mm, _ in CASES:
            print("rendering", cid, flush=True)
            p, s = panels_for(split, cid, mm)
            rows.append(p); stats.append(s)
        np.savez_compressed(cache, stats=np.array(stats, dtype=object),
                            **{f"p{r}{c}": rows[r][c] for r in range(len(rows)) for c in range(4)})
    if os.environ.get("OVERLAY") == "missed":
        mcache = cache.replace(".npz", "_missed.npz")
        if os.path.exists(mcache):
            z = np.load(mcache); miss = [z[f"b{r}"] for r in range(len(CASES))]
        else:
            miss = []
            for split, cid, mm, _ in CASES:
                print("rendering missed-branch overlay", cid, flush=True)
                pb, bnd, nm, nt = missed_panel(split, cid, mm)
                print(f"  {cid}: {nm} of {nt} reference branches missed, BND {bnd:.2f}", flush=True)
                miss.append(pb)
            np.savez_compressed(mcache, **{f"b{r}": b for r, b in enumerate(miss)})
        for row, b in zip(rows, miss):
            row[1] = b
    rows = [[im[int(r0 * im.shape[0]):int(r1 * im.shape[0]), int(c0 * im.shape[1]):int(c1 * im.shape[1])]
             for im in row] for row, (r0, r1, c0, c1) in zip(rows, [content_box(r) for r in rows])]
    # PANEL_COLS selects panels: "abcd" (default) or "abd" (no refined mask)
    keep = ["abcd".index(ch) for ch in os.environ.get("PANEL_COLS", "abcd")]
    nc = len(keep)
    aspects = [row[0].shape[0] / row[0].shape[1] for row in rows]
    W = 15.2 * nc / 4
    fig, axes = plt.subplots(len(rows), nc, squeeze=False, figsize=(W, W / nc * sum(aspects) + 1.6),
                             gridspec_kw={"height_ratios": aspects})
    letters = "abcdefgh"
    for r, (row, s, (split, cid, _, desc)) in enumerate(zip(rows, stats, CASES)):
        labels = [f"Dice {s['dice_union']:.1f}", f"Dice {s['dice_union']:.1f}", f"Dice {s['dice_refined']:.1f}", "reference"]
        if os.environ.get("OVERLAY") == "missed":
            labels = ["Confidence map", "Segmentation (blue) vs.\nreference (red)", "", "Reference (red)"]
        for c, src in enumerate(keep):
            im = row[src]
            im = im[..., :3] * im[..., 3:] + (1.0 - im[..., 3:])     # flatten on white
            ax = axes[r][c]; ax.set_axis_off(); ax.imshow(np.clip(im, 0, 1), interpolation="lanczos")
            ax.text(0.5, -0.03, f"({letters[nc * r + c]})  {labels[src]}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=int(os.environ.get("LABEL_FS", "24")), color="#1a1a19")
        axes[r][0].text(-0.02, 0.5, desc, transform=axes[r][0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=18, color="#555")
    # colour bar for the confidence
    cax = fig.add_axes([0.04, 0.975, 0.18 * 4 / nc, 0.012])
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(0, 1)); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.ax.set_xticklabels(["0", "P(airway)  0.5", "1"], fontsize=14); cb.outline.set_linewidth(0.5)
    fig.subplots_adjust(wspace=0.02, hspace=float(os.environ.get("FIG_HSPACE", "0.16")), left=0.02, right=0.995, top=0.94, bottom=0.06)
    for ext in FORMATS:
        fig.savefig(f"{OUT}.{ext}", dpi=DPI, bbox_inches="tight", facecolor="white",
                    **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}))
    print("wrote", OUT + ".pdf")
    for (split, cid, _, _), s in zip(CASES, stats):
        print(cid, {k: round(float(v), 2) for k, v in s.items()})


if __name__ == "__main__":
    main()
