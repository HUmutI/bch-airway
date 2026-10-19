#!/usr/bin/env python
"""Five-panel figure: each ensemble member, their union, and the selected component.

Every panel is the same scan on ONE shared crop and ONE isotropic grid, so what
differs on the page is the mask and nothing else. The shared crop is taken from
the union, which contains all five, and the final component is highlighted by
colour rather than by scale.
"""
import os, sys, numpy as np, nibabel as nib
ROOT = os.environ.get("AIRWAY_ROOT", "/path/to/airway-data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
import airway_surface_render as sr

# the 0.9-year-old infant used throughout the paper, and its expert annotation
D = os.environ.get("CASE_DIR", f"{ROOT}/airway-seg/age_00-01/sub-0012/sub-0012_acc-0295")
GT = os.environ.get("GT_PATH", f"{ROOT}/pediatric_annotations/anish_revised__sub-0012_acc-0295.nii.gz")
OUT = os.environ.get("FIG_OUT", f"{ROOT}/paper/figures/01_member_panels")
MARGIN_MM = 4.0

files = ["airway_bin.nii.gz", "medpseg_airway.nii.gz", "tfenet_airway.nii.gz"]
# Only the letter and the volume are set inside the figure. This canvas is 15
# inches wide and lands at the text width of a proceedings page, a reduction of
# about two thirds, so a model name set here would print at four points. The
# names are in the caption, which reads each panel in order anyway.
titles = ["(a)", "(b)", "(c)", "(d)", "(e)"]

# MEDPSEG_PATH swaps in another MEDPSeg mask on the same grid (e.g. MEDPSeg-R); the final
# mask is then recomputed with the method's rule instead of read from union3_lcc.nii.gz.
MEDPSEG_PATH = os.environ.get("MEDPSEG_PATH")
masks, spacing = [], None
for f in files:
    path = MEDPSEG_PATH if (MEDPSEG_PATH and f == "medpseg_airway.nii.gz") else os.path.join(D, f)
    m, sp = sr.load_lps(path); masks.append(m > 0); spacing = sp
if MEDPSEG_PATH:
    lab, _ = ndimage.label(masks[0] | masks[1] | masks[2], structure=ndimage.generate_binary_structure(3, 1))
    hit = lab[masks[0]]; hit = hit[hit > 0]
    final = lab == int(np.bincount(hit).argmax())          # anchored selection, Eq. (1)
else:
    final, _ = sr.load_lps(os.path.join(D, "union3_lcc.nii.gz")); final = final > 0
# volumes of the unclipped masks (the inlet clip below is display only)
raw_masks, raw_final = [m.copy() for m in masks], final.copy()
lung, _ = sr.load_lps(os.path.join(D, "lunglobe.nii.gz")); lung = lung > 0

# thoracic-inlet clip, applied identically to every panel
inlet = np.where(lung.any(axis=(0, 1)))[0].max()
for m in masks: m[:, :, inlet + 1:] = False
final[:, :, inlet + 1:] = False
union = masks[0] | masks[1] | masks[2]

# One shared bounding box for all five panels, taken from the union so that
# nothing any member produced is cropped away. The spurious material is the
# point of the figure, so the frame has to contain it, and the airway therefore
# occupies only part of each panel.
pad = np.ceil(MARGIN_MM / spacing).astype(int)
sl = []
for axis, extent in enumerate(union.shape):
    present = np.where(union.any(axis=tuple(i for i in range(3) if i != axis)))[0]
    sl.append(slice(max(int(present.min()) - pad[axis], 0),
                    min(int(present.max()) + pad[axis] + 1, extent)))
sl = tuple(sl)

def field_of(mask, render_mm=0.4, clean=False):
    """Isosurface field for one panel.

    `render_mm` sets the sampling of the render grid, not the crop, so a finer
    value gives the same physical volume more pixels. Smoothing is held at a
    constant 0.45 mm physically, otherwise a finer grid would simply expose the
    voxel staircase. `clean` drops pieces that smoothing has pinched off from
    the main structure, which is wanted for the recovered airway and expressly
    not wanted for the members, whose debris is the point of the figure.
    """
    f = ndimage.zoom(mask[sl].astype(np.float32), spacing / render_mm,
                     order=1, prefilter=False)
    f = ndimage.gaussian_filter(f, 0.45 / render_mm, output=np.float32)
    if clean:
        lab, n = ndimage.label(f > sr.LEVEL,
                               structure=ndimage.generate_binary_structure(3, 3))
        if n > 1:
            sz = np.bincount(lab.ravel()); sz[0] = 0
            f = np.where(lab == sz.argmax(), f, 0.0).astype(np.float32)
            print(f"  panel cleanup: removed {n-1} pinched-off fragment(s)")
    return f

def ml(mask):
    return mask.sum() * float(np.prod(spacing)) / 1000.0


# Dice of every panel against the expert annotation, on the unclipped masks,
# exactly as benchmark_pediatric scores them (the inlet clip above is display
# only). The annotation is on the source grid; load it the same way.
def dice_table():
    import SimpleITK as sitk
    ref = sitk.ReadImage(os.path.join(D, files[0]))
    g = sitk.ReadImage(GT)
    assert g.GetSize() == ref.GetSize(), (g.GetSize(), ref.GetSize())
    dg = np.array(g.GetDirection()).reshape(3, 3); dr = np.array(ref.GetDirection()).reshape(3, 3)
    gt = sitk.GetArrayFromImage(g) > 0
    for axis_sitk, sgn in enumerate(np.diag(dg) / np.diag(dr)):
        if sgn < 0:
            gt = np.flip(gt, 2 - axis_sitk)
    raw = {f: sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(D, f))) > 0 for f in files}
    a, m, t = (raw[f] for f in files)
    u = a | m | t
    fin = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(D, "union3_lcc.nii.gz"))) > 0
    def sel(x):
        lab, n = ndimage.label(x, structure=ndimage.generate_binary_structure(3, 1))
        hit = lab[a]; hit = hit[hit > 0]
        return lab == int(np.bincount(hit).argmax())
    d = lambda p: 100.0 * 2 * (p & gt).sum() / (p.sum() + gt.sum())
    return [[d(a), d(m), d(t), d(u), d(fin)], [d(sel(a)), d(sel(m)), d(sel(t)), d(sel(u))]]

def seeded(mask, seed):
    """The component of `mask` holding the most voxels of `seed`, 6-connected.

    This is the selection rule the method applies to the union. Applied to a
    member on its own it answers a different question, which is what the member
    contributes once its detached material is set aside, and that is what the
    second row of the figure shows.
    """
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if n == 0:
        return np.zeros_like(mask)
    hit = lab[seed]
    hit = hit[hit > 0]
    if hit.size == 0:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        return lab == sz.argmax()
    return lab == int(np.bincount(hit).argmax())


# Row one is every mask as its model produced it, where the spurious material is
# the point. Row two is each of them reduced to the component that carries the
# AirMorph seed, which is the airway alone. The last column is already the
# result of that rule, so it repeats unchanged and shows the two rows agreeing.
seed = masks[0]
_r = raw_masks; _u = _r[0] | _r[1] | _r[2]
full_vols = [[ml(_r[0]), ml(_r[1]), ml(_r[2]), ml(_u), ml(raw_final)],
             [ml(seeded(_r[0], _r[0])), ml(seeded(_r[1], _r[0])), ml(seeded(_r[2], _r[0])), ml(seeded(_u, _r[0]))]]
rows = [
    [(masks[0], None), (masks[1], None), (masks[2], None),
     (union, None), (final, "#1baf7a")],
    [(seeded(masks[0], seed), None), (seeded(masks[1], seed), None),
     (seeded(masks[2], seed), None), (seeded(union, seed), "#1baf7a")],
]
# The fifth column has no second row. Selection is what that column already is,
# so repeating it would print the same mask and the same volume twice.
letters = [["(a)", "(b)", "(c)", "(d)", "(e)"],
           ["(f)", "(g)", "(h)", "(i)"]]

# Shading ten panels takes about half an hour, and the layout usually needs
# several attempts, so the images are cached. Delete the cache to re-render.
import matplotlib.colors as mc
CACHE = OUT + "_panels.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE)
    imgs = [[z[f"p{r}{c}"] for c in range(len(rows[r]))] for r in range(2)]
    vols = z["vols"]
    print("loaded cached panels from", CACHE)
else:
    imgs, vols = [], []
    for r, row in enumerate(rows):
        out, vr = [], []
        for mask, colour in row:
            final_panel = colour is not None
            # The second row is displayed large, so it is sampled on the fine
            # grid as well. At 0.40 mm the enlargement shows the voxel staircase.
            sr.RENDER_MM = 0.20 if (final_panel or r == 1) else 0.40
            rgb, alpha = sr.shade(field_of(mask, render_mm=sr.RENDER_MM,
                                           clean=final_panel or r == 1))
            if colour is not None:
                tint = np.array(mc.to_rgb(colour))
                lum = rgb.mean(axis=2, keepdims=True)
                rgb = np.clip(lum * tint * 1.45, 0, 1)
            out.append(np.dstack([rgb, alpha])); vr.append(ml(mask))
        imgs.append(out); vols.append(vr)
    np.savez_compressed(CACHE, vols=np.array([v + [0.0] * (5 - len(v)) for v in vols]),
                        **{f"p{r}{c}": imgs[r][c]
                           for r in range(2) for c in range(len(imgs[r]))})
    print("cached panels to", CACHE)

# Both rows are drawn on the crop taken from the union, so the second row
# reserves the height where the first row's debris sits and shows nothing there.
# Each row is therefore cut to the rows of pixels that row actually uses, and the
# figure gives each one a height in proportion. The millimetres per pixel are
# identical either way, so the panels stay comparable across rows, and the blank
# band disappears.
def content_box(row):
    """Rows and columns of pixels the row uses, as fractions of each panel."""
    r0, r1, c0, c1 = 1.0, 0.0, 1.0, 0.0
    for im in row:
        a = im[..., 3] > 0.01
        ri, ci = np.where(a.any(axis=1))[0], np.where(a.any(axis=0))[0]
        if ri.size:
            r0 = min(r0, ri.min() / im.shape[0]); r1 = max(r1, (ri.max() + 1) / im.shape[0])
            c0 = min(c0, ci.min() / im.shape[1]); c1 = max(c1, (ci.max() + 1) / im.shape[1])
    m = 0.03
    return max(r0 - m, 0.0), min(r1 + m, 1.0), max(c0 - m, 0.0), min(c1 + m, 1.0)


# The first row keeps the crop taken from the union, because the spurious
# material is what it is there to show and cropping it away would defeat the
# figure. The second row holds only airway, so it is cropped to that and drawn
# larger. The two rows are therefore not at one scale, and the volume printed
# above each panel is what carries size between them.
boxes = [(0.0, 1.0, 0.0, 1.0), content_box(imgs[1])]
imgs = [[im[int(r0 * im.shape[0]):int(r1 * im.shape[0]),
            int(c0 * im.shape[1]):int(c1 * im.shape[1])] for im in row]
        for row, (r0, r1, c0, c1) in zip(imgs, boxes)]
aspects = [im[0].shape[0] / im[0].shape[1] for im in imgs]
print("  row aspects after cropping:", [round(a, 3) for a in aspects])

LABEL = os.environ.get("PANEL_LABEL", "dice")             # "volume": mL of the unclipped masks
dices = dice_table() if LABEL == "dice" and not MEDPSEG_PATH else None
for r in range(2):
    print(f"  row {r+1}:", [f"{v:.1f}" for v in (dices or full_vols)[r]])
NCOL = 5
PANEL_W = 15.2 / NCOL
fig_h = PANEL_W * sum(aspects) + 1.5
fig, axgrid = plt.subplots(2, NCOL, figsize=(15.2, fig_h),
                           gridspec_kw={"height_ratios": aspects})
for r in range(2):
    for c in range(NCOL):
        ax = axgrid[r][c]
        ax.set_axis_off()
        if c >= len(imgs[r]):
            continue
        ax.imshow(imgs[r][c])
        # the letter and the Dice under the panel; volumes are in the caption text
        txt = f"Dice {dices[r][c]:.1f}" if dices else f"{full_vols[r][c]:.1f} mL"
        ax.text(0.5, -0.03, f"{letters[r][c]}  {txt}", transform=ax.transAxes,
                ha="center", va="top", fontsize=24, color="#1a1a19")
fig.subplots_adjust(wspace=0.015, hspace=0.18, left=0.005, right=0.995,
                    top=0.995, bottom=0.06)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", OUT + ".pdf")
for r in range(2):
    print(f"  row {r+1} volumes:", [f"{v:.1f}" for v in vols[r][:len(imgs[r])]])
