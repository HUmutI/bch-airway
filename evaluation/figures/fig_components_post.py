#!/usr/bin/env python
"""Fig. 2 (2026-10-18, Ege's proposal): released post-processing versus anchored selection
in the 0.9-year-old infant.

Top row, each model with its own released post-processing:
  (a) AirMorph-R  A = LCC of the 3-of-5 vote                                   (blue)
  (b) MEDPSeg-R   --post: largest 26-connected component (poly_pipeline.py)     (red)
  (c) TfeNet      postprocessing.py: largest 6-connected component + hole filling (purple)
  In this infant (b) and (c) keep a non-airway object (MEDPSeg-R epoch 35 and TfeNet).
Bottom row, our pipeline on the raw outputs:
  (d) union U = A | M~ | T~, coloured by the model that contributed each voxel
      (A blue, otherwise MEDPSeg-R red, otherwise TfeNet purple)
  (e) the largest component of U (what size-based selection keeps), same colours
  (f) the anchored component F (Eq. 1): green, enlarged
Below each panel: Dice against the final pediatric ground truth and the volume.
Panels (a)-(c) and (e) share one crop and scale; (d) shows the whole union; (f) is cropped to F. Volumes are
of the full masks (the thoracic-inlet clip is display only).

    MEDPSEG_PATH=<MEDPSeg-R raw nii on the case grid> FIG_OUT=<path> python fig_components_post.py
"""
import os, sys
import numpy as np
import cc3d
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import airway_surface_render as sr

ROOT = os.environ.get("AIRWAY_ROOT", "/fileserver/external/quin/ege")
D = os.environ.get("CASE_DIR", f"{ROOT}/airway-seg/age_00-01/sub-0012/sub-0012_acc-0295")
MEDPSEG_PATH = os.environ["MEDPSEG_PATH"]
OUT = os.environ["FIG_OUT"]
MARGIN_MM = 4.0
# one colour per model throughout: AirMorph-R blue, MEDPSeg-R red, TfeNet purple; F green
BLUE, RED, PURPLE, GREEN = (mc.to_rgb(c) for c in ("#347ac2", "#d62728", "#8a4fb8", "#1baf7a"))
if os.environ.get("MONO"):                              # single-colour variant: everything AirMorph blue
    RED = PURPLE = GREEN = BLUE


def lcc(m, conn):
    lab = cc3d.connected_components(m.astype(np.uint8), connectivity=conn)
    n = np.bincount(lab.ravel()); n[0] = 0
    return lab == n.argmax()


A, spacing = sr.load_lps(os.path.join(D, "airway_bin.nii.gz")); A = A > 0
Mr, _ = sr.load_lps(MEDPSEG_PATH); Mr = Mr > 0
Tr, _ = sr.load_lps(os.path.join(D, "tfenet_airway.nii.gz")); Tr = Tr > 0
lung, _ = sr.load_lps(os.path.join(D, "lunglobe.nii.gz")); lung = lung > 0
# ground truth on the members' grid, loaded the same way (for the Dice under each panel)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fusion_uncertainty"))
import common, SimpleITK as sitk, tempfile
_ref = sitk.ReadImage(os.path.join(D, "airway_bin.nii.gz"))
_g = sitk.GetImageFromArray(common._aligned_gt(common.ped_gt_path(os.path.basename(D), "pediatric_v3"),
                                               os.path.join(D, "airway_bin.nii.gz")).astype(np.uint8))
_g.CopyInformation(_ref); _tmp = os.path.join(tempfile.mkdtemp(), "gt.nii.gz"); sitk.WriteImage(_g, _tmp)
G, _ = sr.load_lps(_tmp); G = G > 0

M = lcc(Mr, 26)                                         # MEDPSeg --post
T = ndimage.binary_fill_holes(lcc(Tr, 6))               # TfeNet postprocessing.py
U = A | Mr | Tr
lab, n = ndimage.label(U, structure=ndimage.generate_binary_structure(3, 1))
hit = lab[A]; anc = int(np.bincount(hit[hit > 0]).argmax())
sizes = np.bincount(lab.ravel()); sizes[0] = 0; big = int(sizes.argmax())
F, L = lab == anc, lab == big
vox = float(np.prod(spacing)) / 1000.0
vols = {k: v.sum() * vox for k, v in dict(a=A, b=M, c=T, d=U, e=L, f=F).items()}
dice = {k: 200.0 * (v & G).sum() / (v.sum() + G.sum()) for k, v in dict(a=A, b=M, c=T, d=U, e=L, f=F).items()}
print(f"union {U.sum()} voxels in {n} components; anchored {F.sum()} voxels; largest {L.sum()} voxels; "
      f"largest == anchored: {anc == big}; TfeNet post overlaps A: {int((T & A).sum())} voxels")

# display only: thoracic-inlet clip, as in the original Fig. 2
inlet = np.where(lung.any(axis=(0, 1)))[0].max()
Mr_only, Tr_only = Mr & ~A, Tr & ~A & ~Mr                 # union voxels by contributing model
for m in (A, M, T, U, F, L, Mr_only, Tr_only):
    m[:, :, inlet + 1:] = False
lab[:, :, inlet + 1:] = 0


def crop_of(mask):
    pad = np.ceil(MARGIN_MM / spacing).astype(int); sl = []
    for ax, ext in enumerate(mask.shape):
        p = np.where(mask.any(axis=tuple(i for i in range(3) if i != ax)))[0]
        sl.append(slice(max(int(p.min()) - pad[ax], 0), min(int(p.max()) + pad[ax] + 1, ext)))
    return tuple(sl)


SDF = bool(os.environ.get("SDF"))                      # signed-distance surface: thin branches stay connected


def field(mask, sl, mm):
    if SDF:
        m = mask[sl].astype(bool)
        sd = ndimage.distance_transform_edt(m, sampling=spacing) - ndimage.distance_transform_edt(~m, sampling=spacing)
        f0 = np.clip(0.5 + sd / (2.0 * spacing.max()), 0.0, 1.0).astype(np.float32)
        f = ndimage.zoom(f0, spacing / mm, order=1, prefilter=False)
        return ndimage.gaussian_filter(f, 0.3 / mm, output=np.float32)
    f = ndimage.zoom(mask[sl].astype(np.float32), spacing / mm, order=1, prefilter=False)
    return ndimage.gaussian_filter(f, 0.45 / mm, output=np.float32)


def plain(mask, sl, mm, colour):
    sr.RENDER_MM = mm
    rgb, alpha = sr.shade(field(mask, sl, mm))
    lum = rgb.mean(axis=2, keepdims=True)
    return np.dstack([np.clip(lum * np.asarray(colour) * 1.45, 0, 1), alpha])


def labelled(region, sl, mm):
    """Surface of `region`, coloured by the model that contributed each voxel."""
    sr.RENDER_MM = mm
    cls = np.where(region & A, 1, np.where(region & Mr_only, 2, np.where(region & Tr_only, 3, 0))).astype(np.int16)
    f = field(region, sl, mm)
    z = ndimage.zoom(cls[sl], spacing / mm, order=0, prefilter=False)
    g = np.zeros(f.shape, np.int16); t = tuple(slice(0, min(a, b)) for a, b in zip(z.shape, f.shape)); g[t] = z[t]
    _, near = ndimage.distance_transform_edt(g == 0, return_indices=True)
    g = g[tuple(near)]
    lut = np.array([[0, 0, 0], BLUE, RED, PURPLE])
    rgb, alpha, _ = sr.shade_labeled(f, g, lambda k: lut[np.clip(k, 0, 3)])
    return np.dstack([rgb, alpha])


# (a)-(c) and (e) share one crop around what they show; (d) shows the whole union; (f) is F alone
so, su, sf = crop_of(A | M | T | L), crop_of(U), crop_of(F)
# render grid (mm): FINE=1 halves it for print
g_o, g_u, g_f = (0.15, 0.25, 0.1) if os.environ.get("FINE") else (0.3, 0.4, 0.2)
panels = [plain(A, so, g_o, BLUE), plain(M, so, g_o, RED), plain(T, so, g_o, PURPLE),
          labelled(U, su, g_u), labelled(L, so, g_o), plain(F, sf, g_f, GREEN)]
labels = ["(a) $A$", "(b) $M$", "(c) $T$", r"(d) $A\cup\tilde M\cup\tilde T$", "(e) largest", "(f) anchored $F$"]
keys = "abcdef"
fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6))
for ax, im, lab_, k in zip(axes.ravel(), panels, labels, keys):
    im = im[..., :3] * im[..., 3:] + (1 - im[..., 3:])
    ax.imshow(np.clip(im, 0, 1), interpolation="lanczos"); ax.set_axis_off()
    ax.text(0.5, -0.02, f"{lab_}\nDice {dice[k]:.1f}, {vols[k]:.1f} mL", transform=ax.transAxes, ha="center", va="top", fontsize=10)
fig.subplots_adjust(wspace=0.03, hspace=0.25, left=0.01, right=0.99, top=0.99, bottom=0.1)
for ext in os.environ.get("FIG_FORMATS", "pdf,png").split(","):
    fig.savefig(f"{OUT}.{ext}", dpi=int(os.environ.get("FIG_DPI", "600")), bbox_inches="tight", facecolor="white")
print("wrote", OUT, {k: (round(dice[k], 1), round(v, 2)) for k, v in vols.items()})
