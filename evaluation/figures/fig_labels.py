#!/usr/bin/env python
"""Airway surface coloured by the assigned anatomical label, at print resolution.

Rendered on a 0.15 mm isotropic grid rather than the 0.4 mm used for the
overview panels, because this figure is read close up and the distal branches of
an infant airway are only a few voxels across. Only the connected airway is
drawn, so nothing appears as a floating fragment.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
from scipy import ndimage
import airway_surface_render as sr

sr.RENDER_MM = 0.20          # finer grid: this figure is inspected closely
sr.SMOOTH_VOX = 2.25         # 0.45 mm physical, the same as the coarse render

CASE = "EXAMPLE_CASE"
D = "/path/to/airway-data/airway-seg/AGE_BAND/SUBJECT/EXAMPLE_CASE"
OUT = "/path/to/airway-data/paper/figures/02_branch_labels"
TABLE = "/path/to/airway-data/or_all3_run/normal_airway_branch_morphometry.csv"

# The lobar label set puts the trachea and both main bronchi in one class, so
# the three are separated here using the main-bronchus branch identifiers the
# cohort table already records. Remaining branches carrying the tracheal label
# are proximal ones the vocabulary leaves unnamed and stay neutral.
ORDER = ["Trachea", "Right main bronchus", "Left main bronchus",
         "Right Upper Lobe", "Right Middle Lobe", "Right Lower Lobe",
         "Left Upper Lobe", "Left Lower Lobe", "Proximal, unnamed"]
# The three central airways take dark, clearly separated hues so they cannot be
# confused with the five lobar colours, which use the categorical slots in order.
COLOURS = ["#1a1a19", "#008300", "#8d5fd3",
           "#2a78d6", "#eb6834", "#1baf7a",
           "#eda100", "#e87ba4", "#b4b3aa"]

COHORT = "/path/to/airway-data/cohort90_results/cohort_90.csv"
c = pd.read_csv(COHORT); c = c[c.case_id == CASE].iloc[0]
ROOT, LMB, RMB = int(c.root_branch_id), int(c.lmb_branch_id), int(c.rmb_branch_id)

d = pd.read_csv(TABLE); d = d[d.case_id == CASE]
cls = {n: i + 1 for i, n in enumerate(ORDER)}
LOBE = {"Right Upper Lobe", "Right Middle Lobe", "Right Lower Lobe",
        "Left Upper Lobe", "Left Lower Lobe"}

def class_of(row):
    bid = int(row.branch_instance_id)
    if bid == ROOT: return cls["Trachea"]
    if bid == RMB:  return cls["Right main bronchus"]
    if bid == LMB:  return cls["Left main bronchus"]
    if row.lobar_name in LOBE: return cls[row.lobar_name]
    return cls["Proximal, unnamed"]

gen_of = {int(r.branch_instance_id): class_of(r) for _, r in d.iterrows()}
print("root/LMB/RMB ids:", ROOT, LMB, RMB)

mask, spacing = sr.load_lps(os.path.join(D, "union3_lcc.nii.gz")); mask = mask > 0
parse, _ = sr.load_lps(os.path.join(D, f"{CASE}_parse.nii.gz"))
lung, _ = sr.load_lps(os.path.join(D, "lunglobe.nii.gz")); lung = lung > 0
inlet = np.where(lung.any(axis=(0, 1)))[0].max()
mask[:, :, inlet + 1:] = False

# keep only the connected airway, so no branch tip reads as a floating fragment
lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
if n > 1:
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    dropped = int(mask.sum() - sizes.max())
    mask = lab == sizes.argmax()
    print(f"dropped {dropped} voxels in {n-1} detached component(s)")

pad = np.ceil(3.0 / spacing).astype(int)
sl = tuple(slice(max(int(np.where(mask.any(axis=tuple(i for i in range(3) if i != a)))[0].min()) - pad[a], 0),
                 min(int(np.where(mask.any(axis=tuple(i for i in range(3) if i != a)))[0].max()) + pad[a] + 1, mask.shape[a]))
           for a in range(3))

zoom = spacing / sr.RENDER_MM
field = ndimage.gaussian_filter(
    ndimage.zoom(mask[sl].astype(np.float32), zoom, order=1, prefilter=False),
    sr.SMOOTH_VOX, output=np.float32)

# Smoothing can pinch a one-voxel-wide branch below the isolevel and leave a
# fragment floating in the projection, so the connectivity check is repeated on
# the rendered field rather than only on the source mask.
solid = field > sr.LEVEL
lab_f, n_f = ndimage.label(solid, structure=ndimage.generate_binary_structure(3, 3))
if n_f > 1:
    sz = np.bincount(lab_f.ravel()); sz[0] = 0
    field = np.where(lab_f == sz.argmax(), field, 0.0).astype(np.float32)
    print(f"render field: removed {n_f-1} pinched-off fragment(s)")

inst = np.where(mask[sl], parse[sl], 0)
labels = np.zeros(inst.shape, dtype=np.int16)
for k, v in gen_of.items():
    labels[inst == k] = v
labels = ndimage.zoom(labels, zoom, order=0, prefilter=False)
if labels.shape != field.shape:
    g = np.zeros(field.shape, dtype=np.int16)
    t = tuple(slice(0, min(a, b)) for a, b in zip(labels.shape, field.shape))
    g[t] = labels[t]; labels = g
_, near = ndimage.distance_transform_edt(labels == 0, return_indices=True)
labels = labels[tuple(near)]

palette = np.zeros((len(ORDER) + 1, 3), dtype=np.float32)
palette[0] = mc.to_rgb("#c9c8c0")   # unassigned shell
for i, c in enumerate(COLOURS, start=1):
    palette[i] = mc.to_rgb(c)
rgb, alpha, _ = sr.shade_labeled(field, labels, lambda g: palette[np.clip(g, 0, len(ORDER))])
print("render grid:", field.shape)

fig, ax = plt.subplots(figsize=(7.2, 7.4))
ax.imshow(np.dstack([rgb, alpha]), interpolation="antialiased")
ax.set_axis_off()
handles = [mpatches.Patch(facecolor=c, edgecolor="none",
                          label=n.replace(" Lobe", "").replace(" main bronchus", " main"))
           for n, c in zip(ORDER, COLOURS)]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10),
          ncol=3, frameon=False, fontsize=10.5, handlelength=1.4, columnspacing=1.4)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.09)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print("wrote", OUT + ".pdf")
