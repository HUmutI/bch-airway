#!/usr/bin/env python
"""Rotating view of one labelled airway, for the repository front page.

The still figures show the tree from the front only, which hides how the lobes
sit relative to one another in depth. Turning it makes that readable at a
glance, and a reader who has never opened the paper can see what the pipeline
produces.

Rendered on the coarse grid on purpose. This is a web animation, not a figure,
and the fine grid would multiply the cost of every frame for detail a small
looping GIF cannot show anyway.

    python fig_spin.py            # 36 frames, 10 degrees apart
    python fig_spin.py --frames 24 --width 460
"""
import argparse, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.colors as mc
from scipy import ndimage
from PIL import Image
import airway_surface_render as sr

CASE = "EXAMPLE_CASE"
D = f"/path/to/airway-data/airway-seg/AGE_BAND/SUBJECT/{CASE}"
OUT = "/path/to/airway-data/paper/figures/06_spin"
TABLE = "/path/to/airway-data/or_all3_run/normal_airway_branch_morphometry.csv"
COHORT = "/path/to/airway-data/cohort90_results/cohort_90.csv"

ORDER = ["Trachea", "Right main bronchus", "Left main bronchus",
         "Right Upper Lobe", "Right Middle Lobe", "Right Lower Lobe",
         "Left Upper Lobe", "Left Lower Lobe", "Proximal, unnamed"]
COLOURS = ["#1a1a19", "#008300", "#8d5fd3", "#2a78d6", "#eb6834",
           "#1baf7a", "#eda100", "#e87ba4", "#b4b3aa"]
LOBE = set(ORDER[3:8])

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=36)
ap.add_argument("--width", type=int, default=520)
ap.add_argument("--render-mm", type=float, default=0.30)
args = ap.parse_args()
sr.RENDER_MM = args.render_mm
# Smoothing is held at 0.45 mm physically, as in the still figures. Scaling it
# with the grid instead pushes the sigma to a millimetre on a coarse grid, which
# pinches every distal branch below the isolevel and leaves only the trachea.
sr.SMOOTH_VOX = 0.45 / args.render_mm

mask, spacing = sr.load_lps(os.path.join(D, "union3_lcc.nii.gz")); mask = mask > 0
parse, _ = sr.load_lps(os.path.join(D, f"{CASE}_parse.nii.gz"))
lung, _ = sr.load_lps(os.path.join(D, "lunglobe.nii.gz")); lung = lung > 0
mask[:, :, int(np.where(lung.any(axis=(0, 1)))[0].max()) + 1:] = False

c = pd.read_csv(COHORT); c = c[c.case_id == CASE].iloc[0]
ROOT, LMB, RMB = int(c.root_branch_id), int(c.lmb_branch_id), int(c.rmb_branch_id)
d = pd.read_csv(TABLE); d = d[d.case_id == CASE]
cls = {n: i + 1 for i, n in enumerate(ORDER)}


def class_of(row):
    b = int(row.branch_instance_id)
    if b == ROOT: return cls["Trachea"]
    if b == RMB:  return cls["Right main bronchus"]
    if b == LMB:  return cls["Left main bronchus"]
    return cls[row.lobar_name] if row.lobar_name in LOBE else cls["Proximal, unnamed"]


gen_of = {int(r.branch_instance_id): class_of(r) for _, r in d.iterrows()}

# A cube crop, so the tree does not walk across the frame as it turns.
pad = np.ceil(4.0 / spacing).astype(int)
sl, centres = [], []
for a in range(3):
    idx = np.where(mask.any(axis=tuple(i for i in range(3) if i != a)))[0]
    lo, hi = int(idx.min()) - pad[a], int(idx.max()) + pad[a] + 1
    sl.append(slice(max(lo, 0), min(hi, mask.shape[a])))
sl = tuple(sl)

zoom = spacing / sr.RENDER_MM
field = ndimage.gaussian_filter(
    ndimage.zoom(mask[sl].astype(np.float32), zoom, order=1, prefilter=False),
    sr.SMOOTH_VOX, output=np.float32)
inst = np.where(mask[sl], parse[sl], 0)
lbl = np.zeros(inst.shape, dtype=np.int16)
for k, v in gen_of.items():
    lbl[inst == k] = v
lbl = ndimage.zoom(lbl, zoom, order=0, prefilter=False)
if lbl.shape != field.shape:
    g = np.zeros(field.shape, dtype=np.int16)
    t = tuple(slice(0, min(a, b)) for a, b in zip(lbl.shape, field.shape))
    g[t] = lbl[t]; lbl = g
_, near = ndimage.distance_transform_edt(lbl == 0, return_indices=True)
lbl = lbl[tuple(near)]

# Pad to a square footprint in the transverse plane so rotation cannot clip a
# corner of the tree out of the volume.
n = int(np.ceil(np.hypot(field.shape[0], field.shape[1])))
px, py = n - field.shape[0], n - field.shape[1]
pw = ((px // 2, px - px // 2), (py // 2, py - py // 2), (0, 0))
field = np.pad(field, pw); lbl = np.pad(lbl, pw)
print("frame grid", field.shape)

pal = np.zeros((len(ORDER) + 1, 3), dtype=np.float32)
pal[0] = mc.to_rgb("#c9c8c0")
for i, col in enumerate(COLOURS, start=1):
    pal[i] = mc.to_rgb(col)

raw = []
for k in range(args.frames):
    ang = 360.0 * k / args.frames
    f = ndimage.rotate(field, ang, axes=(0, 1), reshape=False, order=1,
                       mode="constant", cval=0.0)
    l = ndimage.rotate(lbl, ang, axes=(0, 1), reshape=False, order=0,
                       mode="constant", cval=0)
    rgb, alpha, _ = sr.shade_labeled(f, l, lambda g: pal[np.clip(g, 0, len(ORDER))])
    raw.append((rgb, alpha))
    print(f"  frame {k+1}/{args.frames} at {ang:.0f} deg", flush=True)

# Squaring the volume for rotation leaves a wide white margin, and the tree does
# not reach the same corner at every angle. One box that contains every frame
# removes the margin without letting the subject drift between frames.
acc = np.zeros_like(raw[0][1], dtype=bool)
for _, a in raw:
    acc |= a > 0.01
ri, ci = np.where(acc.any(axis=1))[0], np.where(acc.any(axis=0))[0]
m = 6
r0, r1 = max(ri.min() - m, 0), min(ri.max() + m + 1, acc.shape[0])
c0, c1 = max(ci.min() - m, 0), min(ci.max() + m + 1, acc.shape[1])
print(f"  common crop {r1-r0} x {c1-c0} from {acc.shape}")

frames = []
for rgb, alpha in raw:
    rgb, alpha = rgb[r0:r1, c0:c1], alpha[r0:r1, c0:c1]
    out = rgb * alpha[..., None] + (1 - alpha[..., None])      # white background
    im = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    im = im.resize((args.width, int(args.width * im.height / im.width)),
                   Image.LANCZOS)
    frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))

frames[0].save(f"{OUT}.gif", save_all=True, append_images=frames[1:],
               duration=90, loop=0, optimize=True, disposal=2)
print("wrote", f"{OUT}.gif", f"{os.path.getsize(OUT + '.gif')/1e6:.1f} MB")
