#!/usr/bin/env python
"""The whole pipeline on one examination, stage by stage.

Five panels read left to right: what the scanner produced, the airway the
ensemble recovered from it, the centreline of that airway, the centreline cut
into branches, and those branches carrying anatomical names. Every panel after
the first is rendered from the same crop, on the same grid, with the same
camera and light, so the reader is looking at one object changing
representation rather than at five different pictures.

The first panel is a minimum intensity projection through the same slab. A
single slice would show one cross section of one bronchus, which says nothing
about a tree, whereas projecting the darkest value along the viewing axis lets
the airway lumen read as a dark tree against aerated lung, in the same
orientation as the renders beside it.

Rendered at 0.15 mm because the distal branches of a nine-month-old are two to
three voxels across in the source data and this figure is inspected closely.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
from scipy import ndimage
import airway_surface_render as sr

sr.RENDER_MM = 0.15
sr.SMOOTH_VOX = 3.0          # 0.45 mm physical, as in the other renders

CASE = "EXAMPLE_CASE"
D = f"/path/to/airway-data/airway-seg/AGE_BAND/SUBJECT/{CASE}"
OUT = "/path/to/airway-data/paper/figures/03_pipeline"
TABLE = "/path/to/airway-data/or_all3_run/normal_airway_branch_morphometry.csv"
COHORT = "/path/to/airway-data/cohort90_results/cohort_90.csv"

ORDER = ["Trachea", "Right main bronchus", "Left main bronchus",
         "Right Upper Lobe", "Right Middle Lobe", "Right Lower Lobe",
         "Left Upper Lobe", "Left Lower Lobe", "Proximal, unnamed"]
COLOURS = ["#1a1a19", "#008300", "#8d5fd3",
           "#2a78d6", "#eb6834", "#1baf7a",
           "#eda100", "#e87ba4", "#b4b3aa"]
LOBE = set(ORDER[3:8])

# ----------------------------------------------------------------- geometry
mask, spacing = sr.load_lps(os.path.join(D, "union3_lcc.nii.gz")); mask = mask > 0
skel, _ = sr.load_lps(os.path.join(D, "union3_lcc_skeleton.nii.gz")); skel = skel > 0
parse, _ = sr.load_lps(os.path.join(D, f"{CASE}_parse.nii.gz"))
sparse, _ = sr.load_lps(os.path.join(D, f"{CASE}_skel_parsing.nii.gz"))
ct, _ = sr.load_lps(os.path.join(D, "image.nii.gz"))
lung, _ = sr.load_lps(os.path.join(D, "lunglobe.nii.gz")); lung = lung > 0

# Clip at the top of the segmented lung, the thoracic-inlet proxy this cohort
# uses, so the panel is not stretched by however much neck the scan covered.
inlet = int(np.where(lung.any(axis=(0, 1)))[0].max())
for v in (mask, skel):
    v[:, :, inlet + 1:] = False

# keep the connected airway only, so nothing reads as a floating fragment
lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
if n > 1:
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    print(f"dropped {int(mask.sum() - sizes.max())} voxels in {n-1} detached component(s)")
    mask = lab == sizes.argmax()
skel &= mask

pad = np.ceil(3.0 / spacing).astype(int)
sl = tuple(slice(max(int(np.where(mask.any(axis=tuple(i for i in range(3) if i != a)))[0].min()) - pad[a], 0),
                 min(int(np.where(mask.any(axis=tuple(i for i in range(3) if i != a)))[0].max()) + pad[a] + 1,
                     mask.shape[a]))
           for a in range(3))
zoom = spacing / sr.RENDER_MM
print("crop", [(s.start, s.stop) for s in sl], "zoom", np.round(zoom, 3))


def to_field(binary):
    """Source-space binary to the shared render grid, smoothed to a scalar field.

    Smoothing can pinch a one-voxel branch below the isolevel and leave a speck
    floating in the projection, so connectivity is enforced again on the grid.
    """
    f = ndimage.gaussian_filter(
        ndimage.zoom(binary[sl].astype(np.float32), zoom, order=1, prefilter=False),
        sr.SMOOTH_VOX, output=np.float32)
    lb, k = ndimage.label(f > sr.LEVEL, structure=ndimage.generate_binary_structure(3, 3))
    if k > 1:
        sz = np.bincount(lb.ravel()); sz[0] = 0
        f = np.where(lb == sz.argmax(), f, 0.0).astype(np.float32)
        print(f"  removed {k-1} pinched-off fragment(s)")
    return f


def on_grid(labels_source, field_shape):
    """Carry integer labels to the render grid and fill the smoothed shell."""
    g = ndimage.zoom(labels_source[sl], zoom, order=0, prefilter=False)
    if g.shape != field_shape:
        h = np.zeros(field_shape, dtype=g.dtype)
        t = tuple(slice(0, min(a, b)) for a, b in zip(g.shape, field_shape))
        h[t] = g[t]; g = h
    _, near = ndimage.distance_transform_edt(g == 0, return_indices=True)
    return g[tuple(near)]


# --------------------------------------------------------------- the panels
# Shading and the distance transforms behind the label panels cost about forty
# minutes on this grid, and the layout usually needs several attempts, so the
# rendered images are cached. Delete the cache to force a re-render.
CACHE = OUT + "_panels.npz"
panels = []

if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    titles = list(z["titles"])
    panels = [(t, ("mip" if i == 0 else "rgba", z[f"p{i}"]))
              for i, t in enumerate(titles)]
    GRID = tuple(z["grid"])
    print("loaded cached panels from", CACHE)
else:
  # (a) minimum intensity projection over the same slab. The source is smoothed
  # first: a minimum projection takes the darkest value along every ray, so in a
  # low dose infant scan it would otherwise collect the noise floor and the lung
  # would read as speckle rather than parenchyma.
  sub = ndimage.gaussian_filter(ct[sl].astype(np.float32), 1.0)
  mip = np.flip(sub.min(axis=1).T, axis=0)
  panels.append(("Computed tomography", ("mip", mip)))

  # (b) the segmented airway
  field = to_field(mask)
  rgb, alpha = sr.shade(field)
  panels.append(("Airway segmentation", ("rgba", np.dstack([rgb, alpha]))))
  GRID = field.shape
  print("render grid", GRID)

  # (c) the centreline, thickened so that a one-voxel curve survives projection
  sk_grid = ndimage.zoom(skel[sl].astype(np.uint8), zoom, order=0, prefilter=False)
  if sk_grid.shape != GRID:
    h = np.zeros(GRID, dtype=np.uint8)
    t = tuple(slice(0, min(a, b)) for a, b in zip(sk_grid.shape, GRID))
    h[t] = sk_grid[t]; sk_grid = h
  tube = ndimage.binary_dilation(sk_grid > 0, ndimage.generate_binary_structure(3, 3),
                               iterations=3)          # ~0.45 mm radius
  tube_field = ndimage.gaussian_filter(tube.astype(np.float32), 1.6, output=np.float32)
  rgb, alpha = sr.shade(tube_field)
  panels.append(("Centreline", ("rgba", np.dstack([rgb, alpha]))))

  # (d) the centreline cut into branches, each its own colour
  sp = on_grid(sparse, GRID)
  sp = np.where(tube, sp, 0)
  ids = np.unique(sp); ids = ids[ids > 0]
  rng = np.random.default_rng(7)
  base = np.vstack([plt.get_cmap("tab20").colors, plt.get_cmap("tab20b").colors,
                  plt.get_cmap("tab20c").colors])
  pick = rng.permutation(len(base))
  pal = np.zeros((int(sp.max()) + 1, 3), dtype=np.float32)
  pal[0] = mc.to_rgb("#c9c8c0")
  for i, b in enumerate(ids):
    pal[b] = base[pick[i % len(base)]]
  rgb, alpha, _ = sr.shade_labeled(tube_field, sp, lambda g: pal[np.clip(g, 0, len(pal) - 1)])
  panels.append((f"Branch parcellation ({len(ids)})", ("rgba", np.dstack([rgb, alpha]))))

  # (e) anatomical names on the airway surface
  c = pd.read_csv(COHORT); c = c[c.case_id == CASE].iloc[0]
  ROOT, LMB, RMB = int(c.root_branch_id), int(c.lmb_branch_id), int(c.rmb_branch_id)
  d = pd.read_csv(TABLE); d = d[d.case_id == CASE]
  cls = {n: i + 1 for i, n in enumerate(ORDER)}


  def class_of(row):
    bid = int(row.branch_instance_id)
    if bid == ROOT: return cls["Trachea"]
    if bid == RMB:  return cls["Right main bronchus"]
    if bid == LMB:  return cls["Left main bronchus"]
    if row.lobar_name in LOBE: return cls[row.lobar_name]
    return cls["Proximal, unnamed"]


  gen_of = {int(r.branch_instance_id): class_of(r) for _, r in d.iterrows()}
  inst = np.where(mask, parse, 0)
  named = np.zeros(inst.shape, dtype=np.int16)
  for k, v in gen_of.items():
    named[inst == k] = v
  named = on_grid(named, GRID)
  lpal = np.zeros((len(ORDER) + 1, 3), dtype=np.float32)
  lpal[0] = mc.to_rgb("#c9c8c0")
  for i, col in enumerate(COLOURS, start=1):
    lpal[i] = mc.to_rgb(col)
  rgb, alpha, _ = sr.shade_labeled(field, named, lambda g: lpal[np.clip(g, 0, len(ORDER))])
  panels.append(("Anatomical assignment", ("rgba", np.dstack([rgb, alpha]))))

  np.savez_compressed(CACHE, titles=np.array([t for t, _ in panels]),
                      grid=np.array(GRID),
                      **{f"p{i}": img for i, (_, (_, img)) in enumerate(panels)})
  print("cached panels to", CACHE)

# ----------------------------------------------------------------- assemble
# The renders come back indexed (superior-inferior, left-right), and the shader
# projects along the anterior axis, so the image is GRID[2] rows by GRID[0]
# columns. Getting this pair the wrong way round letterboxes every panel.
h, w = GRID[2], GRID[0]                  # rows superior-inferior, columns lateral
aspect = h / w

# Two rows, not one. Five panels across the text width of a proceedings page
# leave each of them under 25 mm, which reduced this canvas to a two centimetre
# strip. Three and two gives every panel two and a half times the area, and the
# sequence still reads left to right and then down. The sixth cell carries the
# key, so nothing is squeezed underneath the panels.
PANEL_W = 4.2
fig, axgrid = plt.subplots(2, 3, figsize=(3 * PANEL_W, 2 * PANEL_W * aspect))
axes = list(axgrid.ravel())
for ax, (title, (kind, img)) in zip(axes, panels):
    if kind == "mip":
        # The projection comes off the source grid, the renders off the finer
        # render grid. Both cover the same crop, so resampling the projection to
        # the render shape puts all five panels on one geometry and stops the
        # first one from being drawn larger than the rest.
        if img.shape != (h, w):
            img = ndimage.zoom(img, (h / img.shape[0], w / img.shape[1]), order=1)
        ax.imshow(img, cmap="gray", vmin=-1000, vmax=-400, interpolation="antialiased")
    else:
        ax.imshow(img, interpolation="antialiased")
    ax.set_axis_off()

# Stage names would print at four points after the reduction, so they live in the
# caption and only the panel letters stay, sized to survive it.
for ax, letter in zip(axes, "abcde"):
    ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes, fontsize=26,
            va="top", ha="left")

key = axes[5]
key.set_axis_off()
handles = [mpatches.Patch(facecolor=col, edgecolor="none",
                          label=n.replace(" Lobe", "").replace(" main bronchus", " main"))
           for n, col in zip(ORDER, COLOURS)]
# Nine entries have to fit inside one cell. At a larger size the key grows past
# the cell and rides up into the panel above it, which is why the spacing here
# is tight rather than generous.
key.legend(handles=handles, loc="center", ncol=1, frameon=False, fontsize=15,
           handlelength=1.3, handleheight=1.3, labelspacing=0.45,
           borderaxespad=0.0)
fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004,
                    wspace=0.015, hspace=0.02)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print("wrote", OUT + ".pdf")
