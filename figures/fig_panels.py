#!/usr/bin/env python
"""Five-panel figure: each ensemble member, their union, and the selected component.

Every panel is the same scan on ONE shared crop and ONE isotropic grid, so what
differs on the page is the mask and nothing else. The shared crop is taken from
the union, which contains all five, and the final component is highlighted by
colour rather than by scale.
"""
import os, sys, numpy as np, nibabel as nib
sys.path.insert(0, "/path/to/airway-data/BALANCED_COMBINED_90_2026-09-15/scripts")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
import airway_surface_render as sr

D = "/path/to/airway-data/airway-seg/AGE_BAND/SUBJECT/EXAMPLE_CASE"
OUT = "/path/to/airway-data/AIRWAY_METHOD_PAPER/figures/01_member_panels"
MARGIN_MM = 4.0

files = ["airway_bin.nii.gz", "medpseg_airway.nii.gz", "tfenet_airway.nii.gz"]
# Only the letter and the volume are set inside the figure. This canvas is 15
# inches wide and lands at the text width of a proceedings page, a reduction of
# about two thirds, so a model name set here would print at four points. The
# names are in the caption, which reads each panel in order anyway.
titles = ["(a)", "(b)", "(c)", "(d)", "(e)"]

masks, spacing = [], None
for f in files:
    m, sp = sr.load_lps(os.path.join(D, f)); masks.append(m > 0); spacing = sp
final, _ = sr.load_lps(os.path.join(D, "union3_lcc.nii.gz")); final = final > 0
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
rows = [
    [(masks[0], None), (masks[1], None), (masks[2], None),
     (union, None), (final, "#1baf7a")],
    [(seeded(masks[0], seed), None), (seeded(masks[1], seed), None),
     (seeded(masks[2], seed), None), (seeded(union, seed), None),
     (final, "#1baf7a")],
]
letters = [["(a)", "(b)", "(c)", "(d)", "(e)"],
           ["(f)", "(g)", "(h)", "(i)", "(j)"]]

fig, axgrid = plt.subplots(2, 5, figsize=(15.2, 7.4))
for r, (row, lets) in enumerate(zip(rows, letters)):
    for ax, title, (mask, colour) in zip(axgrid[r], lets, row):
        final_panel = colour is not None
        sr.RENDER_MM = 0.20 if final_panel else 0.40
        rgb, alpha = sr.shade(field_of(mask, render_mm=sr.RENDER_MM,
                                       clean=final_panel or r == 1))
        if colour is not None:
            import matplotlib.colors as mc
            tint = np.array(mc.to_rgb(colour))
            lum = rgb.mean(axis=2, keepdims=True)
            rgb = np.clip(lum * tint * 1.45, 0, 1)
        ax.imshow(np.dstack([rgb, alpha]))
        ax.set_title(f"{title}  {ml(mask):.1f} mL", fontsize=24,
                     color="#1a1a19", pad=8)
        ax.set_axis_off()
fig.subplots_adjust(wspace=0.015, hspace=0.12, left=0.005, right=0.995,
                    top=0.955, bottom=0.005)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", OUT + ".pdf")
for r, row in enumerate(rows):
    print(f"  row {r+1} volumes:", [f"{ml(m):.1f}" for m, _ in row])
