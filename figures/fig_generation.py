#!/usr/bin/env python
"""Two panels on how the pipeline behaves with airway depth.

Left. The paper's central asymmetry. Segmentation accuracy falls steeply with
generation while naming accuracy stays near flat, so once a branch has been
recovered it is named reliably and the periphery is a segmentation problem
rather than a labelling one. Sensitivity and branch detection are drawn beside
Dice because they fall far more slowly, which is the same point seen from the
other side: the tree structure survives deeper than voxel agreement does.

Right. Where the reference branches are. Most of them sit at the generations
where accuracy has already dropped, which is why a whole-tree mean flatters the
method and why the paper reports per generation at all.

    python fig_generation.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This canvas is 11.4 inches wide and lands at the text width of a proceedings
# page, a reduction of about two and a half. Type set at the matplotlib defaults
# would print at three or four points, so everything is scaled to survive it.
plt.rcParams.update({
    "font.size": 19, "axes.labelsize": 21, "axes.titlesize": 22,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 17,
    "axes.linewidth": 1.1, "xtick.major.width": 1.1, "ytick.major.width": 1.1,
})

B = "/path/to/airway-data/AIRWAY_METHOD_PAPER"
OUT = os.path.join(B, "figures", "04_by_generation")

seg = pd.read_csv(f"{B}/recompute_2026-09-16/per_generation_atm26.csv")
lab = pd.read_csv(f"{B}/labeling_2026-09-17/per_generation_labeling.csv")

seg = seg[seg.generation <= 10]
# Branch detection is a count ratio, so it is pooled over cases rather than
# averaged per case, which would weight a volume with three branches at that
# generation the same as one with three hundred.
g = seg.groupby("generation")
agg = pd.DataFrame({
    "DSC": g.DSC.mean(), "DSC_sd": g.DSC.std(),
    "Sens": g.Sensitivity.mean(),
    "Prec": g.Precision.mean(),
    "BND": 100 * g.branches_detected.sum() / g.branches_total.sum(),
    "branches": g.gt_branches.sum(),
})
# The labelling table pools everything past generation 7 into one bin, so that
# value is drawn as a segment spanning the generations it covers rather than as
# a point at a generation it does not correspond to.
lab_pt = lab[lab.generation != "8+"].copy()
lab_pt["generation"] = lab_pt.generation.astype(int)
lab_hi = lab[lab.generation == "8+"].iloc[0]

C = {"dsc": "#2a78d6", "sens": "#1baf7a", "prec": "#eb6834",
     "bnd": "#8d5fd3", "lob": "#1a1a19", "seg": "#e87ba4"}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 4.6),
                             gridspec_kw={"width_ratios": [1.55, 1]})

x = agg.index.values
ax.fill_between(x, agg.DSC - agg.DSC_sd, agg.DSC + agg.DSC_sd,
                color=C["dsc"], alpha=0.13, lw=0)
for k, c, m, lbl in [("DSC", C["dsc"], "o", "Dice"),
                     ("Sens", C["sens"], "s", "Sensitivity"),
                     ("Prec", C["prec"], "^", "Precision"),
                     ("BND", C["bnd"], "D", "Branch detected")]:
    ax.plot(x, agg[k], marker=m, ms=7.5, lw=3.0, color=c, label=lbl)
for k, c, m, lbl in [("lobar", C["lob"], "o", "Naming, lobar"),
                     ("segmental", C["seg"], "s", "Naming, segmental")]:
    ax.plot(lab_pt.generation, lab_pt[k], marker=m, ms=7.5, lw=3.0, ls="--",
            color=c, label=lbl)
    ax.plot([8, 10], [lab_hi[k]] * 2, lw=3.0, ls="--", color=c)
    ax.plot([lab_pt.generation.iloc[-1], 8], [lab_pt[k].iloc[-1], lab_hi[k]],
            lw=3.0, ls="--", color=c)
# Anything written in this region lands on a curve, so the pooled bin is
# explained in the caption rather than on the axes.
ax.set_xlabel("Airway generation"); ax.set_ylabel("Per cent")
ax.set_xticks(x); ax.set_ylim(60, 101); ax.grid(alpha=0.25, lw=1.0)
ax.legend(fontsize=16, ncol=2, loc="lower left", frameon=False)
ax.set_title("Segmentation falls with depth, naming does not")

bx.bar(x, agg.branches, color="#b4b3aa", edgecolor="#6f6e68", lw=1.0)
bx.set_xlabel("Airway generation"); bx.set_ylabel("Reference branches")
bx.set_xticks(x); bx.grid(axis="y", alpha=0.25, lw=0.6)
bx.set_title("Where the branches are")
for s in ("top", "right"):
    ax.spines[s].set_visible(False); bx.spines[s].set_visible(False)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=400, facecolor="white", bbox_inches="tight")
print("wrote", OUT + ".pdf")
print(agg.round(2).to_string())
print("\nlabeling:\n", lab.to_string())
