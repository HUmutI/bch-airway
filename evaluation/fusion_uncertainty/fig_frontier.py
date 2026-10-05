#!/usr/bin/env python
"""DSC against branch detection for every combination rule: the 51 voxel rules
of the earlier sweep (grey) and the proposed fusion along its kappa curve
(coloured), on one split. A rule above and to the right of another is better
on both axes.

    python fig_frontier.py --split val --refine results/refine_val.csv --out fig_frontier_val.png
"""
import argparse, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--refine", required=True)
    ap.add_argument("--scheme", default="calibrated")
    ap.add_argument("--prune-tau", type=float, default=0.5)
    ap.add_argument("--core-min-vox", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    old = pd.read_csv(os.path.join(HERE, "..", "rule_sweep", f"{a.split}_sweep_full.csv"))
    og = old.groupby("model")[["DSC", "BND", "TLD", "Precision"]].mean()
    new = pd.read_csv(a.refine)
    ng = new.groupby(["scheme", "prune_tau", "kappa", "core_min_vox"])[["DSC", "BND", "TLD", "Precision"]].mean().reset_index()
    base = ng[ng.scheme == "none"].iloc[0]

    def curve(core):
        sel = ng[(ng.scheme == a.scheme) & (ng.prune_tau == a.prune_tau) & ((ng.core_min_vox == core) | (ng.kappa == 0))]
        return sel.sort_values("kappa")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, ykey, ylabel in ((axes[0], "BND", "Branches detected (%)"), (axes[1], "TLD", "Tree length detected (%)")):
        ax.scatter(og.DSC, og[ykey], s=18, c="#9a9a9a", label="51 voxel-level rules (2026-09 sweep)")
        for name, mk, col in (("or_all3", "s", "#d62728"), ("vote2of3", "D", "#1f77b4"), ("solo_medpseg", "^", "#2ca02c")):
            if name in og.index:
                ax.scatter(og.loc[name, "DSC"], og.loc[name, ykey], s=70, marker=mk, c=col, edgecolor="k", zorder=5,
                           label={"or_all3": "union + seeded (current)", "vote2of3": "majority vote", "solo_medpseg": "MEDPSeg alone"}[name])
        for core, style, lab in ((a.core_min_vox, "-o", f"proposed, core {a.core_min_vox:g} vox (kappa 0 → 0.9)"),
                                 (1.0, "--s", "proposed, core 1 vox")):
            cv = curve(core)
            ax.plot(cv.DSC, cv[ykey], style, c="#ff7f0e", ms=4, zorder=6, label=lab, alpha=1 if style == "-o" else 0.6)
            if style == "-o":
                for _, r in cv.iterrows():
                    if r.kappa > 0:
                        ax.annotate(f"{r.kappa:g}", (r.DSC, r[ykey]), textcoords="offset points", xytext=(4, -9), fontsize=7, color="#ff7f0e")
        ax.set_xlabel("DSC (%)"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
        ax.set_xlim(89.5, 95.5); ax.set_ylim(72 if ykey == "BND" else 82, 97)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"ATM'22 {a.split} split, {new.case.nunique()} volumes: accuracy against branch recovery for every fusion rule (zoomed; rules below the window omitted)", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
