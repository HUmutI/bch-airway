#!/usr/bin/env python
"""DSC against branch detection for every combination rule: the 51 voxel rules
of the earlier sweep (grey), the branch-relative refinement along its kappa
curve (orange) and, with --genspec, the adopted generation-specific rule
(per-case CSV of ablation/generation_specific), on one split. A rule above and to the right of another is better
on both axes.

    python fig_frontier.py --split test --refine results/refine_test.csv \
        --genspec ablation/generation_specific/percase_test_B3_core.csv --out 07_frontier.pdf

Leaf pruning is off by default (--prune-tau 0), as in the paper tables.
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
    ap.add_argument("--prune-tau", type=float, default=0.0)
    ap.add_argument("--core-min-vox", type=float, default=1.5)
    ap.add_argument("--genspec", help="per-case CSV of the generation-specific rule on this split")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    old = pd.read_csv(os.path.join(HERE, "..", "rule_sweep", f"{a.split}_sweep_full.csv"))
    og = old.groupby("model")[["DSC", "BND", "TLD", "Precision"]].mean()
    new = pd.read_csv(a.refine)
    ng = new.groupby(["scheme", "prune_tau", "kappa", "core_min_vox"])[["DSC", "BND", "TLD", "Precision"]].mean().reset_index()
    base = ng[ng.scheme == "none"].iloc[0]
    gs = pd.read_csv(a.genspec)[["DSC", "BND", "TLD", "Precision"]].mean() if a.genspec else None

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
        for core, style, lab in ((a.core_min_vox, "-o", f"branch-relative, core {a.core_min_vox:g} vox (kappa 0 → 0.9)"),
                                 (1.0, "--s", "branch-relative, core 1 vox")):
            cv = curve(core)
            ax.plot(cv.DSC, cv[ykey], style, c="#ff7f0e", ms=4, zorder=6, label=lab, alpha=1 if style == "-o" else 0.6)
            if style == "-o":
                for _, r in cv.iterrows():
                    if abs(r.kappa - 0.4) < 1e-9:
                        ax.annotate("kappa 0.4", (r.DSC, r[ykey]), textcoords="offset points", xytext=(-34, 7), fontsize=7, color="#ff7f0e")
        if gs is not None:
            ax.scatter(gs.DSC, gs[ykey], s=150, marker="*", c="#9467bd", edgecolor="k", zorder=7,
                       label="generation-specific (adopted)")
        ax.set_xlabel("DSC (%)"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
        ax.set_xlim(89.5, 95.5); ax.set_ylim(72 if ykey == "BND" else 82, 97)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"ATM'22 {a.split} split, {new.case.nunique()} volumes: accuracy against branch recovery for every fusion rule (zoomed; rules below the window omitted)", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
