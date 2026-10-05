#!/usr/bin/env python
"""Reliability of the vote-pattern confidence: the probability predicted from
the validation split against the observed precision on another split, one
point per vote pattern (area = voxel count), plus the share of all
false-positive voxels captured by the least-confident part of the mask.

    python fig_reliability.py --rel results/unc_test/reliability_test_calibrated.csv \
        --rel-ped results/unc_pediatric/reliability_pediatric_calibrated.csv --out fig_reliability.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", required=True); ap.add_argument("--rel-ped", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper", action="store_true", help="no in-figure titles (the caption carries them)")
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6 if a.paper else 4.1))
    for ax, path, title in ((axes[0], a.rel, "ATM'22 test (30 adult volumes)"),
                            (axes[1], a.rel_ped, "pediatric (9 expert-annotated scans)")):
        r = pd.read_csv(path)
        w = r.voxels / r.voxels.sum()
        ax.plot([0, 100], [0, 100], "k--", lw=0.8)
        ax.scatter(r["predicted_%"], r["observed_%"], s=20 + 2500 * w, alpha=0.6, c="#1f77b4", edgecolor="k", lw=0.5)
        for _, q in r.iterrows():
            lab = f"A{int(q.airmorph_votes)}{'M' if q.medpseg else ''}{'T' if q.tfenet else ''}"
            ax.annotate(lab, (q["predicted_%"], q["observed_%"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ece = float((w * (r["predicted_%"] - r["observed_%"]).abs()).sum())
        ax.set_title(f"{title}\nexpected calibration error {ece:.1f} points", fontsize=9)
        ax.set_xlabel("predicted P(airway), from validation (%)"); ax.set_ylabel("observed precision on this split (%)")
        ax.set_xlim(0, 102); ax.set_ylim(0, 102); ax.grid(alpha=0.3)
    if not a.paper:
        fig.suptitle("Calibration of the vote-pattern confidence (A = AirMorph votes of 5, M = MEDPSeg, T = TfeNet; area = voxel share)", fontsize=9)
    fig.tight_layout(); fig.savefig(a.out, dpi=160); print("wrote", a.out)


if __name__ == "__main__":
    main()
