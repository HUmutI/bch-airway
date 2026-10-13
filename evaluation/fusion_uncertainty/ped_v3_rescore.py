#!/usr/bin/env python
"""Re-score every stored pediatric mask against the final pediatric references.

The final, radiologist-reviewed references (2026-10-10) are the split
"pediatric_v3" (pediatric_annotations_v3_final/); the earlier ones stay the
split "pediatric". The masks are not recomputed: every rule was fitted on adult
validation volumes and never saw a pediatric reference, so only the scoring
changes. Scoring the same masks against the earlier references is a check that
the published numbers are reproduced.

    python ped_v3_rescore.py --splits pediatric pediatric_v3 --out-dir ../../../fusion_uncertainty_run/results_ped_v3
"""
import argparse, os, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import common
from hd_refined import distances

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation"))
import uncertainty_weighted as uw  # noqa: E402

ABL = os.path.join(common.STUDY, "ablation")
# strategy -> mask path pattern (relative to STUDY), {cid}
MASKS = {
    "kappa_0.4": "masks/topofuse/pediatric/ATM26_{cid}.npz",
    "generation_specific": "ablation/generation_specific/masks/pediatric/ATM26_{cid}.npz",
    "stacking": "ablation/stacking/masks/pediatric/ATM26_{cid}.npz",
    "boundary_aware": "ablation/boundary_aware/masks/pediatric/ATM26_{cid}.npz",
    "graph_score": "ablation/graph_score/masks/pediatric/ATM26_{cid}.npz",
    "multiscale": "ablation/multiscale/masks/pediatric/ATM26_{cid}.npz",
    "global_weights": "ablation/global_weights/masks/pediatric/ATM26_{cid}.npz",
    "refine_cnn": "ablation/refine_cnn/masks/pediatric/{cid}.npz",
}
HD_FOR = ("union_seeded", "generation_specific", "kappa_0.4")


def load_mask(path):
    with np.load(path) as p:
        key = "mask" if "mask" in p.files else p.files[0]
        return p[key].astype(np.uint8)


def uw_mask(c):
    cfg = {"method": "c", "t": 0.15, "sigma": 0.0}
    cc = uw.crop_case(c)
    m, _ = uw.apply_config(cc, cfg, common.pattern_table("val"))
    return uw.embed(m, cc).astype(np.uint8)


def run(args):
    split, cid = args
    c = common.load_cached(split, cid)
    masks = {"union_seeded": c["M"].astype(np.uint8)}
    for k, rel in MASKS.items():
        masks[k] = load_mask(os.path.join(common.STUDY, rel.format(cid=cid)))
    masks["uncertainty_weighted"] = uw_mask(c)
    rows = []
    for k, m in masks.items():
        assert m.shape == c["gt"].shape, (cid, k, m.shape)
        r = {"split": split, "case": cid, "strategy": k, **common.score(m, c, cldice=True)}
        if k in HD_FOR:
            r.update(distances(m, c["gt"], tuple(c["spacing"])))
        rows.append(r)
    print(split, cid, " ".join(f"{r['strategy']}={r['DSC']:.2f}" for r in rows), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["pediatric", "pediatric_v3"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(s, cid) for s in a.splits for cid in common.cases(s)]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(run, jobs) for r in rs]
    df = pd.DataFrame(rows).sort_values(["split", "strategy", "case"])
    df.to_csv(os.path.join(a.out_dir, "rescore_percase.csv"), index=False)
    cols = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "HD95"]
    summ = df.groupby(["split", "strategy"])[cols].mean().round(2)
    summ.to_csv(os.path.join(a.out_dir, "rescore_summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()
