#!/usr/bin/env python
"""Score the chosen fusion on a split with the full metric set, beside the
reference strategies, from the study cache; write its masks for the
per-generation benchmark and the branch tables with per-branch confidence.

    python apply_final.py --split test --name topofuse \
        --scheme calibrated --prune-tau 0.5 --kappa 0.5 --core-min-vox 1.5 \
        --out results/whole_test.csv

Reference strategies: MEDPSeg alone (best single model), majority vote of the
three with seeded selection, and the union with seeded selection (the current
method). The calibration table and the logistic scorer always come from the
validation split.
"""
import argparse, json, os, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import common, refine
from sweep import branch_prob, prune_ids

CFG = None


def fuse(c, a, logit, table):
    """The proposed fusion on a cached case: returns (mask, branch table)."""
    tab, bm, M = c["tab"].copy(), c["branch_map"], c["M"]
    conf = common.confidence_from_pattern(c["pattern"], a.scheme, table)
    sk = (c["dist"] == 0) & M
    ids = bm[sk]; num = len(tab) - 1
    tab["conf_skel_cal"] = np.bincount(ids, weights=conf[sk], minlength=num + 1) / np.maximum(np.bincount(ids, minlength=num + 1), 1)
    p = branch_prob("logit", tab, logit)
    tab["prob"] = p
    alive = prune_ids(tab, p, a.prune_tau, 3) if a.prune_tau > 0 else np.ones(len(tab), bool)
    tab["kept"] = alive
    Mp = (M & alive[bm]).astype(np.uint8)
    if a.kappa > 0:
        t = tab.copy()
        if a.scheme == "calibrated":
            t["conf_skel"] = tab["conf_skel_cal"]
        r = tab["r_mean_mm"].to_numpy(np.float32).copy(); r[0] = 0
        R = refine.refine(Mp, conf, bm, t, a.kappa, 0.5, None, 0.6, c["spacing"], (c["dist"], r[bm]), core_min_vox=a.core_min_vox)
    else:
        R = Mp
    # per-branch confidence of the final mask: mean calibrated probability of its voxels
    inside = R.astype(bool)
    ids = bm[inside]
    tab["conf_final_vox"] = np.bincount(ids, weights=conf[inside], minlength=num + 1) / np.maximum(np.bincount(ids, minlength=num + 1), 1)
    return R, tab, conf


def run_case(cid):
    a, logit, table = CFG
    t0 = time.time()
    c = common.load_cached(a.split, cid)
    vote = (c["airmorph"].astype(np.uint8) + c["medpseg"].astype(np.uint8) + c["tfenet"].astype(np.uint8)) >= 2
    refs = {"medpseg_raw": c["medpseg"].astype(np.uint8),
            "vote2of3_seeded": common.keep_seeded_component(vote, c["airmorph"]),
            "union_seeded": c["M"].astype(np.uint8)}
    R, tab, conf = fuse(c, a, logit, table)
    out_dir = os.path.join(common.STUDY, "masks", a.name, a.split)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"ATM26_{cid}.npz"), mask=R)
    tab.to_csv(os.path.join(out_dir, f"{cid}_branches.csv"), index=False)
    rows = []
    for name, mk in list(refs.items()) + [(a.name, R)]:
        rows.append({"split": a.split, "case": cid, "strategy": name, **common.score(mk, c)})
    print(f"{cid}: removed {int((~tab.kept).sum())}/{len(tab)-1} branches; DSC {rows[-2]['DSC']} -> {rows[-1]['DSC']}, "
          f"BND {rows[-2]['BND']} -> {rows[-1]['BND']}, Prec {rows[-2]['Precision']} -> {rows[-1]['Precision']} ({time.time()-t0:.0f}s)", flush=True)
    return rows


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--name", default="topofuse")
    ap.add_argument("--scheme", default="calibrated", choices=("model", "calibrated"))
    ap.add_argument("--prune-tau", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--core-min-vox", type=float, default=1.5)
    ap.add_argument("--logit-json", default=os.path.join(common.STUDY, "results", "logit_val.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    CFG = (a, json.load(open(a.logit_json)), common.pattern_table("val"))
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(run_case, common.cases(a.split)) for r in rs]
    df = pd.DataFrame(rows).sort_values(["strategy", "case"])
    df.to_csv(a.out, index=False)
    g = df.groupby("strategy")[["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "FPR"]].agg(["mean", "std"]).round(2)
    pd.set_option("display.width", 250)
    print(g.to_string())


if __name__ == "__main__":
    main()
