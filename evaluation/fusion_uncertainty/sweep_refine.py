#!/usr/bin/env python
"""Sweep the branch-relative refinement on the cached volumes.

Configurations: confidence scheme (model / calibrated), leaf pruning (none or
the logistic scorer at tau 0.5), kappa, and the protected core radius in
voxels. Scores DSC, TLD, BND, Sensitivity, Precision per case (clDice is
computed separately for the chosen configurations by apply_final.py).

    python sweep_refine.py --split val --out results/refine_val.csv
"""
import argparse, json, os, time
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

import common, refine
from sweep import branch_prob, prune_ids

CFG = None


def run_case(cid):
    split, logit, table, grid = CFG
    t0 = time.time()
    c = common.load_cached(split, cid)
    tab, bm, M = c["tab"], c["branch_map"], c["M"]
    r = tab["r_mean_mm"].to_numpy(np.float32).copy(); r[0] = 0
    geom = (c["dist"], r[bm])
    confs = {"model": common.confidence_from_pattern(c["pattern"], "model"),
             "calibrated": common.confidence_from_pattern(c["pattern"], "calibrated", table)}
    alive_sets = {0.0: np.ones(len(tab), bool),
                  0.5: prune_ids(tab, branch_prob("logit", tab, logit), 0.5, 3)}
    rows = [{"case": cid, "scheme": "none", "prune_tau": 0.0, "kappa": 0.0, "core_min_vox": 0.0,
             **common.score(M.astype(np.uint8), c, cldice=False)}]
    for scheme, ptau, kappa, cmv in grid:
        Mp = (M & alive_sets[ptau][bm]).astype(np.uint8)
        if kappa == 0.0:
            R = Mp
        else:
            # for the calibrated scheme the branch's own centreline value must be
            # in the same units: recompute conf_skel from the calibrated map
            t = tab
            if scheme == "calibrated":
                t = tab.copy()
                sk = c["dist"] == 0
                ids = bm[sk & M]
                vals = confs[scheme][sk & M]
                num = len(tab) - 1
                t["conf_skel"] = np.bincount(ids, weights=vals, minlength=num + 1) / np.maximum(np.bincount(ids, minlength=num + 1), 1)
            R = refine.refine(Mp, confs[scheme], bm, t, kappa, 0.5, None, 0.6, c["spacing"], geom, core_min_vox=cmv)
        rows.append({"case": cid, "scheme": scheme, "prune_tau": ptau, "kappa": kappa, "core_min_vox": cmv,
                     **common.score(R, c, cldice=False)})
    print(f"{cid}: {len(rows)} rows {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logit-json", default=os.path.join(common.STUDY, "results", "logit_val.json"))
    ap.add_argument("--table-split", default="val")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    logit = json.load(open(a.logit_json))
    table = common.pattern_table(a.table_split)
    grid = [(s, pt, k, cm) for s, pt, k, cm in product(("model", "calibrated"), (0.0, 0.5),
                                                       (0.0, 0.4, 0.5, 0.6, 0.75, 0.9), (1.0, 1.5, 2.0))
            if not (k == 0.0 and (cm != 1.0 or s == "calibrated"))]
    CFG = (a.split, logit, table, grid)
    with Pool(a.workers) as p:
        rows = [r for rs in p.imap_unordered(run_case, common.cases(a.split)) for r in rs]
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    g = df.groupby(["scheme", "prune_tau", "kappa", "core_min_vox"])[["DSC", "TLD", "BND", "Sensitivity", "Precision"]].mean().round(2)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    print(g.to_string())


if __name__ == "__main__":
    main()
