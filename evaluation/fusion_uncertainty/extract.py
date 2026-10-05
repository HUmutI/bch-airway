#!/usr/bin/env python
"""Per case: build the branch tree of the current fusion (union, seeded
component), record per-branch evidence and ground-truth overlap, and the
voxel-confidence histograms. Everything the sweep and the uncertainty analysis
need is written to the study cache so neither has to touch the volumes again.

    python extract.py --split val [--workers 8] [--cases ...]

Writes STUDY/cache/<split>/<case>/{branch_map.npz, branches.csv, counts.npz}.
counts.npz holds, per predicted branch, the voxel count, the true-positive
count and the number of GT-centreline voxels covered, the sparse
(pred branch x GT branch) GT-centreline coverage matrix, the GT totals, and the
TP/FP histograms over confidence values inside the fused mask. With these the
sweep recomputes DSC, Sensitivity, Precision, TLD and BND of any pruning
exactly, without the volumes.
"""
import argparse, os, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import common, tree


def run_case(args):
    split, cid = args
    out = os.path.join(common.CACHE, split, cid)
    if os.path.exists(os.path.join(out, "counts.npz")):
        return f"{cid}: cached"
    t0 = time.time()
    c = common.load_case(split, cid)
    M = common.union_seeded(c)
    bm, tab = tree.build_tree(M, c["spacing"], c, gt=c["gt"])
    num = len(tab["id"]) - 1

    gt = c["gt"].astype(bool)
    flat_bm = bm.ravel()
    idx = np.flatnonzero(flat_bm)
    b = flat_bm[idx]
    n_vox = np.bincount(b, minlength=num + 1)
    n_tp = np.bincount(b, weights=gt.ravel()[idx], minlength=num + 1)
    # GT centreline: branch id per GT skeleton voxel (gt_skeleton * gt_parsing)
    gsk = c["gt_skeleton"].astype(bool)
    gb = (c["gt_parsing"] * gsk).ravel()
    gsk_idx = np.flatnonzero(gsk.ravel())
    gt_branch_of_skel = gb[gsk_idx]                     # GT branch id (0 = unparsed)
    pred_branch_of_skel = flat_bm[gsk_idx]              # predicted branch covering it (0 = none)
    n_gt_branches = int(gt_branch_of_skel.max())
    gt_branch_len = np.bincount(gt_branch_of_skel, minlength=n_gt_branches + 1)
    cov = pred_branch_of_skel > 0
    pairs = np.stack([pred_branch_of_skel[cov], gt_branch_of_skel[cov]], 1)
    uniq, cnt = np.unique(pairs, axis=0, return_counts=True)

    # voxel confidence histograms inside the fused mask
    hists = {}
    for scheme in ("model", "seven"):
        conf = common.confidence(c, scheme).ravel()[idx]
        q = np.round(conf * 105).astype(np.int64)        # 105 = lcm(15, 7): exact bins
        hists[f"{scheme}_tp"] = np.bincount(q, weights=gt.ravel()[idx], minlength=106)
        hists[f"{scheme}_all"] = np.bincount(q, minlength=106)
    # same, restricted to a 1-voxel inner rim vs the core, to see where the
    # uncertainty lives
    rim = M.astype(bool) & ~tree.ndimage.binary_erosion(M.astype(bool))
    rim_flat = rim.ravel()[idx]
    conf = common.confidence(c, "model").ravel()[idx]
    q = np.round(conf * 105).astype(np.int64)
    hists["model_rim_tp"] = np.bincount(q[rim_flat], weights=gt.ravel()[idx][rim_flat], minlength=106)
    hists["model_rim_all"] = np.bincount(q[rim_flat], minlength=106)

    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "branch_map.npz"), branch_map=bm)
    pd.DataFrame(tab).to_csv(os.path.join(out, "branches.csv"), index=False)
    np.savez(os.path.join(out, "counts.npz"),
             n_vox=n_vox, n_tp=n_tp, pair_pred=uniq[:, 0], pair_gt=uniq[:, 1], pair_cnt=cnt,
             gt_branch_len=gt_branch_len, gt_voxels=int(gt.sum()), gt_skel_voxels=int(gsk.sum()),
             n_voxels_total=int(gt.size), spacing=c["spacing"], **hists)
    base = common.score(M, c)
    pd.DataFrame([{"case": cid, "split": split, **base}]).to_csv(os.path.join(out, "baseline_score.csv"), index=False)
    return (f"{cid}: {num} branches, {int((np.asarray(tab['gt_skel_frac']) < 0.5).sum())} false, "
            f"DSC {base['DSC']} BND {base['BND']}  {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("val", "test", "pediatric"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*")
    a = ap.parse_args()
    cases = a.cases or common.cases(a.split)
    jobs = [(a.split, c) for c in cases]
    with Pool(a.workers) as p:
        for msg in p.imap_unordered(run_case, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
