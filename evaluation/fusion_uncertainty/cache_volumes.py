#!/usr/bin/env python
"""Second cache pass: everything a voxel-level experiment needs, per case, so
that sweeps over the refinement never reload the seven member volumes.

Writes STUDY/cache/<split>/<case>/volumes.npz with
  pattern   uint8, per voxel inside the fused mask: AirMorph votes (0..5)*4
            + 2*MEDPSeg + TfeNet, 255 outside the mask. Any confidence scheme
            is a lookup over these 24 patterns.
  dist      float16 distance (mm) from the fused mask's skeleton, 0 outside.
  gt        uint8 ground truth; gt_parsing int32; gt_skeleton uint8.
  airmorph  uint8 AirMorph mask (the seed), medpseg/tfenet uint8 (reference rules).
and pattern_hist.npz: TP / all counts per pattern inside the fused mask.
"""
import argparse, os, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import common, tree, refine


def run_case(args):
    split, cid = args
    d = os.path.join(common.CACHE, split, cid)
    if os.path.exists(os.path.join(d, "volumes.npz")):
        return f"{cid}: cached"
    t0 = time.time()
    c = common.load_case(split, cid)
    M = common.union_seeded(c).astype(bool)
    tab = pd.read_csv(os.path.join(d, "branches.csv"))
    bm = np.load(os.path.join(d, "branch_map.npz"))["branch_map"]
    a5 = np.zeros(M.shape, np.uint8)
    for k in common.AIRMORPH_MEMBERS:
        a5 += c[k]
    pattern = (a5 * 4 + 2 * c["medpseg"].astype(np.uint8) + c["tfenet"].astype(np.uint8)).astype(np.uint8)
    pattern[~M] = 255
    dist, _ = refine.geometry(M, bm, tab, c["spacing"])
    gt = c["gt"].astype(bool)
    pat_in = pattern[M]
    hist_all = np.bincount(pat_in, minlength=24)[:24]
    hist_tp = np.bincount(pat_in, weights=gt[M], minlength=24)[:24]
    np.savez_compressed(os.path.join(d, "volumes.npz.partial.npz"),
                        pattern=pattern, dist=dist.astype(np.float16), gt=c["gt"].astype(np.uint8),
                        gt_parsing=c["gt_parsing"].astype(np.int32), gt_skeleton=c["gt_skeleton"].astype(np.uint8),
                        airmorph=c["airmorph"].astype(np.uint8), medpseg=c["medpseg"].astype(np.uint8),
                        tfenet=c["tfenet"].astype(np.uint8), spacing=c["spacing"])
    os.replace(os.path.join(d, "volumes.npz.partial.npz"), os.path.join(d, "volumes.npz"))
    np.savez(os.path.join(d, "pattern_hist.npz"), all=hist_all, tp=hist_tp)
    return f"{cid}: {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    jobs = [(a.split, c) for c in common.cases(a.split)]
    with Pool(a.workers) as p:
        for m in p.imap_unordered(run_case, jobs):
            print(m, flush=True)


if __name__ == "__main__":
    main()
