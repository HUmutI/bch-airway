#!/usr/bin/env python
"""HD, HD95 and ASSD (mm) of the refined masks, with the same surface-distance
definition as benchmark_pediatric/hausdorff.py, so the row can join Table 1
and the pediatric table of the paper.

    python hd_refined.py --split test --name topofuse --out hd_refined_test.csv
"""
import argparse, os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy import ndimage
import common

ST6 = ndimage.generate_binary_structure(3, 1)
CFG = None


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask, ST6, border_value=0)


def distances(pred, gt, spacing):
    sp, sg = surface(pred.astype(bool)), surface(gt.astype(bool))
    if not sp.any() or not sg.any():
        return {"HD": float("nan"), "HD95": float("nan"), "ASSD": float("nan")}
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d_pg, d_gp = dt_g[sp], dt_p[sg]
    both = np.concatenate([d_pg, d_gp])
    return {"HD": float(max(d_pg.max(), d_gp.max())), "HD95": float(np.percentile(both, 95)), "ASSD": float(both.mean())}


def run(cid):
    split, name = CFG
    c = common.load_cached(split, cid)
    R = np.load(os.path.join(common.STUDY, "masks", name, split, f"ATM26_{cid}.npz"))["mask"]
    rows = [{"case": cid, "strategy": "Union, refined", **distances(R, c["gt"], tuple(c["spacing"]))},
            {"case": cid, "strategy": "All three members (check)", **distances(c["M"].astype(np.uint8), c["gt"], tuple(c["spacing"]))}]
    print(cid, rows[0]["HD95"], flush=True)
    return rows


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--name", default="topofuse"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    CFG = (a.split, a.name)
    with Pool(a.workers) as p:
        rows = [r for rs in p.imap_unordered(run, common.cases(a.split)) for r in rs]
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    print(df.groupby("strategy")[["HD", "HD95", "ASSD"]].mean().round(3))


if __name__ == "__main__":
    main()
