#!/usr/bin/env python
"""Vote-pattern confidence refitted with MEDPSeg-R (2026-10-18).

The paper's members are now AirMorph-R (A, five networks), MEDPSeg-R (raw output,
retrained on our split; medpseg_r/outputs/epoch035) and TfeNet (raw). The fused
mask is F = anchored component of A | M~ | T~ (Eq. 1), identical to Ege's
`union_A3` masks (asserted). For every voxel of F the vote pattern is
a*4 + 2*m + t (a = AirMorph-R votes 0..5, m/t = MEDPSeg-R / TfeNet raw).

  1. per case: pattern volume inside F (uint8, 255 outside) and the per-pattern
     voxel / true-positive histogram -> CACHE_R/<split>/<case>/
  2. the lookup table P(airway | pattern) from the 30 validation volumes
     -> confidence_lut_medpseg_r.json (same format as confidence_lut.json)
  3. scored once on test (and pediatric, reported only): pooled AUROC for FP
     voxels, FP share in the lowest-confidence 5/10/20 % of F, ECE.

    python medpseg_r_confidence.py   (airmorph env)
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

import common
from uncertainty_eval import auroc_from_hist, fp_capture

E = "/fileserver/external/quin/ege"
MR = f"{E}/medpseg_r/outputs/epoch035"
EGE_F = f"{E}/fusion_research_2026-10-14/medpseg_r_eval/medpseg_r_ep035/masks/union_A3"
CACHE_R = f"{common.STUDY}/cache_medpseg_r"
OUT = f"{common.STUDY}/results_medpseg_r_confidence"
HERE = os.path.dirname(os.path.abspath(__file__))


def mr_raw(split, cid):
    name = f"ATM26_{cid}.npz" if split in ("val", "test") else f"{cid}.npz"
    with np.load(os.path.join(MR, split, name)) as p:
        return p["raw"].astype(bool)


def run(args):
    split, cid = args
    c = common.load_case(split, cid)
    A, T = c["airmorph"].astype(bool), c["tfenet"].astype(bool)
    M = mr_raw(split, cid)
    assert M.shape == A.shape, (cid, M.shape, A.shape)
    F = common.keep_seeded_component(A | M | T, A).astype(bool)
    with np.load(os.path.join(EGE_F, split, f"{cid}.npz")) as p:
        ege = p["mask"].astype(bool)
    assert (ege == F).all(), f"{cid}: F differs from Ege's union_A3"
    votes = sum(c[k].astype(np.uint8) for k in common.AIRMORPH_MEMBERS)
    pat = np.full(F.shape, 255, np.uint8)
    pat[F] = (votes[F] * 4 + 2 * M[F] + T[F]).astype(np.uint8)
    gt = c["gt"].astype(bool)
    al = np.bincount(pat[F], minlength=24)[:24].astype(float)
    tp = np.bincount(pat[F & gt], minlength=24)[:24].astype(float)
    d = os.path.join(CACHE_R, split, cid); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "pattern_hist.npz"), all=al, tp=tp)
    np.savez_compressed(os.path.join(d, "fused.npz"), F=F.astype(np.uint8), pattern=pat)
    print(split, cid, int(F.sum()), flush=True)
    return split, cid, al, tp


def table_from(rows):
    al = sum(r[2] for r in rows); tp = sum(r[3] for r in rows)
    p = np.where(al > 0, tp / np.maximum(al, 1), np.nan)
    return {"p": p, "n": al, "tp": tp}


def evaluate(rows, table, split):
    al = sum(r[2] for r in rows); tp = sum(r[3] for r in rows); fp = al - tp
    pred = np.nan_to_num(table["p"], nan=0.5); order = np.argsort(pred)
    au = auroc_from_hist(fp[order][::-1], tp[order][::-1])
    obs = np.where(al > 0, tp / np.maximum(al, 1), np.nan); ok = al > 0
    ece = float((al[ok] / al[ok].sum() * np.abs(pred[ok] - obs[ok])).sum())
    per = [auroc_from_hist((r[2] - r[3])[order][::-1], r[3][order][::-1]) for r in rows]
    cum_all = np.cumsum(al[order]) / al.sum(); cum_fp = np.cumsum(fp[order]) / fp.sum()
    full = al[[5 * 4 + 3]].sum() / al.sum()      # pattern (5,1,1): all seven predictors agree
    return {"split": split, "n": len(rows), "AUROC_fp": round(au, 4),
            "AUROC_fp_percase_mean": round(float(np.mean(per)), 4), "ECE_points": round(100 * ece, 2),
            "mask_precision_%": round(100 * tp.sum() / al.sum(), 2),
            "all_seven_agree_%_of_mask": round(100 * full, 1),
            "all_seven_agree_precision_%": round(100 * tp[23] / max(al[23], 1), 1), **fp_capture(cum_all, cum_fp)}


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, c) for s in ("val", "test", "pediatric_v3") for c in common.cases(s)]
    with Pool(3) as pool:
        res = pool.map(run, jobs)
    by = {s: [r for r in res if r[0] == s] for s in ("val", "test", "pediatric_v3")}
    table = table_from(by["val"])
    a5 = np.arange(24) // 4; md = (np.arange(24) // 2) % 2; tf = np.arange(24) % 2
    lut = {"description": "P(voxel is airway | vote pattern), fitted on the 30 ATM'26 validation volumes inside the "
                          "anchored fused mask F = keep_seeded(A | MEDPSeg-R raw | TfeNet raw, A). "
                          "pattern = airmorph_votes*4 + 2*medpseg_r + tfenet.",
           "source": "bch-airway/evaluation/fusion_uncertainty/medpseg_r_confidence.py; MEDPSeg-R epoch 35",
           "table": [{"pattern": int(i), "airmorph_votes": int(a5[i]), "medpseg_r": int(md[i]), "tfenet": int(tf[i]),
                      "p_airway": None if np.isnan(table["p"][i]) else round(float(table["p"][i]), 4),
                      "n_val_voxels": int(table["n"][i])} for i in range(24)]}
    json.dump(lut, open(os.path.join(HERE, "confidence_lut_medpseg_r.json"), "w"), indent=1)
    summ = pd.DataFrame([evaluate(by[s], table, s) for s in ("val", "test", "pediatric_v3")])
    summ.to_csv(os.path.join(OUT, "confidence_summary.csv"), index=False)
    rel = pd.DataFrame({"airmorph_votes": a5, "medpseg_r": md, "tfenet": tf, "val_voxels": table["n"].astype(int),
                        "val_precision_%": (100 * table["p"]).round(1)})
    for s in ("test", "pediatric_v3"):
        al = sum(r[2] for r in by[s]); tp = sum(r[3] for r in by[s])
        rel[f"{s}_voxels"] = al.astype(int); rel[f"{s}_precision_%"] = np.round(100 * tp / np.maximum(al, 1), 1)
    rel.to_csv(os.path.join(OUT, "pattern_table.csv"), index=False)
    print(summ.to_string(index=False))
    print(rel[rel.val_voxels > 0].to_string(index=False))


if __name__ == "__main__":
    main()
