#!/usr/bin/env python
"""How well does the training-free confidence map predict error?

Voxel level, inside the fused mask: for every confidence value the fraction of
voxels that are ground-truth airway (reliability), the AUROC of (1 - confidence)
as a detector of false-positive voxels, and the share of all false-positive
voxels that fall in the least-confident 5/10/20 % of the mask. The histograms
come from extract.py so this never touches the volumes.

Branch level: AUROC of the branch confidence (and of the fitted logistic
probability) as a detector of false branches, and the false-branch rate per
confidence band.

    python uncertainty_eval.py --splits val test pediatric --out-dir results/
"""
import argparse, json, os

import numpy as np
import pandas as pd

import common
from sweep import load_split, apply_logit, truth


def auroc_from_hist(pos, neg):
    """AUROC of a score where higher bins mean 'more positive'; pos/neg are
    histograms over the same bins (ties count half)."""
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    # positives ranked above negatives: for each bin, negatives in lower bins + half ties
    cum_neg = np.cumsum(neg) - neg
    return float(((cum_neg + 0.5 * neg) * pos).sum() / (P * N))


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    # average ranks for ties
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    P = y.sum(); N = len(y) - P
    return float((ranks[y].sum() - P * (P + 1) / 2) / (P * N))


def fp_capture(cum_all, cum_fp):
    """Share of all FP voxels inside the least-confident k% of the mask, with
    linear interpolation inside the confidence bin that straddles k%."""
    out = {}
    for k in (0.05, 0.10, 0.20):
        v = float(np.interp(k, np.r_[0.0, cum_all], np.r_[0.0, cum_fp]))
        out[f"FP_in_lowest_{int(k*100)}%"] = round(100 * v, 1)
    return out


def voxel_table(data, scheme):
    tp = sum(cnt[f"{scheme}_tp"] for _, cnt, _ in data.values())
    al = sum(cnt[f"{scheme}_all"] for _, cnt, _ in data.values())
    fp = al - tp
    conf = np.arange(106) / 105.0
    nz = al > 0
    rel = pd.DataFrame({"confidence": conf[nz].round(3), "voxels": al[nz].astype(int),
                        "share_of_mask_%": (100 * al[nz] / al.sum()).round(2),
                        "precision_%": (100 * tp[nz] / al[nz]).round(2),
                        "share_of_all_FP_%": (100 * fp[nz] / fp.sum()).round(2)})
    # FP detector: score = 1 - conf, i.e. low confidence bins are "positive"
    au = auroc_from_hist(fp[::-1], tp[::-1])
    # FP captured by the least confident k% of the mask
    order = np.argsort(conf)         # ascending confidence
    cum_all = np.cumsum(al[order]) / al.sum(); cum_fp = np.cumsum(fp[order]) / fp.sum()
    capture = fp_capture(cum_all, cum_fp)
    summary = {"scheme": scheme, "AUROC_fp": round(au, 4), "mask_precision_%": round(100 * tp.sum() / al.sum(), 2),
               "fraction_mask_fullagree_%": round(100 * al[-1] / al.sum(), 1),
               "precision_fullagree_%": round(100 * tp[-1] / max(al[-1], 1), 2), **capture}
    return rel, summary


def per_case_auroc(data, scheme):
    out = []
    for cid, (_, cnt, _) in data.items():
        tp = cnt[f"{scheme}_tp"]; al = cnt[f"{scheme}_all"]; fp = al - tp
        out.append({"case": cid, "AUROC_fp": auroc_from_hist(fp[::-1], tp[::-1]),
                    "FP_voxels": int(fp.sum()), "mask_voxels": int(al.sum())})
    return pd.DataFrame(out)


def branch_table(data, model):
    rows, per_case = [], []
    for cid, (tab, _, _) in data.items():
        t = tab.iloc[1:]
        y = ~truth(t)                           # false branch
        for name, s in (("conf_skel", t.conf_skel.to_numpy()), ("conf_vox", t.conf_vox.to_numpy()),
                        ("logit", apply_logit(model, t))):
            per_case.append({"case": cid, "scorer": name, "AUROC_false_branch": auroc(1 - s, y),
                             "n_branches": len(t), "n_false": int(y.sum())})
        rows.append(pd.DataFrame({"case": cid, "conf_skel": t.conf_skel, "logit": apply_logit(model, t),
                                  "n_models": t.n_models, "generation": t.generation, "false": y,
                                  "n_vox": t.n_vox}))
    allb = pd.concat(rows)
    bands = pd.cut(allb.conf_skel, [-0.01, 0.34, 0.5, 0.67, 0.9, 0.999, 1.0],
                   labels=["<=1/3", "1/3-1/2", "1/2-2/3", "2/3-0.9", "0.9-<1", "1 (all agree)"])
    allb = allb.reset_index(drop=True)
    allb["fp_vox"] = np.where(allb["false"], allb.n_vox, 0.0)
    band = allb.groupby(bands.reset_index(drop=True), observed=False).agg(
        branches=("false", "size"), false_branches=("false", "sum"),
        false_rate_pct=("false", lambda s: 100 * s.mean()), fp_voxels=("fp_vox", "sum")).round(1)
    bym = allb.groupby("n_models").agg(branches=("false", "size"), false_rate_pct=("false", lambda s: 100 * s.mean())).round(1)
    pooled = {s: round(auroc(1 - allb[s].to_numpy(), allb.false.to_numpy()), 4) for s in ("conf_skel", "logit")}
    return band, bym, pd.DataFrame(per_case), pooled


def pattern_eval(split, table):
    """Calibrated scheme: order the 24 vote patterns by their validation
    precision and score that ordering on this split; also the reliability of
    the validation probabilities on this split (predicted vs observed)."""
    al = np.zeros(24); tp = np.zeros(24); per = []
    for cid in common.cases(split):
        f = os.path.join(common.CACHE, split, cid, "pattern_hist.npz")
        if not os.path.exists(f):
            continue
        h = np.load(f); al += h["all"]; tp += h["tp"]
        order = np.argsort(np.nan_to_num(table["p"], nan=0.5))
        fp = h["all"] - h["tp"]
        per.append(auroc_from_hist(fp[order][::-1], h["tp"][order][::-1]))
    order = np.argsort(np.nan_to_num(table["p"], nan=0.5))
    fp = al - tp
    au = auroc_from_hist(fp[order][::-1], tp[order][::-1])
    pred = np.nan_to_num(table["p"], nan=0.5)
    obs = np.where(al > 0, tp / np.maximum(al, 1), np.nan)
    ok = al > 0
    ece = float((al[ok] / al[ok].sum() * np.abs(pred[ok] - obs[ok])).sum())
    a5 = np.arange(24) // 4; md = (np.arange(24) // 2) % 2; tf = np.arange(24) % 2
    rel = pd.DataFrame({"airmorph_votes": a5, "medpseg": md, "tfenet": tf, "voxels": al.astype(int),
                        "predicted_%": (100 * pred).round(1), "observed_%": (100 * obs).round(1)})[ok]
    # FP captured by the least confident k% of the mask under the calibrated ordering
    cum_all = np.cumsum(al[order]) / al.sum(); cum_fp = np.cumsum(fp[order]) / fp.sum()
    capture = fp_capture(cum_all, cum_fp)
    summ = {"scheme": "calibrated", "split": split, "AUROC_fp": round(au, 4), "ECE": round(100 * ece, 2),
            "AUROC_fp_percase_mean": round(float(np.mean(per)), 4), "AUROC_fp_percase_sd": round(float(np.std(per, ddof=1)), 4),
            "mask_precision_%": round(100 * tp.sum() / al.sum(), 2), **capture}
    return rel.sort_values("predicted_%"), summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "pediatric"])
    ap.add_argument("--logit-json", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    model = json.load(open(a.logit_json))
    table = common.pattern_table("val")
    summ, lines = [], []
    for split in a.splits:
        data = load_split(split)
        if not data:
            continue
        lines.append(f"\n## {split} ({len(data)} cases)\n")
        for scheme in ("model", "seven"):
            rel, s = voxel_table(data, scheme)
            rel.to_csv(os.path.join(a.out_dir, f"reliability_{split}_{scheme}.csv"), index=False)
            pc = per_case_auroc(data, scheme)
            s.update({"split": split, "AUROC_fp_percase_mean": round(pc.AUROC_fp.mean(), 4),
                      "AUROC_fp_percase_sd": round(pc.AUROC_fp.std(), 4)})
            summ.append(s)
            lines.append(f"voxel [{scheme}]: " + ", ".join(f"{k}={v}" for k, v in s.items() if k not in ("split", "scheme")))
            if scheme == "model":
                lines.append("\nreliability (model scheme):\n" + rel.to_string(index=False))
        if any(os.path.exists(os.path.join(common.CACHE, split, cid, "pattern_hist.npz")) for cid in data):
            rel, sc = pattern_eval(split, table)
            rel.to_csv(os.path.join(a.out_dir, f"reliability_{split}_calibrated.csv"), index=False)
            summ.append(sc)
            lines.append("voxel [calibrated, table from val]: " + ", ".join(f"{k}={v}" for k, v in sc.items() if k not in ("split", "scheme")))
            lines.append("\nreliability of validation probabilities on this split:\n" + rel.to_string(index=False))
        band, bym, pc, pooled = branch_table(data, model)
        band.to_csv(os.path.join(a.out_dir, f"branch_bands_{split}.csv"))
        pc.to_csv(os.path.join(a.out_dir, f"branch_auroc_percase_{split}.csv"), index=False)
        lines.append("\nbranch false-rate by confidence band:\n" + band.to_string())
        lines.append("\nbranch false-rate by number of supporting models:\n" + bym.to_string())
        lines.append(f"\nbranch AUROC pooled: {pooled}; per-case mean: "
                     + str(pc.groupby('scorer').AUROC_false_branch.mean().round(4).to_dict()))
    pd.DataFrame(summ).to_csv(os.path.join(a.out_dir, "voxel_summary.csv"), index=False)
    txt = "\n".join(lines)
    open(os.path.join(a.out_dir, "uncertainty_report.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
