#!/usr/bin/env python
"""Sweep topology-aware pruning rules on the cached branch tables.

A rule is (how a branch's probability of being airway is scored, the threshold
tau below which a leaf is cut, the generations that are never cut). Leaves are
cut repeatedly, so a false subtree disappears from its tips inward while the
tree stays connected. Metrics are recomputed exactly from the per-branch counts
written by extract.py; the no-pruning row must reproduce the stored baseline.

Scorers
  conf    the branch's centreline confidence: mean over its centreline voxels of
          (AirMorph members/5 + MEDPSeg + TfeNet)/3. Training-free.
  logit   logistic regression on branch evidence and geometry, fitted on the
          validation split (fit_logit) and applied unchanged elsewhere.
  oracle  the ground truth itself (branch is airway iff >=50% of its centreline
          lies within one voxel of the GT): the ceiling of any leaf pruning.

    python sweep.py --split val --out sweep_val.csv
    python sweep.py --split test --out sweep_test.csv --logit-from val
"""
import argparse, json, os

import numpy as np
import pandas as pd

import common

FEATURES = ["conf_skel", "conf_vox", "n_models", "skel_airmorph5", "skel_medpseg", "skel_tfenet",
            "log_n_skel", "r_mean_mm", "r_ratio_parent", "log_tubularity", "generation_c", "is_leaf"]


def load_split(split):
    out = {}
    for cid in common.cases(split):
        d = os.path.join(common.CACHE, split, cid)
        if not os.path.exists(os.path.join(d, "counts.npz")):
            continue
        tab = pd.read_csv(os.path.join(d, "branches.csv"))
        cnt = dict(np.load(os.path.join(d, "counts.npz")))
        base = pd.read_csv(os.path.join(d, "baseline_score.csv")).iloc[0].to_dict()
        out[cid] = (tab, cnt, base)
    return out


def featurize(tab):
    X = pd.DataFrame(index=tab.index)
    X["conf_skel"] = tab.conf_skel; X["conf_vox"] = tab.conf_vox
    X["n_models"] = tab.n_models
    X["skel_airmorph5"] = tab.skel_airmorph5; X["skel_medpseg"] = tab.skel_medpseg; X["skel_tfenet"] = tab.skel_tfenet
    X["log_n_skel"] = np.log1p(tab.n_skel)
    X["r_mean_mm"] = tab.r_mean_mm.clip(0, 10)
    X["r_ratio_parent"] = tab.r_ratio_parent.clip(0, 3)
    X["log_tubularity"] = np.log(tab.tubularity.clip(0.1, 50))
    X["generation_c"] = tab.generation.where(tab.generation > 0, 12).clip(1, 15)
    X["is_leaf"] = (tab.n_children == 0).astype(float)
    return X[FEATURES].to_numpy(float)


def truth(tab):
    return (tab.gt_skel_frac_dil >= 0.5).to_numpy()


def fit_logit(split_data, l2=1.0):
    """Ridge-regularised logistic regression by IRLS, features standardised."""
    X = np.concatenate([featurize(t.iloc[1:]) for t, _, _ in split_data.values()])
    y = np.concatenate([truth(t.iloc[1:]) for t, _, _ in split_data.values()]).astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    R = l2 * np.eye(Z.shape[1]); R[0, 0] = 0
    for _ in range(50):
        p = 1 / (1 + np.exp(-Z @ w))
        W = p * (1 - p) + 1e-9
        H = Z.T @ (Z * W[:, None]) + R
        g = Z.T @ (y - p) - R @ w
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "features": FEATURES,
            "n_branches": int(len(y)), "n_false": int((~y.astype(bool)).sum())}


def apply_logit(model, tab):
    X = featurize(tab)
    Z = np.c_[np.ones(len(X)), (X - np.array(model["mu"])) / np.array(model["sd"])]
    return 1 / (1 + np.exp(-Z @ np.array(model["w"])))


def branch_prob(scorer, tab, model=None):
    """Probability per branch id (index 0 = background, forced to 1)."""
    if scorer == "conf":
        p = tab.conf_skel.to_numpy(float)
    elif scorer == "conf_vox":
        p = tab.conf_vox.to_numpy(float)
    elif scorer == "logit":
        p = apply_logit(model, tab)
    elif scorer == "oracle":
        p = truth(tab).astype(float)
    else:
        raise ValueError(scorer)
    p = p.copy(); p[0] = 1.0
    return p


def prune_ids(tab, p, tau, protect_gen):
    num = len(tab) - 1
    parent = tab.parent.to_numpy(np.int64)
    gen = tab.generation.to_numpy(np.int64)
    root = tab.root.to_numpy(bool)
    nchild = np.zeros(num + 1, np.int64)
    for i in range(1, num + 1):
        if parent[i] > 0:
            nchild[parent[i]] += 1
    alive = np.ones(num + 1, bool)
    removed = []
    frontier = [i for i in range(1, num + 1) if nchild[i] == 0]
    while frontier:
        nxt = []
        for i in frontier:
            if not alive[i] or root[i] or nchild[i] > 0:
                continue
            if gen[i] != -1 and gen[i] <= protect_gen:
                continue
            if p[i] < tau:
                alive[i] = False; removed.append(i)
                if parent[i] > 0:
                    nchild[parent[i]] -= 1
                    if nchild[parent[i]] == 0:
                        nxt.append(parent[i])
        frontier = nxt
    return alive


def metrics_from_counts(alive, cnt):
    n_pred = float(cnt["n_vox"][alive].sum()); tp = float(cnt["n_tp"][alive].sum())
    gt = float(cnt["gt_voxels"]); s = 1e-5
    keep = alive[cnt["pair_pred"]]
    cov = float(cnt["pair_cnt"][keep].sum())
    per_gt = np.bincount(cnt["pair_gt"][keep], weights=cnt["pair_cnt"][keep], minlength=len(cnt["gt_branch_len"]))
    L = cnt["gt_branch_len"]
    nz = L[1:] > 0
    det = (per_gt[1:][nz] / L[1:][nz]) >= 0.8
    fp = n_pred - tp
    return {"DSC": 100 * (2 * tp + s) / (n_pred + gt + s),
            "TLD": 100 * cov / float(cnt["gt_skel_voxels"]),
            "BND": 100 * det.mean(),
            "Sensitivity": 100 * (tp + s) / (gt + s),
            "Precision": 100 * (tp + s) / (n_pred + s),
            "FPR": 100 * fp / (float(cnt["n_voxels_total"]) - gt),
            "pred_voxels": n_pred, "removed_branches": int((~alive).sum() )}


def check_baseline(data):
    """No pruning must reproduce the stored metrics (to rounding)."""
    worst = 0
    for cid, (tab, cnt, base) in data.items():
        m = metrics_from_counts(np.ones(len(tab), bool), cnt)
        for k in ("DSC", "TLD", "BND", "Sensitivity", "Precision"):
            worst = max(worst, abs(m[k] - base[k]))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logit-from", default=None, help="split to fit the logistic scorer on (default: this split)")
    ap.add_argument("--logit-json", default=None, help="write/read the fitted logistic model here")
    a = ap.parse_args()
    data = load_split(a.split)
    print(f"{a.split}: {len(data)} cases; baseline reproduction worst abs diff = {check_baseline(data):.3f}")

    if a.logit_json and os.path.exists(a.logit_json) and a.logit_from is None:
        model = json.load(open(a.logit_json))
    else:
        fit_data = data if a.logit_from in (None, a.split) else load_split(a.logit_from)
        model = fit_logit(fit_data)
        if a.logit_json:
            json.dump(model, open(a.logit_json, "w"), indent=1)
    print("logit weights (standardised):", {f: round(w, 2) for f, w in zip(["bias"] + FEATURES, model["w"])})

    rows = []
    grid = [("none", 0.0, 0)]
    for scorer in ("conf", "conf_vox", "logit", "oracle"):
        taus = [0.5] if scorer == "oracle" else list(np.round(np.arange(0.05, 0.96, 0.05), 2))
        for tau in taus:
            for pg in (0, 2, 3):
                grid.append((scorer, float(tau), pg))
    for scorer, tau, pg in grid:
        for cid, (tab, cnt, base) in data.items():
            if scorer == "none":
                alive = np.ones(len(tab), bool)
            else:
                alive = prune_ids(tab, branch_prob(scorer, tab, model), tau, pg)
            m = metrics_from_counts(alive, cnt)
            rows.append({"scorer": scorer, "tau": tau, "protect_gen": pg, "case": cid, **m})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    g = df.groupby(["scorer", "tau", "protect_gen"])[["DSC", "TLD", "BND", "Sensitivity", "Precision", "removed_branches"]].mean()
    base = g.loc[("none", 0.0, 0)]
    g["dDSC"] = g.DSC - base.DSC; g["dBND"] = g.BND - base.BND; g["dTLD"] = g.TLD - base.TLD
    pd.set_option("display.width", 200)
    print("\nbaseline (union, seeded):\n", base.round(2).to_string())
    print("\ntop by DSC with BND loss <= 1.0:")
    ok = g[g.dBND >= -1.0].sort_values("DSC", ascending=False)
    print(ok.head(25).round(2).to_string())
    print("\ntop by DSC overall:")
    print(g.sort_values("DSC", ascending=False).head(10).round(2).to_string())
    print("\noracle:\n", g.loc["oracle"].round(2).to_string())


if __name__ == "__main__":
    main()
