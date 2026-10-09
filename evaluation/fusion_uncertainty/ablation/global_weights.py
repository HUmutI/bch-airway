"""Ablation 8: heterogeneous ensemble with optimised global member weights and
member subsets (method dir `global_weights`).

Each voxel carries a 7-bit vote code (bit i = member common.SEVEN[i]). Every
fusion rule considered here is a function of that code, so the whole study
reduces to (1) one pass that caches the code volume and per-code histograms per
case and (2) cheap look-up-table (LUT) sweeps.

    extract  <split>          cache code volumes + histograms (3 workers)
    sweep                     (a) weight grid, (b) subsets, (c) logistic
                              regression, (d) member reliability — on val
                              histograms (voxel-level proxy, no component rule)
    exact    <split> <tag>    exact metrics (seeded component, no clDice) for
                              the candidate LUTs in <outdir>/candidates_<tag>.json
    final                     evaluate the val-chosen LUT on test + pediatric
                              once (with clDice), save test masks, write RESULTS.md

All selection happens on val. Run with the airmorph env python.
"""
import json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402

OUT = os.path.join(common.STUDY, "ablation", "global_weights")
CACHE = os.path.join(OUT, "cache")
SEVEN = common.SEVEN
NB = len(SEVEN)                       # 7 members, 128 codes
CODES = np.arange(1 << NB)
BITS = ((CODES[:, None] >> np.arange(NB)[None, :]) & 1).astype(np.float64)  # (128, 7)
METRICS = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "FPR"]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "logs", "log.txt"), "a") as fh:
        fh.write(line + "\n")


# ----------------------------------------------------------------- pass 1: extraction

def extract_one(args):
    split, cid = args
    f = os.path.join(CACHE, split, f"{cid}.npz")
    if os.path.exists(f):
        return cid, "cached"
    t0 = time.time()
    c = common.load_case(split, cid)
    code = np.zeros(c["gt"].shape, np.uint8)
    for i, k in enumerate(SEVEN):
        code |= (c[k].astype(np.uint8) << i)
    union7 = code > 0
    dil = ndimage.binary_dilation(union7, ndimage.generate_binary_structure(3, 1))
    gt = c["gt"].astype(bool)
    # per-code histograms inside the dilated union (code 0 = the 1-voxel rim)
    hist_all = np.bincount(code[dil], minlength=128).astype(np.int64)
    hist_tp = np.bincount(code[dil & gt], minlength=128).astype(np.int64)
    skel = c["gt_skeleton"].astype(bool)
    skel_all = np.bincount(code[skel], minlength=128).astype(np.int64)
    lb = (c["gt_skeleton"] * c["gt_parsing"]).astype(np.int64)
    nz = lb > 0
    nbr = int(lb.max())
    branch_code = np.bincount((lb[nz] - 1) * 128 + code[nz], minlength=nbr * 128).reshape(nbr, 128)
    # how AirMorph's own mask relates to its five members
    a5 = sum(c[k].astype(np.int8) for k in common.AIRMORPH_MEMBERS)
    am = c["airmorph"]
    rel = {"n_airmorph": int(am.sum()), "n_maj5": int((a5 >= 3).sum()), "n_union5": int((a5 >= 1).sum()),
           "n_am_and_maj5": int((am & (a5 >= 3)).sum()), "n_am_and_union5": int((am & (a5 >= 1)).sum()),
           "n_gt": int(gt.sum()), "n_gt_outside_dil": int((gt & ~dil).sum()),
           "n_gt_outside_union7": int((gt & ~union7).sum())}
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez_compressed(f, code=code, airmorph=am.astype(np.uint8), hist_all=hist_all, hist_tp=hist_tp,
                        skel_all=skel_all, branch_code=branch_code, rel=json.dumps(rel),
                        spacing=c["spacing"])
    return cid, f"{time.time() - t0:.0f}s"


def extract(split, workers=3):
    jobs = [(split, cid) for cid in common.cases(split)]
    log(f"extract {split}: {len(jobs)} cases, {workers} workers")
    with Pool(workers) as pool:
        for cid, msg in pool.imap_unordered(extract_one, jobs):
            log(f"  {split} {cid} {msg}")


def hist_one(args):
    """Histograms restricted to R = seeded component of the union of all 7
    (every LUT mask's seeded component lies inside R), a far better proxy
    region than the dilated union, whose far-away false-positive blobs the
    component rule removes anyway."""
    split, cid = args
    f = os.path.join(CACHE, split, f"{cid}_histR.npz")
    if os.path.exists(f):
        return cid, "cached"
    t0 = time.time()
    c, code, am = _load_exact(split, cid)
    R = common.keep_seeded_component(code > 0, am).astype(bool)
    gt = c["gt"].astype(bool)
    hist_all = np.bincount(code[R], minlength=128).astype(np.int64)
    hist_tp = np.bincount(code[R & gt], minlength=128).astype(np.int64)
    skel = c["gt_skeleton"].astype(bool)
    skel_all = np.bincount(code[skel & R], minlength=128).astype(np.int64)
    lb = (c["gt_skeleton"] * c["gt_parsing"]).astype(np.int64)
    nbr = int(lb.max()); nz = lb > 0
    branch_total = np.bincount(lb[nz] - 1, minlength=nbr)
    nzR = nz & R
    branch_code = np.bincount((lb[nzR] - 1) * 128 + code[nzR], minlength=nbr * 128).reshape(nbr, 128)
    np.savez_compressed(f, hist_all=hist_all, hist_tp=hist_tp, skel_all=skel_all, branch_code=branch_code,
                        branch_total=branch_total, n_gt=int(gt.sum()), n_gt_outside=int((gt & ~R).sum()),
                        skel_total=int(skel.sum()), n_R=int(R.sum()))
    return cid, f"{time.time() - t0:.0f}s"


def hist(split, workers=3):
    jobs = [(split, cid) for cid in common.cases(split)]
    log(f"hist {split}: {len(jobs)} cases, {workers} workers")
    with Pool(workers) as pool:
        for cid, msg in pool.imap_unordered(hist_one, jobs):
            log(f"  {split} {cid} {msg}")


def load_hists(split, region="R"):
    """region 'R': seeded component of the 7-union (used for the sweeps);
    region 'dil': union dilated by 1 voxel (used for the logistic regression)."""
    H = {}
    for cid in common.cases(split):
        if region == "R":
            with np.load(os.path.join(CACHE, split, f"{cid}_histR.npz")) as p:
                H[cid] = {k: p[k] for k in ("hist_all", "hist_tp", "skel_all", "branch_code", "branch_total")}
                H[cid]["n_gt_outside"] = int(p["n_gt_outside"]); H[cid]["skel_total"] = int(p["skel_total"])
        else:
            with np.load(os.path.join(CACHE, split, f"{cid}.npz")) as p:
                H[cid] = {k: p[k] for k in ("hist_all", "hist_tp", "skel_all", "branch_code")}
                rel = json.loads(str(p["rel"]))
                H[cid]["n_gt_outside"] = rel["n_gt_outside_dil"]; H[cid]["skel_total"] = int(p["skel_all"].sum())
                H[cid]["branch_total"] = p["branch_code"].sum(1)
                H[cid]["rel"] = rel
    return H


# ----------------------------------------------------------------- proxy metrics from histograms

def proxy_scores(lut, H):
    """Voxel-level DSC/Sens/Prec, TLD and BND of a 128-entry boolean LUT, per
    case, before the seeded-component rule (which only removes voxels)."""
    sel = lut.astype(bool)
    rows = []
    for cid, h in H.items():
        tp = h["hist_tp"][sel].sum(); pp = h["hist_all"][sel].sum(); gt = h["hist_tp"].sum() + h["n_gt_outside"]
        dsc = 200.0 * tp / max(pp + gt, 1)
        tld = 100.0 * h["skel_all"][sel].sum() / max(h["skel_total"], 1)
        bc = h["branch_code"]
        ratio = bc[:, sel].sum(1) / np.maximum(h["branch_total"], 1)
        bnd = 100.0 * (ratio >= 0.8).mean()
        rows.append({"case": cid, "DSC": dsc, "TLD": tld, "BND": bnd,
                     "Sensitivity": 100.0 * tp / max(gt, 1), "Precision": 100.0 * tp / max(pp, 1)})
    return pd.DataFrame(rows)


def lut_from_weights(w, t):
    return (BITS @ np.asarray(w, float) >= t - 1e-9)


def lut_key(lut):
    return "".join("1" if v else "0" for v in lut)


def compositions(total, parts):
    """All non-negative integer vectors of length `parts` summing to `total`."""
    if parts == 1:
        yield (total,); return
    for i in range(total + 1):
        for rest in compositions(total - i, parts - 1):
            yield (i,) + rest


# ----------------------------------------------------------------- pass 2: sweep on val

def reliability(H, Hd):
    rows = []
    for region, HH in (("dil", Hd), ("R", H)):
        al = sum(h["hist_all"] for h in HH.values()).astype(float); tp = sum(h["hist_tp"] for h in HH.values()).astype(float)
        gt_total = sum(h["hist_tp"].sum() + h["n_gt_outside"] for h in HH.values())
        skel_total = sum(h["skel_total"] for h in HH.values())
        sk = sum(h["skel_all"] for h in HH.values()).astype(float)
        for i, m in enumerate(SEVEN):
            only = (BITS[:, i] == 1) & (BITS.sum(1) == 1)
            has = BITS[:, i] == 1
            others = (BITS.sum(1) - BITS[:, i]) >= 1          # union of the other six
            rows.append({"region": region, "member": m,
                         "n_marked": al[has].sum(), "precision_all": 100 * tp[has].sum() / al[has].sum(),
                         "sensitivity_all": 100 * tp[has].sum() / gt_total,
                         "n_only": al[only].sum(), "precision_only": 100 * tp[only].sum() / max(al[only].sum(), 1),
                         "gt_only_recovered": tp[only].sum(), "sens_contrib_only_pct": 100 * tp[only].sum() / gt_total,
                         "skel_only_pct": 100 * sk[only].sum() / skel_total,
                         "fp_only": al[only].sum() - tp[only].sum(),
                         "precision_union_of_others": 100 * tp[others].sum() / al[others].sum(),
                         "sensitivity_union_of_others": 100 * tp[others].sum() / gt_total})
    return pd.DataFrame(rows)


def sweep():
    H = load_hists("val")
    summary = {}

    def add(name, lut, meta):
        k = lut_key(lut)
        if k in summary:
            summary[k]["names"].append(name); return
        df = proxy_scores(lut, H)
        m = df[["DSC", "TLD", "BND", "Sensitivity", "Precision"]].mean().to_dict()
        summary[k] = {"names": [name], "lut": k, **m, **meta}

    # reference rules on the 3-model structure: union / majority of (A5-majority?) -- we use
    # the plain 7-member rules; the paper's union uses AirMorph's own mask, evaluated exactly later.
    # (b) every subset, union / majority / all
    log("sweep (b): subsets")
    for s in range(1, 1 << NB):
        members = [i for i in range(NB) if s >> i & 1]
        w = np.zeros(NB); w[members] = 1
        votes = BITS @ w
        add(f"union[{s}]", votes >= 1, {"family": "subset_union", "subset": s, "k": len(members)})
        add(f"majority[{s}]", votes >= (len(members) // 2 + 1), {"family": "subset_majority", "subset": s, "k": len(members)})
        add(f"all[{s}]", votes >= len(members), {"family": "subset_all", "subset": s, "k": len(members)})
    # (a) weight grid steps 0.1, thresholds 0.05..0.95
    log("sweep (a): weight grid")
    n = 0
    for comp in compositions(10, NB):
        w = np.array(comp) / 10.0
        for t in np.arange(0.05, 1.0, 0.1):
            lut = lut_from_weights(w, t)
            if not lut[-1] or lut[0]:
                continue
            add(f"w={comp} t={t:.2f}", lut, {"family": "grid", "w": list(comp), "t": round(float(t), 2)})
            n += 1
    log(f"  grid: {n} (w,t) pairs, {len(summary)} distinct LUTs so far")
    # (c) logistic regression of GT on the 7 votes. Two voxel populations, both
    # grouped by code (exact, no subsampling needed): the union dilated by 1
    # voxel (as specified) and the seeded 7-union component R.
    log("sweep (c): logistic regression")
    Hd = load_hists("val", "dil")
    X = np.hstack([BITS, np.ones((128, 1))])
    lr_out = {}
    for region, HH in (("dil", Hd), ("R", H)):
        al = sum(h["hist_all"] for h in HH.values()).astype(float); tp = sum(h["hist_tp"] for h in HH.values()).astype(float)

        def nll(beta):
            z = X @ beta
            return float(np.sum(al * np.logaddexp(0, z) - tp * z)) + 1e-3 * np.sum(beta[:-1] ** 2)

        def grad(beta):
            z = X @ beta
            p = 1 / (1 + np.exp(-z))
            g = X.T @ (al * p - tp); g[:-1] += 2e-3 * beta[:-1]
            return g
        res = optimize.minimize(nll, np.zeros(NB + 1), jac=grad, method="L-BFGS-B")
        beta = res.x
        prob = 1 / (1 + np.exp(-(X @ beta)))
        wn = beta[:-1] / beta[:-1].sum()
        lr_out[region] = {"coef": dict(zip(SEVEN, beta[:-1].round(4).tolist())), "intercept": round(float(beta[-1]), 4),
                          "coef_normalised": dict(zip(SEVEN, wn.round(4).tolist())),
                          "n_voxels": float(al.sum()), "n_gt_voxels": float(tp.sum()), "converged": bool(res.success)}
        for thr in sorted(set(np.round(prob, 4))):
            add(f"logreg_{region} p>={thr:.4f}", prob >= thr - 1e-9, {"family": f"logreg_{region}", "t": float(thr)})
        with np.errstate(invalid="ignore", divide="ignore"):
            tab = pd.DataFrame({"code": CODES, **{m: BITS[:, i].astype(int) for i, m in enumerate(SEVEN)},
                                "n": al, "tp": tp, "p_gt": tp / np.maximum(al, 1), "p_logreg": prob})
        tab.to_csv(os.path.join(OUT, f"code_table_val_{region}.csv"), index=False)
    with open(os.path.join(OUT, "logreg_val.json"), "w") as fh:
        json.dump(lr_out, fh, indent=1)
    log(json.dumps(lr_out, indent=1))
    # (d) per-member reliability, on both voxel populations
    log("sweep (d): member reliability")
    rel = reliability(H, Hd)
    rel.to_csv(os.path.join(OUT, "member_reliability_val.csv"), index=False)
    log("\n" + rel.round(3).to_string())
    # the 3-model views of the 7 (AirMorph counted as majority of its 5) for reference
    S = pd.DataFrame(list(summary.values()))
    S["name"] = S["names"].str[0]; S["n_names"] = S["names"].str.len()
    S = S.drop(columns="names")
    S.to_csv(os.path.join(OUT, "sweep_val_proxy.csv"), index=False)
    log(f"sweep: {len(S)} distinct LUTs evaluated by proxy")
    return S


# ----------------------------------------------------------------- candidate selection

def reference_luts():
    a5 = BITS[:, :5].sum(1); md = BITS[:, 5]; tf = BITS[:, 6]; am = a5 >= 3
    three = am.astype(int) + md + tf
    return {"ref_union3(paper)": (three >= 1), "ref_majority3": (three >= 2), "ref_all3": (three >= 3),
            "ref_airmorph(maj5)": am, "ref_medpseg": md == 1, "ref_tfenet": tf == 1,
            "ref_union7": BITS.sum(1) >= 1, "ref_majority7": BITS.sum(1) >= 4}


def select_candidates(S, bnd_margin=1.5, top=30, tag="val"):
    """Shortlist LUTs from the proxy sweep for exact evaluation: top-`top` by
    proxy DSC among those within `bnd_margin` BND points of the paper union's
    proxy BND, the best constrained LUT of every family, the best
    unconstrained LUTs, and the reference rules."""
    refs = reference_luts()
    union_key = lut_key(refs["ref_union3(paper)"])
    ub = S.loc[S.lut == union_key, "BND"]
    ub = float(ub.iloc[0]) if len(ub) else float(proxy_scores(refs["ref_union3(paper)"], load_hists("val"))["BND"].mean())
    ok = S[S.BND >= ub - bnd_margin].sort_values("DSC", ascending=False)
    chosen = {}
    for _, r in ok.head(top).iterrows():
        chosen[r.lut] = r["name"]
    for fam, g in ok.groupby("family"):
        for _, r in g.head(3).iterrows():
            chosen.setdefault(r.lut, r["name"])
    for _, r in S.sort_values("DSC", ascending=False).head(5).iterrows():
        chosen.setdefault(r.lut, r["name"])
    for nm, lut in refs.items():
        chosen[lut_key(lut)] = nm if lut_key(lut) not in chosen else chosen[lut_key(lut)] + "|" + nm
    cands = [[nm, k] for k, nm in chosen.items()]
    with open(os.path.join(OUT, f"candidates_{tag}.json"), "w") as fh:
        json.dump(cands, fh, indent=0)
    log(f"candidates_{tag}: {len(cands)} LUTs (union proxy BND {ub:.2f}, margin {bnd_margin})")
    return cands


# ----------------------------------------------------------------- exact evaluation

def _load_exact(split, cid):
    c = common.load_cached(split, cid)
    with np.load(os.path.join(CACHE, split, f"{cid}.npz")) as p:
        code = p["code"]; am = p["airmorph"].astype(bool)
    return c, code, am


def exact_one(args):
    split, cid, luts, cldice = args
    c, code, am = _load_exact(split, cid)
    # crop to the box holding every member and the GT: all metrics but FPR are unchanged
    nz = np.nonzero((code > 0) | (c["gt"] > 0))
    sl = tuple(slice(max(int(a.min()) - 1, 0), int(a.max()) + 2) for a in nz)
    c = dict(c, gt=c["gt"][sl], gt_parsing=c["gt_parsing"][sl], gt_skeleton=c["gt_skeleton"][sl])
    code = code[sl]; am = am[sl]
    out = []
    for name, key in luts:
        lut = np.array([ch == "1" for ch in key])
        t0 = time.time()
        mask = common.keep_seeded_component(lut[code], am)
        s = common.score(mask, c, cldice=cldice)
        out.append({"split": split, "case": cid, "name": name, "lut": key, "sec": time.time() - t0, **s})
    return out


def final_one(args):
    """Full-volume evaluation of one LUT with clDice; saves the mask."""
    split, cid, name, key, save = args
    c, code, am = _load_exact(split, cid)
    lut = np.array([ch == "1" for ch in key])
    t0 = time.time()
    mask = common.keep_seeded_component(lut[code], am)
    t_fuse = time.time() - t0
    s = common.score(mask, c, cldice=True)
    if save:
        d = os.path.join(OUT, "masks", split); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, f"ATM26_{cid}.npz"), mask=mask.astype(np.uint8))
    return {"split": split, "case": cid, "name": name, "lut": key, "fuse_sec": t_fuse, **s}


def final(name, key, splits=("test", "pediatric"), workers=3):
    rows = []
    for split in splits:
        jobs = [(split, cid, name, key, True) for cid in common.cases(split)]
        log(f"final {split}: {name}")
        with Pool(workers) as pool:
            for r in pool.imap_unordered(final_one, jobs):
                rows.append(r); log(f"  {split} {r['case']} DSC {r['DSC']:.2f} BND {r['BND']:.2f}")
    df = pd.DataFrame(rows)
    for split, g in df.groupby("split"):
        g.to_csv(os.path.join(OUT, f"final_per_case_{split}.csv"), index=False)
    return df


def report_tables():
    """Summary tables for RESULTS.md: means per split, paired differences vs
    union and refined (per-case rows from the study's whole_<split>.csv)."""
    rows = []; pairs = []
    for split in ("val", "test", "pediatric"):
        f = os.path.join(OUT, f"final_per_case_{split}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f).set_index("case")
        w = pd.read_csv(os.path.join(common.STUDY, "results", f"whole_{split}.csv"))
        ref = {"union": w[w.strategy == "union_seeded"].set_index("case"),
               "refined": w[w.strategy == "topofuse"].set_index("case")}
        rows.append({"split": split, "method": "global_weights (this)", **d[METRICS].mean().to_dict()})
        for nm, r in ref.items():
            rows.append({"split": split, "method": nm, **r[METRICS].mean().to_dict()})
            diff = d[METRICS] - r.loc[d.index, METRICS]
            for m in METRICS:
                pairs.append({"split": split, "vs": nm, "metric": m, "mean_diff": diff[m].mean(),
                              "sd": diff[m].std(), "n_improved": int((diff[m] > 0).sum()),
                              "n_worse": int((diff[m] < 0).sum()), "n": len(diff)})
    T = pd.DataFrame(rows); P = pd.DataFrame(pairs)
    T.to_csv(os.path.join(OUT, "summary.csv"), index=False); P.to_csv(os.path.join(OUT, "paired_diff.csv"), index=False)
    return T, P


def exact(split, tag, cldice=False, workers=3):
    with open(os.path.join(OUT, f"candidates_{tag}.json")) as fh:
        luts = [tuple(x) for x in json.load(fh)]
    jobs = [(split, cid, luts, cldice) for cid in common.cases(split)]
    log(f"exact {split} {tag}: {len(luts)} LUTs x {len(jobs)} cases")
    rows = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(exact_one, jobs):
            rows += r; log(f"  {r[0]['case']} done ({len(rows)} rows)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"exact_{split}_{tag}.csv"), index=False)
    return df


def exact_summary(df):
    g = df.groupby("name")[["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "FPR"]].mean()
    g["n"] = df.groupby("name").size()
    return g


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "extract":
        extract(sys.argv[2], workers=int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif cmd == "hist":
        hist(sys.argv[2], workers=int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif cmd == "sweep":
        S = sweep()
        select_candidates(S)
    elif cmd == "report":
        T, P = report_tables()
        print(T.round(2).to_string()); print(P.round(2).to_string())
    elif cmd == "reliability":
        rel = reliability(load_hists("val"), load_hists("val", "dil"))
        rel.to_csv(os.path.join(OUT, "member_reliability_val.csv"), index=False)
        print(rel.round(3).to_string())
    elif cmd == "select":
        S = pd.read_csv(os.path.join(OUT, "sweep_val_proxy.csv"))
        select_candidates(S, bnd_margin=float(sys.argv[2]) if len(sys.argv) > 2 else 1.5,
                          top=int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    elif cmd == "exact":
        split, tag = sys.argv[2], sys.argv[3]
        cl = len(sys.argv) > 4 and sys.argv[4] == "cldice"
        df = exact(split, tag, cldice=cl)
        print(exact_summary(df).round(2).to_string())
    elif cmd == "final":
        with open(os.path.join(OUT, "chosen.json")) as fh:
            ch = json.load(fh)
        splits = sys.argv[2:] if len(sys.argv) > 2 else ("test", "pediatric")
        df = final(ch["name"], ch["lut"], splits=splits)
        print(df.groupby("split")[METRICS].mean().round(2).to_string())
    else:
        raise SystemExit(f"unknown command {cmd}")
