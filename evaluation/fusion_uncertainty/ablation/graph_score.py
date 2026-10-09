#!/usr/bin/env python
"""Ablation 6: topology-/graph-aware ensemble with an explicit branch score.

    score(b) = alpha * confidence(b) + beta * agreement(b)
             + gamma * connectivity(b) + delta * plausibility(b)

confidence   calibrated P(airway | vote pattern) averaged along the branch's
             centreline (table fitted on val, common.pattern_table("val")).
agreement    n_models / 3 (how many of AirMorph, MEDPSeg, TfeNet cover >= half
             of the centreline); variant: mean of the three member fractions.
connectivity 0.5 * [reached from the root] + 0.5 * d/(d+1), d = number of
             descendant branches (subtree size - 1).
plausibility prior(generation) * (1 - mean of three penalties): child wider
             than parent (r_ratio_parent > 1), blob-like (tubularity > 3),
             branch angle to the parent beyond 90 degrees. prior(generation) =
             fraction of true branches per generation on val (smoothed).

Pruning modes on the branch tree of the union (all topology preserving):
  leaf      iteratively cut leaves with score < tau (sweep.prune_ids)
  subtree   cut every branch with score < tau together with its descendants
  subtree_max  cut a branch and its descendants only when the maximum score
            in that subtree is < tau
Generations <= protect_gen are never cut. Then optionally the branch-relative
refinement of refine.py (kappa, core 1.5 voxels) on the pruned mask.

Phases (all choices on val):
  features   per case: calibrated centreline confidence and branch angle
             (needs the cached volumes once) -> <out>/features/<split>/<case>.csv
  sweep      counts-based grid over (alpha..delta, tau, mode, protect_gen) on val
  refine     volume-based grid: chosen prunings x kappa on val
  final      evaluate the chosen configuration once on test and pediatric
"""
import argparse, json, os, sys, time
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common, refine  # noqa: E402
from sweep import load_split, metrics_from_counts, prune_ids, truth  # noqa: E402

OUT = os.path.join(common.STUDY, "ablation", "graph_score")
FEAT = os.path.join(OUT, "features")
TAUS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
MODES = [("leaf", 0), ("leaf", 3), ("subtree", 3), ("subtree_max", 3)]


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "LOG.txt"), "a") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


# ------------------------------------------------------------------ features

def branch_angles(pts_mm, ids, tab):
    """Angle (degrees) between each branch's direction (proximal end -> distal
    end) and its parent's direction. nan when undefined."""
    num = len(tab) - 1
    parent = tab.parent.to_numpy(np.int64)
    order = np.argsort(ids, kind="stable")
    ids_s = ids[order]; pts_s = pts_mm[order]
    starts = np.searchsorted(ids_s, np.arange(num + 2))
    P = {i: pts_s[starts[i]:starts[i + 1]] for i in range(1, num + 1)}
    prox = {}; dirv = {}
    children = {i: [] for i in range(num + 1)}
    for i in range(1, num + 1):
        if parent[i] > 0:
            children[parent[i]].append(i)
    for i in range(1, num + 1):
        p = P[i]
        if len(p) < 2:
            continue
        if parent[i] > 0 and len(P[parent[i]]) > 0:
            d, _ = cKDTree(P[parent[i]]).query(p)
            prox[i] = p[int(np.argmin(d))]
        elif children[i]:
            att = np.concatenate([P[c] for c in children[i] if len(P[c])]) if any(len(P[c]) for c in children[i]) else None
            if att is None:
                continue
            d, _ = cKDTree(att).query(p)
            prox[i] = p[int(np.argmax(d))]     # root: proximal end is the one far from its children
        else:
            continue
        far = p[int(np.argmax(((p - prox[i]) ** 2).sum(1)))]
        v = far - prox[i]
        n = np.linalg.norm(v)
        if n > 0:
            dirv[i] = v / n
    ang = np.full(num + 1, np.nan)
    for i in range(1, num + 1):
        pi = parent[i]
        if pi > 0 and i in dirv and pi in dirv:
            ang[i] = np.degrees(np.arccos(np.clip(np.dot(dirv[i], dirv[pi]), -1, 1)))
    return ang


def features_case(args):
    split, cid = args
    f = os.path.join(FEAT, split, f"{cid}.csv")
    if os.path.exists(f):
        return f"{cid}: cached"
    t0 = time.time()
    c = common.load_cached(split, cid)
    tab, bm, M = c["tab"], c["branch_map"], c["M"]
    num = len(tab) - 1
    conf = common.confidence_from_pattern(c["pattern"], "calibrated", common.pattern_table("val"))
    sk = (c["dist"] == 0) & M
    zyx = np.argwhere(sk)
    ids = bm[sk]
    cnt = np.bincount(ids, minlength=num + 1)
    conf_cal = np.bincount(ids, weights=conf[sk], minlength=num + 1) / np.maximum(cnt, 1)
    ang = branch_angles(zyx * np.asarray(c["spacing"])[None, :], ids, tab)
    df = pd.DataFrame({"id": np.arange(num + 1), "conf_skel_cal": conf_cal, "n_skel_cal": cnt, "angle_deg": ang})
    os.makedirs(os.path.dirname(f), exist_ok=True)
    df.to_csv(f, index=False)
    return f"{cid}: {num} branches, {time.time()-t0:.0f}s"


def load_features(split, cid):
    return pd.read_csv(os.path.join(FEAT, split, f"{cid}.csv"))


# ------------------------------------------------------------------ the score

def gen_prior_from_val():
    """P(true | generation) on val, generations clipped to 1..12, smoothed with
    a Beta(2, 2) pseudo-count towards the overall rate."""
    rows = []
    for cid, (tab, _, _) in load_split("val").items():
        t = tab.iloc[1:]
        rows.append(pd.DataFrame({"g": t.generation.clip(1, 12), "y": truth(t)}))
    d = pd.concat(rows)
    overall = d.y.mean()
    pr = {}
    for g in range(1, 13):
        s = d[d.g == g]
        pr[g] = float((s.y.sum() + 4 * overall) / (len(s) + 4))
    return pr


def components(tab, feat, prior, agreement="n_models"):
    """The four terms of the score, each in [0, 1], per branch id (row 0 = background)."""
    num = len(tab) - 1
    conf = feat.conf_skel_cal.to_numpy(float)
    if agreement == "n_models":
        agree = tab.n_models.to_numpy(float) / 3.0
    else:
        agree = (tab.skel_airmorph5 + tab.skel_medpseg + tab.skel_tfenet).to_numpy(float) / 3.0
    reached = (tab.generation.to_numpy(np.int64) != -1).astype(float)
    desc = (tab.subtree.to_numpy(float) - 1).clip(0, None)
    conn = 0.5 * reached + 0.5 * desc / (desc + 1.0)
    pen_ratio = np.clip(tab.r_ratio_parent.to_numpy(float) - 1.0, 0, 1)
    pen_tub = np.clip((np.log(tab.tubularity.to_numpy(float).clip(0.1, 100)) - np.log(3.0)) / np.log(3.0), 0, 1)
    ang = feat.angle_deg.to_numpy(float)
    pen_ang = np.where(np.isnan(ang), 0.0, np.clip((ang - 90.0) / 90.0, 0, 1))
    g = tab.generation.to_numpy(np.int64)
    pg = np.array([prior[int(min(max(x, 1), 12))] for x in g])
    plaus = pg * (1.0 - (pen_ratio + pen_tub + pen_ang) / 3.0)
    C = np.stack([conf, agree, conn, plaus], 1)
    C[0] = 1.0
    return C


def score_from(C, w):
    s = C @ np.asarray(w, float)
    s[0] = 1.0
    return s


def prune_subtree(tab, s, tau, protect_gen, use_max=False):
    """Remove every allowed branch with score < tau (or, use_max: whose whole
    subtree has max score < tau) together with its descendants."""
    num = len(tab) - 1
    parent = tab.parent.to_numpy(np.int64)
    gen = tab.generation.to_numpy(np.int64)
    root = tab.root.to_numpy(bool)
    order = sorted(range(1, num + 1), key=lambda i: -gen[i])   # deepest first (gen -1 last-> treat as deep)
    smax = s.copy()
    if use_max:
        for i in order:
            if parent[i] > 0:
                smax[parent[i]] = max(smax[parent[i]], smax[i])
    crit = smax if use_max else s
    allowed = ~root & ((gen == -1) | (gen > protect_gen))
    dead = allowed & (crit < tau); dead[0] = False
    alive = ~dead
    for i in sorted(range(1, num + 1), key=lambda i: gen[i] if gen[i] > 0 else 10 ** 6):
        if parent[i] > 0 and not alive[parent[i]]:
            alive[i] = False
    alive[0] = True
    return alive


def apply_prune(tab, s, tau, mode, protect_gen):
    if mode == "leaf":
        return prune_ids(tab, s, tau, protect_gen)
    if mode == "subtree":
        return prune_subtree(tab, s, tau, protect_gen, use_max=False)
    if mode == "subtree_max":
        return prune_subtree(tab, s, tau, protect_gen, use_max=True)
    raise ValueError(mode)


def weight_grid(step=0.1):
    n = int(round(1 / step))
    ws = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                d = n - a - b - c
                ws.append((a * step, b * step, c * step, d * step))
    return [tuple(round(x, 2) for x in w) for w in ws]


# ------------------------------------------------------------------ sweep (counts)

SWEEP = None


def sweep_worker(w):
    data, feats, prior, agreement = SWEEP
    rows = []
    for cid, (tab, cnt, base) in data.items():
        tr = truth(tab)
        s = score_from(components(tab, feats[cid], prior, agreement), w)
        for (mode, pg), tau in product(MODES, TAUS):
            alive = apply_prune(tab, s, tau, mode, pg)
            m = metrics_from_counts(alive, cnt)
            rem = ~alive; rem[0] = False
            rows.append({"a": w[0], "b": w[1], "g": w[2], "d": w[3], "agreement": agreement, "mode": mode, "protect_gen": pg,
                         "tau": tau, "case": cid, **m, "false_removed": int((rem & ~tr).sum()),
                         "true_removed": int((rem & tr).sum())})
    return rows


def run_sweep(a):
    global SWEEP
    data = load_split("val")
    feats = {cid: load_features("val", cid) for cid in data}
    prior = gen_prior_from_val()
    json.dump(prior, open(os.path.join(OUT, "gen_prior_val.json"), "w"), indent=1)
    log(f"sweep: {len(data)} val cases; generation prior {json.dumps({k: round(v, 3) for k, v in prior.items()})}")
    base_rows = []
    for cid, (tab, cnt, base) in data.items():
        tr = truth(tab)
        m0 = metrics_from_counts(np.ones(len(tab), bool), cnt)
        base_rows.append({"config": "union", "case": cid, **m0, "false_removed": 0, "true_removed": 0, "n_false": int((~tr[1:]).sum())})
        o = tr.astype(float); o[0] = 1
        for mode, pg in MODES:
            alive = apply_prune(tab, o, 0.5, mode, pg)
            m = metrics_from_counts(alive, cnt); rem = ~alive; rem[0] = False
            base_rows.append({"config": f"oracle_{mode}_pg{pg}", "case": cid, **m, "false_removed": int((rem & ~tr).sum()),
                              "true_removed": int((rem & tr).sum()), "n_false": int((~tr[1:]).sum())})
        alive = tr.copy(); alive[0] = True
        m = metrics_from_counts(alive, cnt)
        base_rows.append({"config": "oracle_all_false", "case": cid, **m, "false_removed": int((~tr[1:]).sum()), "true_removed": 0,
                          "n_false": int((~tr[1:]).sum())})
    bdf = pd.DataFrame(base_rows); bdf.to_csv(os.path.join(OUT, "sweep_val_oracle_cases.csv"), index=False)
    log("val oracle ceilings (mean over cases):\n" + bdf.groupby("config")[["DSC", "TLD", "BND", "Sensitivity", "Precision", "false_removed", "true_removed", "n_false"]].mean().round(2).to_string())

    all_rows = []
    for agreement in a.agreement:
        SWEEP = (data, feats, prior, agreement)
        t0 = time.time()
        with Pool(a.workers) as pool:
            for rows in pool.imap_unordered(sweep_worker, weight_grid(), chunksize=4):
                all_rows.extend(rows)
        log(f"sweep agreement={agreement}: {len(all_rows)} rows, {time.time()-t0:.0f}s")
    df = pd.DataFrame(all_rows)
    keys = ["agreement", "mode", "protect_gen", "a", "b", "g", "d", "tau"]
    g = df.groupby(keys)[["DSC", "TLD", "BND", "Sensitivity", "Precision", "removed_branches", "false_removed", "true_removed"]].mean()
    base = bdf[bdf.config == "union"][["DSC", "TLD", "BND"]].mean()
    g["dDSC"] = g.DSC - base.DSC; g["dBND"] = g.BND - base.BND; g["dTLD"] = g.TLD - base.TLD
    g = g.reset_index()
    g.to_csv(os.path.join(OUT, "sweep_val_summary.csv"), index=False)
    # keep per-case rows only for the configurations worth revisiting (file size)
    ok = g[g.dBND >= -1.0].sort_values("DSC", ascending=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    log("top val configurations under the rule (BND loss <= 1):\n" + ok.head(30).round(3).to_string())
    log("best per mode under the rule:\n" + ok.groupby(["agreement", "mode", "protect_gen"]).head(1).round(3).to_string())
    log("best overall DSC regardless of BND:\n" + g.sort_values("DSC", ascending=False).head(10).round(3).to_string())
    # single-term scores for reference
    for name, w in (("conf only", (1, 0, 0, 0)), ("agreement only", (0, 1, 0, 0)), ("plausibility only", (0, 0, 0, 1))):
        sub = ok[(ok.a == w[0]) & (ok.b == w[1]) & (ok.g == w[2]) & (ok.d == w[3])].head(3)
        log(f"{name}:\n" + sub.round(3).to_string())
    top = ok.head(200)
    sel = df.merge(top[keys], on=keys)
    sel.to_csv(os.path.join(OUT, "sweep_val_top_cases.csv"), index=False)


# ------------------------------------------------------------------ refine (volumes)

REF = None


def refine_case(cid):
    split, configs, prior, table = REF
    t0 = time.time()
    c = common.load_cached(split, cid)
    tab, bm, M = c["tab"], c["branch_map"], c["M"]
    feat = load_features(split, cid)
    conf = common.confidence_from_pattern(c["pattern"], "calibrated", table)
    r = tab["r_mean_mm"].to_numpy(np.float32).copy(); r[0] = 0
    geom = (c["dist"], r[bm])
    t = tab.copy(); t["conf_skel"] = feat.conf_skel_cal.to_numpy(float)
    tr = truth(tab)
    rows = [{"case": cid, "config": "union", **common.score(M.astype(np.uint8), c, cldice=False)}]
    for cfg in configs:
        t1 = time.time()
        if cfg["tau"] > 0:
            s = score_from(components(tab, feat, prior, cfg["agreement"]), (cfg["a"], cfg["b"], cfg["g"], cfg["d"]))
            alive = apply_prune(tab, s, cfg["tau"], cfg["mode"], cfg["protect_gen"])
        else:
            alive = np.ones(len(tab), bool)
        Mp = (M & alive[bm]).astype(np.uint8)
        t_prune = time.time() - t1
        if cfg["kappa"] > 0:
            R = refine.refine(Mp, conf, bm, t, cfg["kappa"], 0.5, None, 0.6, c["spacing"], geom, core_min_vox=1.5)
        else:
            R = common.keep_seeded_component(Mp, c["airmorph"]) if cfg["tau"] > 0 else Mp
        t_ref = time.time() - t1 - t_prune
        rem = ~alive; rem[0] = False
        rows.append({"case": cid, "config": cfg["name"], **common.score(R, c, cldice=cfg.get("cldice", False)),
                     "removed_branches": int(rem.sum()), "false_removed": int((rem & ~tr).sum()),
                     "true_removed": int((rem & tr).sum()), "t_prune_s": t_prune, "t_refine_s": t_ref})
        if cfg.get("save"):
            d = os.path.join(OUT, "masks", split); os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"ATM26_{cid}.npz"), mask=R.astype(np.uint8))
    print(f"{cid}: {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    return rows


def run_refine(a):
    global REF
    cfgs = json.load(open(a.configs))
    prior = json.load(open(os.path.join(OUT, "gen_prior_val.json"))); prior = {int(k): v for k, v in prior.items()}
    REF = (a.split, cfgs, prior, common.pattern_table("val"))
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(refine_case, common.cases(a.split)) for r in rs]
    df = pd.DataFrame(rows).sort_values(["config", "case"])
    df.to_csv(a.out, index=False)
    cols = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "removed_branches", "false_removed", "true_removed", "t_prune_s", "t_refine_s"]
    g = df.groupby("config")[[c for c in cols if c in df]].mean()
    base = g.loc["union"]
    g["dDSC"] = g.DSC - base.DSC; g["dBND"] = g.BND - base.BND
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    log(f"{a.split} ({os.path.basename(a.out)}):\n" + g.sort_values("DSC", ascending=False).round(3).to_string())


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("features", "sweep", "refine"))
    ap.add_argument("--splits", nargs="+", default=["val", "test", "pediatric"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--agreement", nargs="+", default=["n_models", "fractions"])
    ap.add_argument("--configs", help="json list of configurations for the refine phase")
    ap.add_argument("--out", help="per-case csv for the refine phase")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.phase == "features":
        jobs = [(s, cid) for s in a.splits for cid in common.cases(s)]
        with Pool(a.workers) as pool:
            for msg in pool.imap_unordered(features_case, jobs):
                print(msg, flush=True)
        log(f"features done for {a.splits}")
    elif a.phase == "sweep":
        run_sweep(a)
    elif a.phase == "refine":
        run_refine(a)


if __name__ == "__main__":
    main()
