#!/usr/bin/env python
"""Generation-specific ensembling (ablation method 4).

Every voxel of the union mask M gets the generation of its branch in M's own
branch tree (tab.generation via branch_map; generation -1 would count as
deepest).  Generations are grouped into bands and one fusion rule is chosen per
band on the validation split:

    union            keep every voxel of M
    maj2of3          at least two of {AirMorph, MEDPSeg, TfeNet}
    am|md, am|tf, md|tf   pairwise unions
    vote>=k          k of the 7 predictors (5 AirMorph members, MEDPSeg, TfeNet), k=2..5
    conf>=t          calibrated P(airway | pattern), table fitted on val, t=0.2..0.7
    <rule>+core      the same, but the 1-voxel core around M's skeleton
                     (dist <= max spacing) is always kept, so that trimming a
                     proximal band cannot disconnect distal bands.

The seeded-component rule (common.keep_seeded_component, seed = AirMorph) is
re-applied last in every case.

Search.  Because each band's voxels are kept or dropped independently of the
other bands (before the final connectivity step), DSC / TLD / BND / Sens / Prec
of any band->rule assignment follow exactly from per-(generation, rule) count
tables: kept voxels, kept true-positive voxels, and which GT-skeleton voxels
are covered (per GT branch).  These tables are computed once per case
(`precompute`), and the greedy coordinate ascent over bands (2 passes) runs
on them in seconds.  The chosen configurations are then verified with the
full pipeline (volume, keep_seeded_component, common.score) on val, and the
connectivity step's effect is reported.  Selection rule (brief): maximise mean
val DSC subject to mean BND loss <= 1.0 point vs the union.

Usage (AIRWAY_ROOT set, airmorph env):
    generation_specific.py precompute --split val|test|pediatric [--workers 3]
    generation_specific.py search                 # val -> configs.json (+ oracle on test tables)
    generation_specific.py evaluate --split val|test|pediatric --config NAME[,NAME..] [--cldice] [--save]
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402

OUT = os.path.join(common.STUDY, "ablation", "generation_specific")
GMAX = 20                     # generations >= GMAX (and -1) pooled into GMAX
THRESH = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
BANDINGS = {
    "B5": [(1, 2), (3, 4), (5, 6), (7, 8), (9, GMAX)],
    "B3": [(1, 3), (4, 6), (7, GMAX)],
    "B10": [(g, g) for g in range(1, 10)] + [(10, GMAX)],
}
# reference means (brief / common.score on the union), used for the constraint
UNION_VAL_BND = 93.23


# ----------------------------------------------------------------- rules

def rule_names():
    base = ["union", "maj2of3", "am|md", "am|tf", "md|tf"]
    base += [f"vote>={k}" for k in (2, 3, 4, 5)]
    base += [f"conf>={t:.1f}" for t in THRESH]
    return base + [b + "+core" for b in base]


def rule_masks(c, idx, lut):
    """Dict rule -> bool vector over the voxels of M (flat index idx)."""
    pat = c["pattern"].ravel()[idx].astype(np.int64)
    am = c["airmorph"].ravel()[idx]; md = c["medpseg"].ravel()[idx]; tf = c["tfenet"].ravel()[idx]
    a5 = pat // 4; m1 = (pat // 2) % 2; t1 = pat % 2
    votes7 = a5 + m1 + t1
    conf = lut[pat]
    core = c["dist"].ravel()[idx] <= float(np.max(c["spacing"]))
    R = {"union": np.ones(idx.size, bool),
         "maj2of3": (am.astype(np.int8) + md + tf) >= 2,
         "am|md": am | md, "am|tf": am | tf, "md|tf": md | tf}
    for k in (2, 3, 4, 5):
        R[f"vote>={k}"] = votes7 >= k
    for t in THRESH:
        R[f"conf>={t:.1f}"] = conf >= t
    for k in list(R):
        R[k + "+core"] = R[k] | core
    return R


def voxel_generation(c, idx):
    gen = c["tab"]["generation"].to_numpy(np.int64).copy()
    gen[gen < 1] = GMAX                      # -1 (unreached) counts as deepest
    gen = np.minimum(gen, GMAX)
    return gen[c["branch_map"].ravel()[idx]]


# ----------------------------------------------------------------- tables

def precompute_case(args):
    split, cid = args
    t0 = time.time()
    c = common.load_cached(split, cid)
    lut = np.zeros(256, np.float32)
    lut[:24] = np.nan_to_num(common.pattern_table("val")["p"], nan=0.5)
    idx = np.flatnonzero(c["M"].ravel())
    gen = voxel_generation(c, idx)
    R = rule_masks(c, idx, lut)
    gt = c["gt"].ravel()[idx].astype(bool)
    skel_all = c["gt_skeleton"].ravel().astype(bool)
    lab_all = (c["gt_parsing"].ravel() * skel_all)                 # label_skeleton * label_parsing
    label_branch_bincount = np.bincount(lab_all)[1:]                # verbatim metric
    sk_in_M = skel_all[idx]                                          # GT skeleton voxels inside M
    sk_idx = np.flatnonzero(sk_in_M)
    sk_lab = lab_all[idx][sk_idx]; sk_gen = gen[sk_idx]
    names = rule_names()
    P = np.zeros((len(names), GMAX + 1), np.int64); TP = np.zeros_like(P)
    kept_sk = np.zeros((len(names), sk_idx.size), bool)
    for i, r in enumerate(names):
        k = R[r]
        P[i] = np.bincount(gen[k], minlength=GMAX + 1)
        TP[i] = np.bincount(gen[k & gt], minlength=GMAX + 1)
        kept_sk[i] = k[sk_idx]
    out = dict(rules=np.array(names), P=P, TP=TP, kept_sk=kept_sk, sk_lab=sk_lab, sk_gen=sk_gen,
               n_gt=int(gt.sum()) + int(c["gt"].sum() - gt.sum()), n_gt_total=int(c["gt"].sum()),
               n_skel_total=int(skel_all.sum()), label_branch_bincount=label_branch_bincount,
               n_bg=int(c["gt"].size - c["gt"].sum()))
    d = os.path.join(OUT, "tables", split); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, cid + ".npz"), **out)
    return cid, time.time() - t0


def load_tables(split):
    T = []
    for cid in common.cases(split):
        z = np.load(os.path.join(OUT, "tables", split, cid + ".npz"))
        T.append({k: z[k] for k in z.files} | {"case": cid})
    return T


def band_of_generation(banding):
    """generation (1..GMAX) -> band index."""
    b = np.zeros(GMAX + 1, np.int64)
    for i, (lo, hi) in enumerate(banding):
        b[lo:hi + 1] = i
    return b


def score_tables(T, banding, assign, rule_index):
    """Exact (pre-connectivity) per-case metrics for one band->rule assignment,
    mirroring the rounding of util.airwayatlas_eval."""
    bog = band_of_generation(banding)
    rule_of_gen = np.array([rule_index[assign[bog[g]]] if g >= 1 else 0 for g in range(GMAX + 1)])
    rows = []
    for t in T:
        gens = np.arange(GMAX + 1)
        n_pred = int(t["P"][rule_of_gen, gens].sum()); tp = int(t["TP"][rule_of_gen, gens].sum())
        n_gt = int(t["n_gt_total"])
        dsc = round((2.0 * tp + 1e-5) / (n_pred + n_gt + 1e-5) * 100, 2)
        sens = round((tp + 1e-5) * 100 / (n_gt + 1e-5), 3)
        prec = round((tp + 1e-5) * 100 / (n_pred + 1e-5), 3)
        kept = t["kept_sk"][rule_of_gen[t["sk_gen"]], np.arange(t["sk_gen"].size)]
        tld = round((kept.sum() + 1e-5) / (t["n_skel_total"] + 1e-5) * 100, 2)
        lb = t["label_branch_bincount"]
        pb = np.bincount(t["sk_lab"][kept], minlength=lb.size + 1)[1:lb.size + 1].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = pb / lb
        bnd = round(np.count_nonzero(np.where(ratio >= 0.8, 1, 0)) * 100 / lb.size, 2)
        fp = n_pred - tp
        rows.append({"case": t["case"], "DSC": dsc, "TLD": tld, "BND": bnd, "Sensitivity": sens,
                     "Precision": prec, "FPR": round((fp + 1e-5) * 100 / (t["n_bg"] + 1e-5), 3)})
    return rows


def means(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "case"}


def greedy(T, banding, candidates, ref_bnd, max_bnd_loss=1.0, passes=2, log=print):
    names = list(T[0]["rules"]); rule_index = {r: i for i, r in enumerate(names)}
    assign = ["union"] * len(banding)
    best = means(score_tables(T, banding, assign, rule_index))
    log(f"  start all-union: {fmt(best)}")
    for p in range(passes):
        changed = False
        for b in range(len(banding)):
            cur = assign[b]
            for r in candidates:
                trial = list(assign); trial[b] = r
                m = means(score_tables(T, banding, trial, rule_index))
                feasible = m["BND"] >= ref_bnd - max_bnd_loss
                if feasible and m["DSC"] > best["DSC"] + 1e-9:
                    best, assign, changed = m, trial, True
            if assign[b] != cur:
                log(f"  pass {p + 1} band {banding[b]}: {cur} -> {assign[b]}  {fmt(best)}")
        if not changed:
            log(f"  pass {p + 1}: no change"); break
    return assign, best


def fmt(m):
    return " ".join(f"{k} {v:.2f}" for k, v in m.items() if k in ("DSC", "TLD", "BND", "Sensitivity", "Precision"))


# ----------------------------------------------------------------- full pipeline

def build_mask(c, banding, assign, lut):
    idx = np.flatnonzero(c["M"].ravel())
    gen = voxel_generation(c, idx)
    R = rule_masks(c, idx, lut)
    bog = band_of_generation(banding)
    keep = np.zeros(idx.size, bool)
    for b, (lo, hi) in enumerate(banding):
        sel = (bog[gen] == b)
        keep[sel] = R[assign[b]][sel]
    pre = np.zeros(c["M"].size, np.uint8); pre[idx[keep]] = 1
    pre = pre.reshape(c["M"].shape)
    out = common.keep_seeded_component(pre, c["airmorph"])
    return pre, out


def evaluate_case(args):
    """One case, several configurations (the case is loaded once)."""
    split, cid, cfgs, cldice, save = args
    t0 = time.time()
    c = common.load_cached(split, cid)
    t1 = time.time()
    lut = np.zeros(256, np.float32)
    lut[:24] = np.nan_to_num(common.pattern_table("val")["p"], nan=0.5)
    out_rows = []
    for name, banding, assign in cfgs:
        t2 = time.time()
        pre, out = build_mask(c, banding, assign, lut)
        t_fuse = time.time() - t2
        s = common.score(out, c, cldice=cldice)
        s.update(case=cid, config=name, n_pre=int(pre.sum()), n_out=int(out.sum()),
                 removed_by_component=int(pre.sum() - out.sum()), t_load=t1 - t0, t_fuse=t_fuse,
                 t_total=time.time() - t2 + (t1 - t0))
        if save:
            d = os.path.join(OUT, "masks", split); os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"ATM26_{cid}.npz"), mask=out.astype(np.uint8))
        out_rows.append(s)
    return out_rows


def run_pool(fn, jobs, workers):
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for r in pool.imap_unordered(fn, jobs):
                yield r
    else:
        for j in jobs:
            yield fn(j)


# ----------------------------------------------------------------- main

def cmd_precompute(a):
    jobs = [(a.split, cid) for cid in common.cases(a.split)]
    for cid, dt in run_pool(precompute_case, jobs, a.workers):
        print(f"{a.split} {cid} tables {dt:.1f}s", flush=True)


def cmd_search(a):
    import pandas as pd
    logf = open(os.path.join(OUT, "logs", "search.log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    log(f"=== search {time.strftime('%Y-%m-%d %H:%M')} ===")
    T = load_tables("val")
    names = list(T[0]["rules"]); rule_index = {r: i for i, r in enumerate(names)}
    union = means(score_tables(T, BANDINGS["B5"], ["union"] * 5, rule_index))
    log(f"val union (tables): {fmt(union)}  (brief: DSC 92.45 BND 93.23)")
    ref_bnd = union["BND"]
    # single-rule-everywhere reference, for the log
    rows = []
    for r in names:
        m = means(score_tables(T, BANDINGS["B3"], [r] * 3, rule_index))
        rows.append({"rule": r, **m}); log(f"  whole mask {r:14s} {fmt(m)}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "val_single_rule_whole_mask.csv"), index=False)
    base = [r for r in names if not r.endswith("+core")]
    core = [r for r in names if r.endswith("+core")]
    configs = {}
    for bname, banding in BANDINGS.items():
        for mode, cands in (("nocore", base), ("core", core + ["union"]), ("any", names)):
            log(f"--- val search {bname} mode {mode}")
            assign, m = greedy(T, banding, cands, ref_bnd, log=log)
            key = f"{bname}_{mode}"
            configs[key] = {"banding": banding, "assign": assign, "val_tables": m, "fitted_on": "val"}
            log(f"  => {key}: {dict(zip([f'{lo}-{hi}' for lo, hi in banding], assign))}  {fmt(m)}")
    # oracle: the same search on the TEST tables (ceiling of the family; not a valid result)
    Tt = load_tables("test")
    union_t = means(score_tables(Tt, BANDINGS["B5"], ["union"] * 5, rule_index))
    log(f"test union (tables): {fmt(union_t)}")
    for bname, banding in BANDINGS.items():
        log(f"--- ORACLE search on TEST {bname} (any)")
        assign, m = greedy(Tt, banding, names, union_t["BND"], log=log)
        key = f"oracle_{bname}"
        configs[key] = {"banding": banding, "assign": assign, "test_tables": m, "fitted_on": "TEST (oracle)"}
        log(f"  => {key}: {dict(zip([f'{lo}-{hi}' for lo, hi in banding], assign))}  {fmt(m)}")
    # table-predicted test numbers for the val-chosen configs (no masks built here)
    for key, cfg in configs.items():
        cfg["test_tables"] = means(score_tables(Tt, cfg["banding"], cfg["assign"], rule_index))
    with open(os.path.join(OUT, "configs.json"), "w") as fh:
        json.dump(configs, fh, indent=1)
    log("wrote configs.json")


def cmd_evaluate(a):
    import pandas as pd
    with open(os.path.join(OUT, "configs.json")) as fh:
        configs = json.load(fh)
    names = a.config.split(",")
    assert not (a.save and len(names) > 1), "save one configuration at a time"
    cfgs = [(n, [tuple(b) for b in configs[n]["banding"]], configs[n]["assign"]) for n in names]
    jobs = [(a.split, cid, cfgs, a.cldice, a.save) for cid in common.cases(a.split)]
    rows = []
    for rs in run_pool(evaluate_case, jobs, a.workers):
        rows.extend(rs)
        for s in rs:
            print(f"{a.split} {s['case']} {s['config']} DSC {s['DSC']:.2f} BND {s['BND']:.2f} "
                  f"removed {s['removed_by_component']} t {s['t_total']:.1f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["config", "case"])
    with open(os.path.join(OUT, "logs", "evaluate.log"), "a") as fh:
        for n in names:
            d = df[df.config == n]
            d.to_csv(os.path.join(OUT, f"percase_{a.split}_{n}.csv"), index=False)
            m = d[["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "FPR"]].mean()
            line = (f"{a.split} {n} cldice={a.cldice}: " + " ".join(f"{k} {v:.2f}" for k, v in m.items())
                    + f" | removed by component {d.removed_by_component.mean():.0f} vox "
                    f"({(d.removed_by_component / d.n_pre).mean() * 100:.3f} %) fuse {d.t_fuse.mean():.1f}s "
                    f"load {d.t_load.mean():.1f}s")
            print("== " + line); fh.write(f"{time.strftime('%Y-%m-%d %H:%M')} {line}\n")


def cmd_report(a):
    """Markdown tables: means per split, paired per-case differences vs union and
    refined (from results/whole_<split>.csv), per-generation table from per_generation.py."""
    import pandas as pd
    cols = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision"]
    out = []
    for split in ("val", "test", "pediatric"):
        f = os.path.join(OUT, f"percase_{split}_{a.config}.csv")
        if not os.path.exists(f):
            continue
        me = pd.read_csv(f).set_index("case")
        ref = pd.read_csv(os.path.join(common.STUDY, "results", f"whole_{split}.csv"))
        un = ref[ref.strategy == "union_seeded"].set_index("case").loc[me.index]
        rf = ref[ref.strategy == "topofuse"].set_index("case").loc[me.index]
        out.append(f"\n### {split} (n={len(me)})\n")
        out.append("| method | " + " | ".join(cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for name, d in (("union, seeded", un), ("union, refined", rf), (f"generation-specific ({a.config})", me)):
            out.append(f"| {name} | " + " | ".join(f"{d[c].mean():.2f}" for c in cols) + " |")
        out.append(f"\nPaired per-case differences ({split}), generation-specific minus reference:\n")
        out.append("| vs | " + " | ".join(f"{c} mean diff (improved/{len(me)})" for c in cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for name, d in (("union", un), ("refined", rf)):
            cells = []
            for c in cols:
                diff = me[c] - d[c]
                cells.append(f"{diff.mean():+.2f} ({int((diff > 0).sum())}/{len(me)}; ties {int((diff == 0).sum())})")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(OUT, f"report_tables_{a.config}.md"), "w") as fh:
        fh.write(txt + "\n")


def cmd_exhaustive(a):
    """Enumerate every band->rule assignment of the B3 banding on one split's
    tables (30^3 configurations) and report the best DSC under BND-loss slacks
    1.0 and 1.2 and unconstrained.  On test this is the (labelled) oracle ceiling
    of the family; on val it checks the greedy search."""
    import itertools, pandas as pd
    T = load_tables(a.split)
    names = list(T[0]["rules"]); ri = {r: i for i, r in enumerate(names)}
    banding = BANDINGS["B3"]
    ref = means(score_tables(T, banding, ["union"] * 3, ri))["BND"]
    rows = []
    t0 = time.time()
    for i, assign in enumerate(itertools.product(names, repeat=3)):
        m = means(score_tables(T, banding, list(assign), ri))
        rows.append({"b1": assign[0], "b2": assign[1], "b3": assign[2], **m})
        if i % 2000 == 0:
            print(f"{i} configs {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df["BND_loss"] = ref - df["BND"]
    df.to_csv(os.path.join(OUT, f"exhaustive_B3_{a.split}.csv"), index=False)
    with open(os.path.join(OUT, "logs", "search.log"), "a") as fh:
        for slack in (1.0, 1.2, float("inf")):
            d = df[df.BND_loss <= slack].sort_values("DSC", ascending=False).head(5)
            line = f"exhaustive B3 on {a.split} (tables, pre-connectivity), BND loss <= {slack}:\n" + d.to_string()
            print(line); fh.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precompute"); p.add_argument("--split", required=True); p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("search")
    p = sub.add_parser("evaluate"); p.add_argument("--split", required=True); p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=3); p.add_argument("--cldice", action="store_true")
    p.add_argument("--save", action="store_true")
    p = sub.add_parser("exhaustive"); p.add_argument("--split", required=True)
    p = sub.add_parser("report"); p.add_argument("--config", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    {"precompute": cmd_precompute, "search": cmd_search, "evaluate": cmd_evaluate, "report": cmd_report, "exhaustive": cmd_exhaustive}[a.cmd](a)


if __name__ == "__main__":
    main()
