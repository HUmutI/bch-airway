#!/usr/bin/env python
"""Ablation 5: boundary-aware fusion of the three airway models.

The union keeps the fattest boundary any member draws; the majority vote loses
distal branches. Here the interior and the boundary are fused separately, via
signed distance functions (SDF, mm, positive inside; EDT inside minus EDT
outside, computed on the union bounding box + 10 voxels in float32):

  a. sdf_mean   : weighted mean of the three SDFs (weights on a 0.1 grid summing
                  to 1), thresholded at level L in {-0.5..+0.5 mm}.
  b. sdf_median : voxel-wise median of the three SDFs, thresholded at L.
  c. core_shell : the union's core (dist <= max(0.5 r_b, cmin voxels) from the
                  union skeleton, r_b = branch radius from the union's branch
                  tree) is always kept; the shell is decided by (a), (b) or the
                  majority vote.
  d. tube       : every union branch is rebuilt as a tube of consensus calibre
                  rho_b around the union's centreline: keep x in the union with
                  dist(x) <= s * rho_b + delta, rho_b = per-branch median radius
                  of the majority-vote mask ("maj") or of the mean member radius
                  ("memb") along the branch; branches the consensus misses keep
                  the union calibre. "tube_scaled" keeps the union's (non-
                  circular) cross-section shape and rescales it by rho_b / r_b.

Everything ends with common.keep_seeded_component(mask, airmorph). All
tunables are chosen on the validation split only (sweep.py-style rule:
maximise mean DSC subject to mean BND loss <= 1 point vs the union); the test
split and the pediatric cases are scored once with the chosen settings.

    python boundary_aware.py sweep --split val --workers 3
    python boundary_aware.py select
    python boundary_aware.py final --split test --workers 3
"""
import argparse, itertools, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common                                   # noqa: E402
from hd_refined import distances as hd_distances  # noqa: E402

OUT = os.path.join(common.STUDY, "ablation", "boundary_aware")
MARGIN = 10
LEVELS = [round(x, 1) for x in np.arange(-0.5, 0.51, 0.1)]
WEIGHTS = [(round(a / 10, 1), round(b / 10, 1), round((10 - a - b) / 10, 1))
           for a in range(11) for b in range(11 - a)]
CMINS = [0.0, 1.0]                 # core floor, in voxels (max spacing)
TUBE_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
TUBE_SCALED_SCALES = [0.9, 1.0, 1.1]
TUBE_DELTAS = [0.0, 0.5, 1.0]      # in voxels (max spacing)
MEMBERS = ("airmorph", "medpseg", "tfenet")


# ----------------------------------------------------------------- geometry

def crop_slices(c, margin=MARGIN):
    """Bounding box of the union (+margin) merged with the GT box, so that the
    voxel metrics computed on the crop equal those on the full grid."""
    M = c["M"]
    objs = ndimage.find_objects(M.astype(np.uint8))[0]
    sl = [slice(max(0, s.start - margin), min(d, s.stop + margin)) for s, d in zip(objs, M.shape)]
    g = ndimage.find_objects(c["gt"])[0]
    return tuple(slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(sl, g))


def sdf(mask, spacing):
    """Signed distance (mm), positive inside, float32."""
    mask = mask.astype(bool)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing).astype(np.float32)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing).astype(np.float32)
    return inside - outside


def prepare(c):
    """Crop a cached case, compute the SDFs and the per-branch calibres, and
    flatten everything onto the candidate voxel set C = {within 1 mm of any
    member} | union | GT. Every fusion below lies inside C (a voxel with mean
    or median SDF > -0.5 mm has some member SDF > -0.5 mm), so scoring on the
    vectors is exact."""
    t0 = time.process_time()
    sl = crop_slices(c)
    sp = tuple(float(x) for x in c["spacing"])
    vox = float(max(sp))
    P = {"sl": sl, "shape": c["gt"].shape, "spacing": sp, "vox": vox}
    M = c["M"][sl]
    gt = c["gt"][sl].astype(bool)
    mem = [c[k][sl] for k in MEMBERS]
    S = [sdf(m, sp) for m in mem]
    C = (np.maximum(np.maximum(S[0], S[1]), S[2]) > -1.0) | M | gt
    P["cand"] = C
    P["seed"] = c["airmorph"][sl]
    P["M"] = M[C]
    P["gt"] = gt[C]
    P["label_branch"] = (c["gt_skeleton"][sl].astype(np.int64) * c["gt_parsing"][sl])[C]
    P["gt_skel"] = c["gt_skeleton"][sl].astype(bool)[C]
    P["dist"] = c["dist"][sl][C]
    P["bm"] = c["branch_map"][sl][C]
    P["sdf"] = [x[C] for x in S]
    votes = sum(m.astype(np.uint8) for m in mem)
    maj = votes >= 2
    P["maj"] = maj[C]
    # local union calibre along the ray through x: distance to the skeleton + distance to the boundary
    edtM = ndimage.distance_transform_edt(M, sampling=sp).astype(np.float32)
    P["R_local"] = P["dist"] + edtM[C]
    # per-branch radius from the union tree (branch 0 -> 0)
    r = c["tab"]["r_mean_mm"].to_numpy(np.float32).copy(); r[0] = 0
    P["r_union"] = r
    # consensus calibre along each branch from the union skeleton voxels
    sk = (P["dist"] == 0) & P["M"]
    ids = P["bm"][sk]
    edt_maj = ndimage.distance_transform_edt(maj, sampling=sp).astype(np.float32)[C][sk]
    memb = np.stack([x[sk] for x in P["sdf"]], 0)       # inside EDT where positive
    pos = memb > 0
    memb_mean = np.where(pos.any(0), (memb * pos).sum(0) / np.maximum(pos.sum(0), 1), 0.0)
    nb = len(r)
    P["rho"] = {}
    for name, vals in (("maj", edt_maj), ("memb", memb_mean)):
        df = pd.DataFrame({"b": ids, "v": vals})
        df = df[df.v > 0]
        med = df.groupby("b").v.median()
        cov = df.groupby("b").size() / np.maximum(np.bincount(ids, minlength=nb)[med.index], 1)
        rho = r.copy()
        ok = med.index[(cov >= 0.5).to_numpy()]
        rho[ok] = med.loc[ok].to_numpy(np.float32)
        P["rho"][name] = rho
    P["prep_cpu"] = time.process_time() - t0
    return P


# ----------------------------------------------------------------- the fusion rules

def fuse(P, method, prm):
    """Return the fused mask on the crop (bool), before the seeded component."""
    M, dist, bm = P["M"], P["dist"], P["bm"]
    if method == "sdf_mean":
        w = prm["w"]
        f = w[0] * P["sdf"][0] + w[1] * P["sdf"][1] + w[2] * P["sdf"][2]
        return f > prm["L"]
    if method == "sdf_median":
        if "sdf_med" not in P:
            P["sdf_med"] = np.median(np.stack(P["sdf"], 0), axis=0).astype(np.float32)
        return P["sdf_med"] > prm["L"]
    if method == "core_shell":
        core = M & (dist <= np.maximum(0.5 * P["r_union"][bm], prm["cmin"] * P["vox"]))
        shell = prm["shell"]
        if shell == "majority":
            sh = P["maj"] & M
        elif shell == "sdf_median":
            sh = fuse(P, "sdf_median", prm)
        else:
            sh = fuse(P, "sdf_mean", prm)
        return core | sh
    if method == "tube":
        rho = P["rho"][prm["rho"]][bm]
        return M & (dist <= prm["s"] * rho + prm["delta"] * P["vox"])
    if method == "tube_scaled":
        # the union's local calibre R(x) (dist to skeleton + dist to boundary) rescaled by the
        # consensus/union calibre ratio of the branch: keeps non-circular cross-sections
        k = P["rho"][prm["rho"]] / np.maximum(P["r_union"], 1e-3); k[0] = 1.0
        return M & (dist <= prm["s"] * np.minimum(k[bm], 1.0) * P["R_local"] + prm["delta"] * P["vox"])
    raise ValueError(method)


def to_crop(P, vec):
    out = np.zeros(P["cand"].shape, bool)
    out[P["cand"]] = vec
    return out


def finish(P, vec):
    """Seeded component on the crop (equivalent to the full grid: the margin
    holds every voxel the fusion can produce). Returns a vector again."""
    return common.keep_seeded_component(to_crop(P, vec), P["seed"]).astype(bool)[P["cand"]]


def to_full(P, vec):
    out = np.zeros(P["shape"], np.uint8)
    out[P["sl"]] = to_crop(P, vec)
    return out


# ----------------------------------------------------------------- fast exact metrics

class Scorer:
    """DSC, TLD, BND, Sens, Prec on the crop, with the ATM'22 rounding.
    Exact because every GT voxel and every predicted voxel lies in the crop."""

    def __init__(self, P):
        self.gt = P["gt"]
        self.n_gt = int(self.gt.sum())
        self.skel = P["gt_skel"]
        self.n_skel = int(self.skel.sum())
        lb = P["label_branch"]
        self.lb = lb
        cnt = np.bincount(lb.ravel())[1:]
        self.lb_count = cnt
        self.total_branch = cnt.shape[0]

    def __call__(self, pred):
        pred = pred.astype(bool)
        n_p = int(pred.sum()); tp = int((pred & self.gt).sum())
        dsc = round(((2.0 * tp + 1e-5) / (n_p + self.n_gt + 1e-5)) * 100, 2)
        tld = round(((pred & self.skel).sum() + 1e-5) / (self.n_skel + 1e-5) * 100, 2)
        pb = np.bincount(self.lb[pred], minlength=self.total_branch + 1)[1:self.total_branch + 1]
        ratio = pb / self.lb_count
        bnd = round(np.count_nonzero(ratio >= 0.8) * 100 / self.total_branch, 2)
        sens = round((tp + 1e-5) / (self.n_gt + 1e-5) * 100, 3)
        prec = round((tp + 1e-5) / (n_p + 1e-5) * 100, 3)
        return {"DSC": dsc, "TLD": tld, "BND": bnd, "Sensitivity": sens, "Precision": prec, "n_vox": n_p}


# ----------------------------------------------------------------- configs

def configs(families=None, cmins=CMINS, levels=LEVELS):
    fam = set(families) if families else {"sdf_mean", "sdf_median", "core_shell", "tube", "tube_scaled"}
    cf = []
    if "sdf_mean" in fam:
        for w in WEIGHTS:
            for L in levels:
                cf.append(("sdf_mean", {"w": w, "L": L}))
    if "sdf_median" in fam:
        for L in levels:
            cf.append(("sdf_median", {"L": L}))
    for cmin in (cmins if "core_shell" in fam else []):
        cf.append(("core_shell", {"cmin": cmin, "shell": "majority"}))
        for L in levels:
            cf.append(("core_shell", {"cmin": cmin, "shell": "sdf_median", "L": L}))
        for w in WEIGHTS:
            for L in levels:
                cf.append(("core_shell", {"cmin": cmin, "shell": "sdf_mean", "w": w, "L": L}))
    if not fam & {"tube", "tube_scaled"}:
        return cf
    for rho in ("maj", "memb"):
        for s in TUBE_SCALES:
            for d in TUBE_DELTAS:
                cf.append(("tube", {"rho": rho, "s": s, "delta": d}))
        for s in TUBE_SCALED_SCALES:
            for d in TUBE_DELTAS:
                cf.append(("tube_scaled", {"rho": rho, "s": s, "delta": d}))
    return cf


def key(method, prm):
    return method + "|" + json.dumps(prm, sort_keys=True)


def parse_key(k):
    m, j = k.split("|", 1)
    prm = json.loads(j)
    if "w" in prm:
        prm["w"] = tuple(prm["w"])
    return m, prm


# ----------------------------------------------------------------- sweep

CFG = None


def sweep_case(cid):
    split = CFG["split"]
    t0 = time.time()
    c = common.load_cached(split, cid)
    P = prepare(c)
    S = Scorer(P)
    rows = [{"case": cid, "key": "union", "method": "union", **S(P["M"])}]
    rows.append({"case": cid, "key": "majority_seeded", "method": "majority", **S(finish(P, P["maj"]))})
    cache = {}
    for method, prm in configs(CFG.get("families"), CFG.get("cmins", CMINS), CFG.get("levels", LEVELS)):
        if method == "sdf_mean" or (method == "core_shell" and prm.get("shell") == "sdf_mean"):
            k = ("mean", prm["w"])
            if k not in cache:
                w = prm["w"]
                cache = {k: w[0] * P["sdf"][0] + w[1] * P["sdf"][1] + w[2] * P["sdf"][2]}
            f = cache[k] > prm["L"]
            if method == "core_shell":
                core = P["M"] & (P["dist"] <= np.maximum(0.5 * P["r_union"][P["bm"]], prm["cmin"] * P["vox"]))
                f = core | f
            m = f
        else:
            m = fuse(P, method, prm)
        rows.append({"case": cid, "key": key(method, prm), "method": method, **S(m)})
    dt = time.time() - t0
    print(f"{cid}: {len(rows)} configs, {dt:.0f}s, prep cpu {P['prep_cpu']:.0f}s", flush=True)
    return rows


def cmd_sweep(a):
    global CFG
    CFG = {"split": a.split}
    if a.families:
        CFG["families"] = a.families.split(",")
    if a.cmins:
        CFG["cmins"] = [float(x) for x in a.cmins.split(",")]
    if a.levels:
        CFG["levels"] = [float(x) for x in a.levels.split(",")]
    ids = common.cases(a.split)
    with Pool(a.workers) as p:
        rows = [r for rs in p.imap_unordered(sweep_case, ids) for r in rs]
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"sweep_{a.split}{a.tag}_percase.csv"), index=False)
    agg = df.groupby(["key", "method"])[["DSC", "TLD", "BND", "Sensitivity", "Precision"]].mean().reset_index()
    agg.to_csv(os.path.join(OUT, f"sweep_{a.split}{a.tag}_mean.csv"), index=False)
    print(agg.sort_values("DSC", ascending=False).head(20).to_string())


# ----------------------------------------------------------------- selection (val, unseeded sweep -> seeded shortlist)

def shortlist(agg, n_per_method=6, bnd_slack=1.0):
    """Per method family, the n best-DSC configs meeting the BND rule on the
    unseeded sweep."""
    union = agg[agg.key == "union"].iloc[0]
    out = []
    for m, g in agg[~agg.method.isin(["union", "majority"])].groupby("method"):
        ok = g[g.BND >= union.BND - bnd_slack].sort_values("DSC", ascending=False).head(n_per_method)
        out.append(ok)
    return pd.concat(out)


def final_case(args):
    cid, split, keys, with_cldice, save_dir, hd = args[:6]
    save_keys = set(args[6]) if len(args) > 6 else set(keys)
    c = common.load_cached(split, cid)
    P = prepare(c)
    rows = []
    sp = tuple(float(x) for x in c["spacing"])
    gt_full = c["gt"]
    for k in keys:
        t0 = time.process_time()
        if k == "union":
            full = c["M"].astype(np.uint8)
        else:
            m, prm = parse_key(k)
            full = to_full(P, finish(P, fuse(P, m, prm)))
        cpu = time.process_time() - t0
        sc = common.score(full, c, cldice=with_cldice)
        row = {"case": cid, "key": k, "cpu_fuse_s": cpu, "cpu_prep_s": P["prep_cpu"], **sc}
        if hd:
            row.update(hd_distances(full, gt_full, sp))
        rows.append(row)
        if save_dir and k != "union" and k in save_keys:
            os.makedirs(save_dir, exist_ok=True)
            np.savez_compressed(os.path.join(save_dir, f"ATM26_{cid}.npz"), mask=full.astype(np.uint8))
    print(cid, "done", flush=True)
    return rows


def cmd_select(a):
    import glob
    agg = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "sweep_val*_mean.csv")))]).drop_duplicates("key")
    sl = shortlist(agg, a.n_per_method, a.slack)
    keys = ["union"] + sl.key.tolist()
    print("shortlist:", len(keys)); print(sl.to_string())
    ids = common.cases("val")
    with Pool(a.workers) as p:
        rows = [r for rs in p.imap_unordered(final_case, [(cid, "val", keys, False, None, False) for cid in ids]) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "shortlist_val_percase.csv"), index=False)
    agg2 = df.groupby("key")[["DSC", "TLD", "BND", "Sensitivity", "Precision", "FPR"]].mean()
    agg2["method"] = [k.split("|")[0] for k in agg2.index]
    union = agg2.loc["union"]
    agg2["BND_loss"] = union.BND - agg2.BND
    agg2["ok"] = agg2.BND_loss <= a.slack
    agg2 = agg2.sort_values("DSC", ascending=False)
    agg2.to_csv(os.path.join(OUT, "shortlist_val_mean.csv"))
    print(agg2.to_string())
    chosen = {}
    for m, g in agg2[agg2.ok & (agg2.method != "union")].groupby("method"):
        chosen[m] = g.index[0]
    best = agg2[agg2.ok & (agg2.method != "union")].index[0]
    with open(os.path.join(OUT, "chosen.json"), "w") as fh:
        json.dump({"primary": best, "per_method": chosen}, fh, indent=1)
    print("primary:", best); print("per method:", chosen)


def cmd_final(a):
    with open(os.path.join(OUT, "chosen.json")) as fh:
        ch = json.load(fh)
    keys = ["union", ch["primary"]] + [k for k in ch["per_method"].values() if k != ch["primary"]]
    ids = common.cases(a.split)
    save = os.path.join(OUT, "masks", a.split) if a.split in ("test", "pediatric", "val") else None
    # only the primary mask is saved
    jobs = [(cid, a.split, keys, True, save, True, [ch["primary"]]) for cid in ids]
    with Pool(a.workers) as p:
        rows = [r for rs in p.imap_unordered(final_case, jobs) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"final_{a.split}_percase.csv"), index=False)
    cols = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "FPR", "HD", "HD95", "ASSD", "cpu_fuse_s", "cpu_prep_s"]
    agg = df.groupby("key")[cols].mean()
    agg.to_csv(os.path.join(OUT, f"final_{a.split}_mean.csv"))
    print(agg.round(3).to_string())


def cmd_tables(a):
    """Markdown tables for RESULTS.md from the final per-case CSVs and the
    study's reference per-case results (results/whole_<split>.csv)."""
    with open(os.path.join(OUT, "chosen.json")) as fh:
        ch = json.load(fh)
    cols = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision"]
    res_dir = os.path.join(os.path.dirname(HERE), "results")
    out = []
    for split in ("val", "test", "pediatric"):
        f = os.path.join(OUT, f"final_{split}_percase.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        ref = pd.read_csv(os.path.join(res_dir, f"whole_{split}.csv"))
        hd = os.path.join(res_dir, f"hd_refined_{split}.csv")
        hd = pd.read_csv(hd) if os.path.exists(hd) else None
        out.append(f"\n### {split} (n={df.case.nunique()})\n")
        out.append("| method | " + " | ".join(cols) + " | HD95 | ASSD |")
        out.append("|---|" + "---|" * (len(cols) + 2))
        for strat, name in (("union_seeded", "union, seeded"), ("topofuse", "union, refined (kappa 0.4)"), ("vote2of3_seeded", "majority vote, seeded")):
            r = ref[ref.strategy == strat][cols].mean()
            h95 = ""; assd = ""
            if hd is not None and strat == "topofuse":
                g = hd[hd.strategy == "Union, refined"]; h95 = f"{g.HD95.mean():.2f}"; assd = f"{g.ASSD.mean():.3f}"
            if hd is not None and strat == "union_seeded":
                g = hd[hd.strategy == "All three members (check)"]; h95 = f"{g.HD95.mean():.2f}"; assd = f"{g.ASSD.mean():.3f}"
            out.append(f"| {name} | " + " | ".join(f"{r[c]:.2f}" for c in cols) + f" | {h95} | {assd} |")
        for k, g in df.groupby("key", sort=False):
            r = g[cols].mean()
            tag = "**" if k == ch["primary"] else ""
            name = "union (this run, check)" if k == "union" else k
            out.append(f"| {tag}{name}{tag} | " + " | ".join(f"{r[c]:.2f}" for c in cols) + f" | {g.HD95.mean():.2f} | {g.ASSD.mean():.3f} |")
        if split == "test":
            out.append("\nPaired per-case differences on test (this method minus reference):\n")
            out.append("| config | vs | dDSC mean | cases DSC up | dBND mean | cases BND up | dTLD mean | dHD95 mean |")
            out.append("|---|---|---|---|---|---|---|---|")
            for k, g in df.groupby("key", sort=False):
                if k == "union":
                    continue
                g = g.set_index("case")
                for strat, name in (("union_seeded", "union"), ("topofuse", "refined")):
                    r = ref[ref.strategy == strat].set_index("case").loc[g.index]
                    d = g[cols] - r[cols]
                    hdd = ""
                    if hd is not None:
                        hs = "Union, refined" if strat == "topofuse" else "All three members (check)"
                        h = hd[hd.strategy == hs].set_index("case").loc[g.index]
                        hdd = f"{(g.HD95 - h.HD95).mean():+.3f}"
                    out.append(f"| {k} | {name} | {d.DSC.mean():+.2f} | {(d.DSC > 0).sum()}/{len(d)} | {d.BND.mean():+.2f} | {(d.BND > 0).sum()}/{len(d)} | {d.TLD.mean():+.2f} | {hdd} |")
        out.append(f"\nCPU seconds per scan ({split}): SDF/calibre preparation {df.cpu_prep_s.mean():.0f} s, fusion+component {df[df.key == ch['primary']].cpu_fuse_s.mean():.1f} s")
    txt = "\n".join(out)
    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep"); s.add_argument("--split", default="val"); s.add_argument("--workers", type=int, default=3)
    s.add_argument("--families", default=""); s.add_argument("--cmins", default=""); s.add_argument("--levels", default=""); s.add_argument("--tag", default="")
    s = sub.add_parser("select"); s.add_argument("--workers", type=int, default=3); s.add_argument("--n-per-method", type=int, default=6); s.add_argument("--slack", type=float, default=1.0)
    s = sub.add_parser("final"); s.add_argument("--split", required=True); s.add_argument("--workers", type=int, default=3)
    sub.add_parser("tables")
    a = ap.parse_args()
    {"sweep": cmd_sweep, "select": cmd_select, "final": cmd_final, "tables": cmd_tables}[a.cmd](a)


if __name__ == "__main__":
    main()
