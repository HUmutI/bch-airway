#!/usr/bin/env python
"""Ablation 3: uncertainty-weighted fusion of three binary airway masks.

The members are binary, so a model's uncertainty has to be estimated from the
masks themselves. Variants (all end with common.keep_seeded_component):

 (a) local reliability: U_m(x) = fraction of voxels in a radius-r ball around x
     where member m disagrees with the majority of the three (norm="ball"), or
     the same count divided by the number of union voxels in the ball
     (norm="union"); w_m(x) = 1/(U_m(x)+eps); p = sum w_m mask_m / sum w_m;
     keep p >= t.
 (b) as (a) with a global prior g_m multiplied into w_m: precision on val of the
     voxels only member m marks (from common.pattern_table("val")).
 (c) calibrated pattern confidence P(airway | pattern) (val table), optionally
     smoothed inside M with a gaussian (sigma voxels, mask-normalised); keep
     conf >= t. The entropy of that confidence is the uncertainty reported.
 (d) (a)/(b) with a branch-relative threshold: keep p >= t * conf_skel(branch),
     conf_skel being the branch's centreline support from the cached tree.

All metrics are computed on the bounding box of (M | gt) padded by the largest
radius; every metric of common.score is a whole-volume sum so this is exact,
except FPR whose denominator is restored to the full volume.

    python ablation/uncertainty_weighted.py sweep        # val grid (3 procs)
    python ablation/uncertainty_weighted.py select       # pick config per brief rule
    python ablation/uncertainty_weighted.py final        # val/test/pediatric once, masks, AUROC
"""
import argparse, itertools, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

OUT = os.path.join(common.STUDY, "ablation", "uncertainty_weighted")
RADII = (2, 3, 5)
PAD = max(RADII) + 2
MEMBERS = ("airmorph", "medpseg", "tfenet")

GRID_A = {"r": RADII, "norm": ("ball", "union"), "eps": (0.02, 0.1, 0.3),
          "prior": (0, 1), "t": (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)}
GRID_C = {"sigma": (0.0, 0.5, 1.0, 2.0),
          "t": (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)}
GRID_D = {"r": RADII, "eps": (0.02, 0.1, 0.3), "prior": (0, 1),
          "t": (0.2, 0.3, 0.4, 0.6)}        # norm = ball only


# ----------------------------------------------------------------- helpers

def ball(r):
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return ((z * z + y * y + x * x) <= r * r).astype(np.float32)


def crop_case(c, pad=PAD):
    """Crop every volume to the bbox of (M | gt) padded by `pad`."""
    reg = c["M"] | (c["gt"] > 0)
    sl = ndimage.find_objects(reg.astype(np.uint8))[0]
    sl = tuple(slice(max(0, s.start - pad), min(n, s.stop + pad)) for s, n in zip(sl, reg.shape))
    cc = {k: c[k][sl] for k in ("gt", "gt_parsing", "gt_skeleton", "pattern", "M",
                                "airmorph", "medpseg", "tfenet", "branch_map")}
    cc.update(spacing=c["spacing"], tab=c["tab"], case=c["case"], split=c["split"],
              slices=sl, full_shape=c["gt"].shape, n_full=int(np.prod(c["gt"].shape)))
    return cc


def score_cropped(pred, cc, cldice=False):
    s = common.score(pred, cc, cldice=cldice)
    fp = int(((pred > 0) & (cc["gt"] == 0)).sum())
    s["FPR"] = round(fp * 100 / (cc["n_full"] - int(cc["gt"].sum()) + 1e-5), 3)
    return s


def global_prior(table):
    """Precision on val of voxels only one member marks: (airmorph, medpseg, tfenet)."""
    n, tp = table["n"], table["tp"]
    idx = np.arange(24); a5 = idx // 4; md = (idx // 2) % 2; tf = idx % 2
    only = {"airmorph": (a5 >= 1) & (md == 0) & (tf == 0),
            "medpseg": (a5 == 0) & (md == 1) & (tf == 0),
            "tfenet": (a5 == 0) & (md == 0) & (tf == 1)}
    return np.array([tp[only[m]].sum() / max(n[only[m]].sum(), 1) for m in MEMBERS], np.float32)


def local_disagreement(cc, r):
    """U_m for the three members under both normalisations: dict norm -> (3, ...) float32."""
    ms = [cc[m] for m in MEMBERS]
    votes = ms[0].astype(np.uint8) + ms[1] + ms[2]
    maj = votes >= 2
    union = (votes >= 1).astype(np.float32)
    k = ball(r); nb = float(k.sum())
    den_u = np.clip(fftconvolve(union, k, mode="same"), 0, None)
    U = {"ball": np.empty((3,) + maj.shape, np.float32), "union": np.empty((3,) + maj.shape, np.float32)}
    for i, m in enumerate(ms):
        d = (m ^ maj).astype(np.float32)
        cnt = np.clip(fftconvolve(d, k, mode="same"), 0, None)
        U["ball"][i] = cnt / nb
        with np.errstate(divide="ignore", invalid="ignore"):
            U["union"][i] = np.where(den_u > 0.5, cnt / np.maximum(den_u, 0.5), 0.0)
    return U


def fuse(cc, U, eps, prior):
    """Uncertainty-weighted soft vote p(x) in [0, 1] (float32)."""
    num = np.zeros(cc["M"].shape, np.float32); den = np.zeros_like(num)
    for i, m in enumerate(MEMBERS):
        w = prior[i] / (U[i] + eps)
        den += w; num += w * cc[m]
    return num / den


def finish(keep, cc):
    return common.keep_seeded_component(keep & cc["M"], cc["airmorph"])


def smoothed_confidence(cc, table, sigma):
    conf = common.confidence_from_pattern(cc["pattern"], "calibrated", table)
    if sigma <= 0:
        return conf
    M = cc["M"].astype(np.float32)
    num = ndimage.gaussian_filter(conf * M, sigma); den = ndimage.gaussian_filter(M, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0).astype(np.float32) * cc["M"]


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def embed(mask_crop, cc):
    out = np.zeros(cc["full_shape"], np.uint8); out[cc["slices"]] = mask_crop
    return out


# ----------------------------------------------------------------- sweep

def sweep_case(args):
    split, cid = args
    t0 = time.time()
    c = common.load_cached(split, cid); cc = crop_case(c); del c
    table = common.pattern_table("val"); g = global_prior(table); ones = np.ones(3, np.float32)
    cs = cc["tab"]["conf_skel"].to_numpy(np.float32).copy(); cs[0] = 1.0
    cs_map = cs[cc["branch_map"]]
    rows = []
    for r in RADII:
        U = local_disagreement(cc, r)
        for norm in GRID_A["norm"]:
            for eps, prior in itertools.product(GRID_A["eps"], GRID_A["prior"]):
                p = fuse(cc, U[norm], eps, g if prior else ones)
                for t in GRID_A["t"]:
                    s = score_cropped(finish(p >= t, cc), cc)
                    rows.append(dict(method="a" if not prior else "b", r=r, norm=norm, eps=eps,
                                     prior=prior, t=t, sigma=np.nan, case=cid, **s))
                if norm == "ball":
                    for t in GRID_D["t"]:
                        s = score_cropped(finish(p >= t * cs_map, cc), cc)
                        rows.append(dict(method="d", r=r, norm=norm, eps=eps, prior=prior, t=t,
                                         sigma=np.nan, case=cid, **s))
        del U
    for sigma in GRID_C["sigma"]:
        conf = smoothed_confidence(cc, table, sigma)
        for t in GRID_C["t"]:
            s = score_cropped(finish(conf >= t, cc), cc)
            rows.append(dict(method="c", r=np.nan, norm="", eps=np.nan, prior=0, t=t, sigma=sigma,
                             case=cid, **s))
    s = score_cropped(cc["M"].astype(np.uint8), cc)
    rows.append(dict(method="union", r=np.nan, norm="", eps=np.nan, prior=0, t=np.nan, sigma=np.nan,
                     case=cid, **s))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sweep_cases", f"{cid}.csv"), index=False)
    print(f"{cid}: {len(rows)} rows in {time.time() - t0:.0f} s", flush=True)
    return cid


def run_sweep(nproc=3):
    os.makedirs(os.path.join(OUT, "sweep_cases"), exist_ok=True)
    todo = [("val", cid) for cid in common.cases("val")
            if not os.path.exists(os.path.join(OUT, "sweep_cases", f"{cid}.csv"))]
    with Pool(nproc) as pool:
        for _ in pool.imap_unordered(sweep_case, todo):
            pass
    frames = [pd.read_csv(os.path.join(OUT, "sweep_cases", f"{cid}.csv")) for cid in common.cases("val")]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(OUT, "sweep_val_percase.csv"), index=False)
    return df


def fine_case(args):
    """Stage 2: finer t grid around the best coarse configs (list of dicts with method,r,norm,eps,prior,sigma,ts)."""
    split, cid, cfgs = args
    c = common.load_cached(split, cid); cc = crop_case(c); del c
    table = common.pattern_table("val"); g = global_prior(table); ones = np.ones(3, np.float32)
    cs = cc["tab"]["conf_skel"].to_numpy(np.float32).copy(); cs[0] = 1.0
    cs_map = cs[cc["branch_map"]]
    rows = []; Ucache = {}
    for cfg in cfgs:
        if cfg["method"] == "c":
            conf = smoothed_confidence(cc, table, float(cfg["sigma"]))
            for t in cfg["ts"]:
                s = score_cropped(finish(conf >= t, cc), cc)
                rows.append(dict(method="c", r=np.nan, norm="", eps=np.nan, prior=0, t=t, sigma=cfg["sigma"], case=cid, **s))
            continue
        r = int(cfg["r"])
        if r not in Ucache:
            Ucache[r] = local_disagreement(cc, r)
        p = fuse(cc, Ucache[r][cfg["norm"]], float(cfg["eps"]), g if int(cfg["prior"]) else ones)
        for t in cfg["ts"]:
            keep = p >= (t * cs_map if cfg["method"] == "d" else t)
            s = score_cropped(finish(keep, cc), cc)
            rows.append(dict(method=cfg["method"], r=r, norm=cfg["norm"], eps=cfg["eps"], prior=cfg["prior"],
                             t=round(t, 3), sigma=np.nan, case=cid, **s))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fine_cases", f"{cid}.csv"), index=False)
    print(f"fine {cid}: {len(rows)} rows", flush=True)
    return cid


def run_fine(nproc=3, top=2):
    """Pick the top `top` admissible coarse configs per method and sweep t in 0.01 steps around them."""
    df = pd.read_csv(os.path.join(OUT, "sweep_val_percase.csv"))
    s, ok, u = select(df)
    cfgs = []
    for m in ("a", "b", "d"):          # (c) with sigma=0 is a discrete LUT: a finer t changes nothing
        for _, row in ok[ok.method == m].head(top).iterrows():
            step = 0.02
            ts = [round(row["t"] + k * step, 3) for k in range(-5, 6) if row["t"] + k * step > 0]
            cfgs.append(dict(method=m, r=row["r"], norm=row["norm"], eps=row["eps"], prior=int(row["prior"]),
                             sigma=row["sigma"], ts=ts))
    print(f"fine stage: {len(cfgs)} configs x {sum(len(c['ts']) for c in cfgs)} rows", flush=True)
    os.makedirs(os.path.join(OUT, "fine_cases"), exist_ok=True)
    with Pool(nproc) as pool:
        list(pool.imap_unordered(fine_case, [("val", cid, cfgs) for cid in common.cases("val")]))
    fine = pd.concat([pd.read_csv(os.path.join(OUT, "fine_cases", f"{cid}.csv")) for cid in common.cases("val")])
    fine.to_csv(os.path.join(OUT, "fine_val_percase.csv"), index=False)
    both = pd.concat([df, fine], ignore_index=True).drop_duplicates(subset=KEYS + ["case"])
    both.to_csv(os.path.join(OUT, "sweep_val_percase.csv"), index=False)
    return both


KEYS = ["method", "r", "norm", "eps", "prior", "t", "sigma"]
METRICS = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "FPR"]


def summarise(df):
    g = df.fillna({"r": -1, "eps": -1, "t": -1, "sigma": -1, "norm": ""}).groupby(KEYS)
    s = g[METRICS].mean().reset_index()
    s["n"] = g.size().values
    return s


def select(df, bnd_loss=1.0):
    s = summarise(df)
    u = s[s.method == "union"].iloc[0]
    s["dBND"] = s["BND"] - u["BND"]; s["dDSC"] = s["DSC"] - u["DSC"]
    ok = s[(s.dBND >= -bnd_loss) & (s.method != "union")].sort_values("DSC", ascending=False)
    return s, ok, u


def config_from_row(row):
    return {k: (None if (isinstance(row[k], float) and (np.isnan(row[k]) or row[k] == -1)) else row[k])
            for k in KEYS}


# ----------------------------------------------------------------- final

def apply_config(cc, cfg, table):
    """Returns (mask_crop uint8, uncertainty_crop float32 in M (higher = less sure))."""
    g = global_prior(table); ones = np.ones(3, np.float32)
    if cfg["method"] in ("a", "b", "d"):
        U = local_disagreement(cc, int(cfg["r"]))[cfg["norm"]]
        p = fuse(cc, U, float(cfg["eps"]), g if int(cfg["prior"]) else ones)
        if cfg["method"] == "d":
            cs = cc["tab"]["conf_skel"].to_numpy(np.float32).copy(); cs[0] = 1.0
            keep = p >= float(cfg["t"]) * cs[cc["branch_map"]]
        else:
            keep = p >= float(cfg["t"])
        unc = (1.0 - p) * cc["M"]
    elif cfg["method"] == "c":
        conf = smoothed_confidence(cc, table, float(cfg["sigma"]))
        keep = conf >= float(cfg["t"])
        unc = entropy(conf) * cc["M"]           # reported; ranking uses 1-conf below
        unc = (1.0 - conf) * cc["M"]
    else:
        raise ValueError(cfg)
    return finish(keep, cc), unc.astype(np.float32)


def final_case(args):
    split, cid, cfg, save = args
    t0 = time.time()
    c = common.load_cached(split, cid); cc = crop_case(c); del c
    table = common.pattern_table("val")
    t1 = time.time()
    mask, unc = apply_config(cc, cfg, table)
    t_method = time.time() - t1
    s = score_cropped(mask, cc, cldice=True)
    s.update(case=cid, split=split, cpu_s_method=round(t_method, 2))
    if save:
        os.makedirs(os.path.join(OUT, "masks", split), exist_ok=True)
        np.savez_compressed(os.path.join(OUT, "masks", split, f"ATM26_{cid}.npz"), mask=embed(mask, cc))
    # uncertainty scores inside M for the FP-AUROC: ours and the calibrated confidence
    inM = cc["M"]
    fp = (inM & (cc["gt"] == 0))[inM]
    conf = common.confidence_from_pattern(cc["pattern"], "calibrated", table)
    np.savez_compressed(os.path.join(OUT, "unc_cases", f"{split}_{cid}.npz"),
                        unc=unc[inM], calib_unc=(1.0 - conf)[inM], fp=fp)
    s["AUROC_fp_ours"] = auroc(unc[inM], fp); s["AUROC_fp_calibrated"] = auroc((1.0 - conf)[inM], fp)
    print(f"{split}/{cid}: {s} ({time.time() - t0:.0f} s)", flush=True)
    return s


def auroc(scores, pos):
    """Rank AUROC with tie handling (higher score = more likely positive)."""
    from scipy.stats import rankdata
    pos = np.asarray(pos, bool)
    if pos.all() or not pos.any():
        return float("nan")
    rk = rankdata(np.asarray(scores, np.float64))
    P = pos.sum(); N = len(pos) - P
    return float((rk[pos].sum() - P * (P + 1) / 2) / (P * N))


def pooled_auroc(split):
    fs = sorted(f for f in os.listdir(os.path.join(OUT, "unc_cases")) if f.startswith(split + "_"))
    U, C, F = [], [], []
    for f in fs:
        z = np.load(os.path.join(OUT, "unc_cases", f)); U.append(z["unc"]); C.append(z["calib_unc"]); F.append(z["fp"])
    U, C, F = np.concatenate(U), np.concatenate(C), np.concatenate(F)
    return {"pooled_AUROC_fp_ours": auroc(U, F), "pooled_AUROC_fp_calibrated": auroc(C, F),
            "n_voxels": int(len(F)), "n_fp": int(F.sum())}


def run_final(cfg, nproc=3):
    os.makedirs(os.path.join(OUT, "unc_cases"), exist_ok=True)
    out = {}
    for split in ("val", "test", "pediatric"):
        jobs = [(split, cid, cfg, split == "test") for cid in common.cases(split)]
        with Pool(nproc) as pool:
            rows = list(pool.imap_unordered(final_case, jobs))
        df = pd.DataFrame(rows).sort_values("case")
        df.to_csv(os.path.join(OUT, f"final_{split}_percase.csv"), index=False)
        out[split] = {"mean": df[METRICS + ["cpu_s_method", "AUROC_fp_ours", "AUROC_fp_calibrated"]].mean().round(3).to_dict(),
                      **pooled_auroc(split)}
        print(split, out[split], flush=True)
    with open(os.path.join(OUT, "final_summary.json"), "w") as fh:
        json.dump({"config": cfg, "results": out}, fh, indent=1, default=float)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "fine", "select", "final"])
    ap.add_argument("--nproc", type=int, default=3)
    ap.add_argument("--config", default=os.path.join(OUT, "selected_config.json"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.stage == "sweep":
        df = run_sweep(a.nproc)
    if a.stage == "fine":
        run_fine(a.nproc)
    if a.stage in ("sweep", "fine", "select"):
        df = pd.read_csv(os.path.join(OUT, "sweep_val_percase.csv"))
        s, ok, u = select(df)
        s.to_csv(os.path.join(OUT, "sweep_val_summary.csv"), index=False)
        print("union on val:", u[METRICS].round(2).to_dict())
        print("best admissible per method (BND loss <= 1.0):")
        for m in ("a", "b", "c", "d"):
            sub = ok[ok.method == m]
            if len(sub):
                print(sub.head(3).round(3).to_string())
        best = ok.iloc[0]
        cfg = config_from_row(best)
        with open(a.config, "w") as fh:
            json.dump({"config": cfg, "val_summary": best.round(4).to_dict()}, fh, indent=1, default=float)
        print("selected:", cfg)
    if a.stage == "final":
        with open(a.config) as fh:
            cfg = json.load(fh)["config"]
        run_final(cfg, a.nproc)
