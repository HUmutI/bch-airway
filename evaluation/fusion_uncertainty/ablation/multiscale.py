#!/usr/bin/env python
"""Ablation 7: multi-scale ensemble, training-free scale-space approximation.

The PDF's idea is an ensemble whose members segment at several resolutions and
whose votes are fused scale by scale. No member was retrained here (no training
budget), so the multi-scale axis is built from the existing seven binary
members instead: the calibrated confidence map P(airway | vote pattern) (and
the plain 7-vote fraction) is taken as the ensemble's evidence, that evidence
is observed at several spatial scales (Gaussian scale space, sigma in
{0, 0.7, 1.5, 3} in-plane voxels, anisotropy corrected by spacing), and the
decision at each voxel is taken at the scale matching the local airway calibre
(branch radius in mm from the branch tree of the union mask):

  (a) scale-space consensus  sigma(x) = snap(alpha * r_mm(x) / dx);
      mask = M & (S_{sigma(x)}(x) >= t) for thick voxels (r_mm > R_thin), the
      raw union for thin ones (R_thin = 0: threshold everywhere);
  (b) calibre-adaptive threshold  t(x) = t0 + t1 * r_mm(x) applied to the raw or
      scale-fused map;
  (c) two-level  decide at 2x block-downsampled resolution where r_mm > R, at
      full resolution (union or thr) elsewhere.

Every variant ends with common.keep_seeded_component(mask, airmorph) and never
grows outside the union M (all masks are subsets of M).

    python multiscale.py --stage sweep1 --split val            # (a), (b), (c) jointly
    python multiscale.py --stage final --split test --config chosen.json
"""
import argparse, json, os, sys, time
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402

OUT = os.path.join(common.STUDY, "ablation", "multiscale")
SIGMAS = (0.0, 0.7, 1.5, 3.0)
PAD = 14
CFG = None


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "log.txt"), "a") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


# ------------------------------------------------------------------ per-case preparation

def snap_sigma(sig):
    """Nearest grid scale for a continuous sigma (in-plane voxels)."""
    edges = [(SIGMAS[i] + SIGMAS[i + 1]) / 2 for i in range(len(SIGMAS) - 1)]
    return np.digitize(sig, edges)          # index into SIGMAS


def prepare(c, table, maps=("cal", "seven")):
    """Crop to the union's bounding box and precompute the scale-space of each
    evidence map (unnormalised and normalised convolution) plus the radius map."""
    M = c["M"]
    sl = ndimage.find_objects(M.astype(np.uint8))[0]
    sl = tuple(slice(max(s.start - PAD, 0), min(s.stop + PAD, n)) for s, n in zip(sl, M.shape))
    sp = c["spacing"]; dx = float(sp[2])
    Mc = M[sl]
    pat = c["pattern"][sl]
    r = c["tab"]["r_mean_mm"].to_numpy(np.float32).copy(); r[0] = 0
    rmap = r[c["branch_map"][sl]] * Mc                   # local calibre, mm
    ev = {}
    if "cal" in maps:
        ev["cal"] = common.confidence_from_pattern(pat, "calibrated", table) * Mc
    if "seven" in maps:
        ev["seven"] = common.confidence_from_pattern(pat, "seven") * Mc
    Mf = Mc.astype(np.float32)
    smooth = {}
    for s in SIGMAS[1:]:
        sig = tuple(s * dx / float(a) for a in sp)       # isotropic in mm
        gM = ndimage.gaussian_filter(Mf, sig)
        for k, v in ev.items():
            g = ndimage.gaussian_filter(v, sig)
            smooth[(k, 0, s)] = g
            with np.errstate(invalid="ignore", divide="ignore"):
                smooth[(k, 1, s)] = np.where(gM > 1e-3, g / np.maximum(gM, 1e-3), 0).astype(np.float32)
    for k, v in ev.items():
        smooth[(k, 0, 0.0)] = v; smooth[(k, 1, 0.0)] = v
    return {"sl": sl, "M": Mc, "rmap": rmap, "ev": ev, "smooth": smooth, "dx": dx,
            "shape": M.shape, "seed": c["airmorph"]}


def finish(prep, keep):
    """Seeded-component rule on the crop (mask is a subset of M, so no component
    lies outside it), pasted back onto the full grid."""
    sub = common.keep_seeded_component((keep & prep["M"]).astype(np.uint8), prep["seed"][prep["sl"]])
    full = np.zeros(prep["shape"], np.uint8)
    full[prep["sl"]] = sub
    return full


def scale_fused_map(prep, emap, norm, alpha):
    """S_{sigma(x)}(x): the evidence observed at the scale set by the local calibre."""
    if alpha == 0:
        return prep["ev"][emap], np.zeros(prep["M"].shape, np.int8)
    idx = snap_sigma(alpha * prep["rmap"] / prep["dx"])
    F = np.zeros(prep["M"].shape, np.float32)
    for i, s in enumerate(SIGMAS):
        m = idx == i
        if m.any():
            F[m] = prep["smooth"][(emap, norm, s)][m]
    return F, idx


def fuse_scalespace(prep, emap, norm, alpha, R_thin, t):
    """Thick voxels (branch radius > R_thin) are decided from the evidence at
    scale sigma(x) = snap(alpha r / dx); thin ones keep the raw union."""
    F, idx = scale_fused_map(prep, emap, norm, alpha)
    keep = F >= t
    if R_thin > 0:
        keep |= prep["rmap"] <= R_thin
    return finish(prep, keep)


def fuse_adaptive(prep, emap, norm, alpha, t0, t1):
    F, _ = scale_fused_map(prep, emap, norm, alpha)
    tx = np.clip(t0 + t1 * prep["rmap"], 0.0, 0.95)
    return finish(prep, F >= tx)


def block_mean_2x(v):
    z, y, x = v.shape
    pz, py, px = z % 2, y % 2, x % 2
    if pz or py or px:
        v = np.pad(v, ((0, pz), (0, py), (0, px)))
    b = v.reshape(v.shape[0] // 2, 2, v.shape[1] // 2, 2, v.shape[2] // 2, 2).mean(axis=(1, 3, 5))
    return b, (z, y, x)


def upsample_2x(b, shape):
    u = np.repeat(np.repeat(np.repeat(b, 2, 0), 2, 1), 2, 2)
    return u[:shape[0], :shape[1], :shape[2]]


def fuse_twolevel(prep, emap, norm, R, fine_rule, t):
    v = prep["ev"][emap]
    cb, shp = block_mean_2x(v)
    if norm:
        mb, _ = block_mean_2x(prep["M"].astype(np.float32))
        with np.errstate(invalid="ignore", divide="ignore"):
            cb = np.where(mb > 0, cb / np.maximum(mb, 1e-3), 0)
    coarse_keep = upsample_2x(cb >= t, shp)
    thick = prep["rmap"] > R
    fine_keep = prep["M"] if fine_rule == "union" else (v >= t)
    return finish(prep, np.where(thick, coarse_keep, fine_keep))


# ------------------------------------------------------------------ grids

def grid(stage, best=None):
    """One joint sweep (the machine is shared; ~12 s per config and case under
    load, so the grid is kept to 34 configurations)."""
    if stage == "sweep1":
        T = {0: (0.25, 0.35, 0.45), 1: (0.4, 0.5, 0.6)}       # unnormalised smoothing halves at the boundary
        # (a) scale-space consensus
        g = [dict(variant="a", emap="cal", norm=n, alpha=0.3, R_thin=R, t=t)
             for n in (0, 1) for R in (0.0, 1.5) for t in T[n]]
        g += [dict(variant="a", emap="cal", norm=1, alpha=0.6, R_thin=1.5, t=t) for t in T[1]]
        g += [dict(variant="a", emap="seven", norm=1, alpha=0.3, R_thin=1.5, t=0.5)]
        # single-scale baselines (plain threshold of the raw map)
        g += [dict(variant="a", emap="cal", norm=0, alpha=0.0, R_thin=0.0, t=t) for t in (0.3, 0.4, 0.5)]
        g += [dict(variant="a", emap="seven", norm=0, alpha=0.0, R_thin=0.0, t=0.5)]
        # (b) calibre-adaptive threshold
        g += [dict(variant="b", emap="cal", norm=1, alpha=0.3, t0=t0, t1=t1) for t0 in (0.2, 0.3) for t1 in (0.03, 0.06, 0.1)]
        g += [dict(variant="b", emap="cal", norm=0, alpha=0.0, t0=0.2, t1=t1) for t1 in (0.06, 0.1)]
        # (c) two-level
        g += [dict(variant="c", emap="cal", norm=n, R=1.5, fine_rule="union", t=t) for n in (0, 1) for t in T[n]]
        return g
    raise ValueError(stage)


def apply(prep, cfg):
    if cfg["variant"] == "a":
        return fuse_scalespace(prep, cfg["emap"], cfg["norm"], cfg["alpha"], cfg["R_thin"], cfg["t"])
    if cfg["variant"] == "b":
        return fuse_adaptive(prep, cfg["emap"], cfg["norm"], cfg["alpha"], cfg["t0"], cfg["t1"])
    if cfg["variant"] == "c":
        return fuse_twolevel(prep, cfg["emap"], cfg["norm"], cfg["R"], cfg["fine_rule"], cfg["t"])
    raise ValueError(cfg)


def cfg_key(cfg):
    return json.dumps(cfg, sort_keys=True)


# ------------------------------------------------------------------ workers

def run_sweep_case(cid):
    split, table, configs = CFG
    t0 = time.time()
    c = common.load_cached(split, cid)
    prep = prepare(c, table)
    tprep = time.time() - t0
    rows = [{"case": cid, "cfg": "union", **common.score(c["M"].astype(np.uint8), c, cldice=False)}]
    for cfg in configs:
        R = apply(prep, cfg)
        rows.append({"case": cid, "cfg": cfg_key(cfg), **common.score(R, c, cldice=False)})
    log(f"{split} {cid}: {len(configs)} cfgs, prep {tprep:.0f}s, total {time.time()-t0:.0f}s, crop {prep['M'].shape}")
    return rows


def run_final_case(cid):
    split, table, cfg = CFG
    t0 = time.time()
    c = common.load_cached(split, cid)
    t1 = time.time()
    prep = prepare(c, table, maps=(cfg["emap"],))
    R = apply(prep, cfg)
    cpu = time.time() - t1
    rows = [{"split": split, "case": cid, "strategy": "multiscale", "cpu_s": cpu, **common.score(R, c)},
            {"split": split, "case": cid, "strategy": "union_seeded", "cpu_s": np.nan, **common.score(c["M"].astype(np.uint8), c)}]
    ref = os.path.join(common.STUDY, "masks", "topofuse", split, f"ATM26_{cid}.npz")
    if os.path.exists(ref):
        rm = np.load(ref)["mask"].astype(np.uint8)
        rows.append({"split": split, "case": cid, "strategy": "refined", "cpu_s": np.nan, **common.score(rm, c)})
    d = os.path.join(OUT, "masks", split); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"ATM26_{cid}.npz"), mask=R.astype(np.uint8))
    log(f"{split} {cid}: DSC union {rows[1]['DSC']} -> ms {rows[0]['DSC']}, BND {rows[1]['BND']} -> {rows[0]['BND']}, "
        f"fuse {cpu:.1f}s, total {time.time()-t0:.0f}s")
    return rows


def select(df, bnd_slack=1.0):
    """Brief's rule: max mean DSC subject to mean BND loss <= slack vs the union."""
    g = df.groupby("cfg")[["DSC", "TLD", "BND", "Sensitivity", "Precision"]].mean()
    g["n"] = df.groupby("cfg").size()
    u = g.loc["union"]
    g["dBND"] = g["BND"] - u["BND"]; g["dDSC"] = g["DSC"] - u["DSC"]
    ok = g[(g["dBND"] >= -bnd_slack) & (g.index != "union")]
    best = ok["DSC"].idxmax() if len(ok) else None
    return g.sort_values("DSC", ascending=False), best


def tables():
    """Markdown tables for RESULTS.md from the final per-case CSVs."""
    cols = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision"]
    out = ["| split | method | DSC | clDice | TLD | BND | Sens | Prec |", "|---|---|---|---|---|---|---|---|"]
    diffs = []
    for split in ("val", "test", "pediatric"):
        f = os.path.join(OUT, f"final_{split}_percase.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        for name, lab in (("union_seeded", "union, seeded"), ("refined", "union, refined (topofuse)"), ("multiscale", "multiscale (this)")):
            d = df[df.strategy == name]
            if len(d):
                m = d[cols].mean()
                out.append(f"| {split} | {lab} | " + " | ".join(f"{m[c]:.2f}" for c in cols) + " |")
        ms = df[df.strategy == "multiscale"].set_index("case")
        for ref in ("union_seeded", "refined"):
            r = df[df.strategy == ref].set_index("case")
            if not len(r):
                continue
            d = ms[cols] - r.loc[ms.index, cols]
            diffs.append(f"\n{split}: multiscale minus {ref} (n={len(d)}), mean / improved cases:")
            diffs.append("| " + " | ".join(cols) + " |"); diffs.append("|" + "---|" * len(cols))
            diffs.append("| " + " | ".join(f"{d[c].mean():+.2f} ({int((d[c] > 0).sum())}/{len(d)})" for c in cols) + " |")
        if "cpu_s" in ms:
            diffs.append(f"{split}: fusion CPU s per scan: mean {ms['cpu_s'].mean():.1f}, max {ms['cpu_s'].max():.1f}")
    return "\n".join(out) + "\n" + "\n".join(diffs)


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("sweep1", "final", "report"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--config", default=None, help="json of the chosen cfg (final) or best (a) cfg (sweep_b)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.stage == "report":
        print(tables()); return
    table = common.pattern_table("val")
    ids = common.cases(a.split)
    if a.limit:
        ids = ids[:a.limit]
    if a.stage.startswith("sweep"):
        best = json.load(open(a.config)) if a.config else None
        configs = grid(a.stage, best)
        log(f"== {a.stage} on {a.split}: {len(configs)} configs x {len(ids)} cases")
        CFG = (a.split, table, configs)
        with Pool(a.workers) as p:
            rows = [r for rs in p.imap_unordered(run_sweep_case, ids) for r in rs]
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, f"{a.stage}_{a.split}_percase.csv"), index=False)
        g, best = select(df)
        g.round(3).to_csv(os.path.join(OUT, f"{a.stage}_{a.split}_summary.csv"))
        pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 120)
        log(g.round(2).to_string())
        log(f"selected ({a.stage}): {best}")
        if best:
            json.dump(json.loads(best), open(os.path.join(OUT, f"{a.stage}_best.json"), "w"), indent=1)
    else:
        cfg = json.load(open(a.config))
        log(f"== final on {a.split} with {cfg}")
        CFG = (a.split, table, cfg)
        with Pool(a.workers) as p:
            rows = [r for rs in p.imap_unordered(run_final_case, ids) for r in rs]
        df = pd.DataFrame(rows).sort_values(["strategy", "case"])
        df.to_csv(os.path.join(OUT, f"final_{a.split}_percase.csv"), index=False)
        g = df.groupby("strategy")[["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "FPR", "cpu_s"]].mean().round(2)
        pd.set_option("display.width", 250)
        log(g.to_string())


if __name__ == "__main__":
    main()
