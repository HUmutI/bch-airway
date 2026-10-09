"""Ablation: learned voxel-wise stacking (meta-learner) over the seven member votes.

Pipeline (two interpreters, see BRIEF.md):
  airmorph env : python stacking.py features --split val|test|pediatric   (3 procs)
  own venv     : python stacking.py train                                  (sklearn + lightgbm)
  airmorph env : python stacking.py select                                 (val sweep, rule: max DSC s.t. BND loss <= 1 vs union)
  airmorph env : python stacking.py evaluate --split val|test|pediatric    (final scores with clDice, masks)

Candidate region = union mask M (seeded union of the three models) dilated by one
voxel (26-neighbourhood).  Per candidate voxel the features are: the 7 binary member
votes, the AirMorph vote count, 3x3x3 and 5x5x5 mean vote of every member, distance to
the union skeleton (mm, 0 outside M), signed EDT distance to the M boundary (mm, >0
inside), CT intensity in HU clipped to [-1024, 400] and its 3x3x3 mean, and the
calibrated pattern confidence P(airway | pattern) estimated on val.  Label = GT voxel.
The meta-learners (logistic regression; LightGBM gradient-boosted trees at three
sizes) are trained on class-balanced subsamples of the val voxels; model size and the
probability threshold are chosen by 5-fold cross-validation by case inside val.  The
final mask is the thresholded probability map restricted to the candidate region and
passed through common.keep_seeded_component with the AirMorph mask as seed.
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = "/fileserver/external/quin/ege/fusion_uncertainty_run/ablation/stacking"
TEST_IMG = "/fileserver/external/quin/ege/fusion_uncertainty_run/test_images"
REF_MASKS = "/fileserver/external/quin/ege/fusion_uncertainty_run/masks/topofuse"
MEMBERS = ["member_model1", "member_model2", "member_model3", "member_wingsnet",
           "member_break", "medpseg", "tfenet"]
FEATURES = (MEMBERS + ["airmorph_votes"]
            + [f"{m}_mean3" for m in MEMBERS] + [f"{m}_mean5" for m in MEMBERS]
            + ["dist_skel_mm", "sdist_boundary_mm", "hu", "hu_mean3", "conf_calibrated"])
CONFIGS = ["lr", "gbt_small", "gbt_medium", "gbt_large"]
# GBT size is chosen by CV voxel log-loss (gbt_small won, see cv_voxel_diagnostics.csv); the
# threshold sweep on full val masks (the expensive part, ~11 s per mask under load) is run for
# the linear learner and that GBT only.  The grid is dense at the low end because the
# selection rule (BND loss <= 1 vs union) forces a permissive operating point.
SWEEP_CONFIGS = ["lr", "gbt_small"]
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(OUT, "LOG.txt"), "a") as fh:
        fh.write(line + "\n")


# ------------------------------------------------------------------ features (airmorph env)

def ct_path(split, cid):
    import common
    if split == "val":
        return os.path.join(common.VAL_RUN, "data/ATM26/imagesTr", f"{cid}_0000.mha")
    if split == "test":
        return os.path.join(TEST_IMG, f"{cid}_0000.mha")
    return os.path.join(common.ped_case_dir(cid), "image.nii.gz")


def read_ct(split, cid, shape):
    import SimpleITK as sitk, common
    img = sitk.ReadImage(ct_path(split, cid))
    if split == "pediatric":   # members live on the airway_bin grid; check the image is the same grid
        ref = sitk.ReadImage(os.path.join(common.ped_case_dir(cid), "airway_bin.nii.gz"))
        assert img.GetSize() == ref.GetSize(), (cid, img.GetSize(), ref.GetSize())
        assert np.allclose(img.GetDirection(), ref.GetDirection(), atol=1e-3), (cid, "direction")
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    assert arr.shape == shape, (cid, arr.shape, shape)
    return arr


def crop_box(mask, margin):
    nz = np.nonzero(mask)
    lo = [max(int(a.min()) - margin, 0) for a in nz]
    hi = [min(int(a.max()) + margin + 1, s) for a, s in zip(nz, mask.shape)]
    return tuple(slice(l, h) for l, h in zip(lo, hi))


def extract_case(args):
    split, cid = args
    out = os.path.join(OUT, "features", split, f"{cid}.npz")
    if os.path.exists(out):
        return cid, "cached"
    import common
    from scipy import ndimage
    t0 = time.time()
    c = common.load_case(split, cid)
    cc = common.load_cached(split, cid)
    table = common.pattern_table("val")
    M = cc["M"]
    cand_full = ndimage.binary_dilation(M, structure=np.ones((3, 3, 3), bool))
    box = crop_box(cand_full, 3)
    cand = cand_full[box]
    idx = np.flatnonzero(cand).astype(np.int32)
    sp = np.asarray(c["spacing"], float)
    hu = np.clip(read_ct(split, cid, c["gt"].shape)[box], -1024, 400)
    Mc = M[box]
    cols = {}
    votes = np.zeros(Mc.shape, np.float32)
    for m in MEMBERS:
        v = c[m][box].astype(np.float32)
        cols[m] = v
        if m.startswith("member_"):
            votes += v
        cols[f"{m}_mean3"] = ndimage.uniform_filter(v, 3, mode="constant")
        cols[f"{m}_mean5"] = ndimage.uniform_filter(v, 5, mode="constant")
    cols["airmorph_votes"] = votes
    cols["dist_skel_mm"] = cc["dist"][box]
    inside = ndimage.distance_transform_edt(Mc, sampling=sp)
    outside = ndimage.distance_transform_edt(~Mc, sampling=sp)
    cols["sdist_boundary_mm"] = (inside - outside).astype(np.float32)
    cols["hu"] = hu
    cols["hu_mean3"] = ndimage.uniform_filter(hu, 3, mode="constant")
    cols["conf_calibrated"] = common.confidence_from_pattern(cc["pattern"][box], "calibrated", table)
    X = np.stack([cols[f].ravel()[idx] for f in FEATURES], axis=1).astype(np.float32)
    y = c["gt"][box].ravel()[idx].astype(np.uint8)
    gt_total = int(c["gt"].sum())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, X=X, y=y, idx=idx, box=np.array([[s.start, s.stop] for s in box]),
             shape=np.array(c["gt"].shape), in_M=Mc.ravel()[idx],
             gt_total=gt_total, n_union=int(M.sum()), extract_sec=time.time() - t0)
    return cid, f"{len(idx)} cand, {int(y.sum())}/{gt_total} GT reachable, {time.time()-t0:.0f}s"


def cmd_features(a):
    import common
    from multiprocessing import Pool
    jobs = [(a.split, cid) for cid in common.cases(a.split)]
    with Pool(a.procs) as pool:
        for cid, msg in pool.imap_unordered(extract_case, jobs):
            log(f"features {a.split} {cid}: {msg}")
    with open(os.path.join(OUT, "features", "feature_names.json"), "w") as fh:
        json.dump(FEATURES, fh, indent=1)


# ------------------------------------------------------------------ training (own venv)

def _balanced_sample(X, y, n_per_class, rng):
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    pos = rng.choice(pos, min(len(pos), n_per_class), replace=False)
    neg = rng.choice(neg, min(len(neg), n_per_class), replace=False)
    sel = np.concatenate([pos, neg]); rng.shuffle(sel)
    return X[sel], y[sel]


def _make_model(name, n_jobs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    # (num_leaves, n_estimators, learning_rate); prediction cost on ~0.5M voxels/case bounds the size
    size = {"gbt_small": (15, 200, 0.05), "gbt_medium": (63, 300, 0.05), "gbt_large": (127, 300, 0.10)}[name]
    return lgb.LGBMClassifier(num_leaves=size[0], n_estimators=size[1], learning_rate=size[2],
                              min_child_samples=200, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=n_jobs, verbose=-1)


def _load_feats(split, cid):
    with np.load(os.path.join(OUT, "features", split, f"{cid}.npz")) as f:
        return f["X"], f["y"]


def _fit_all(train_cases, n_per_class, seed, n_jobs, configs):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    per_case = max(n_per_class // len(train_cases), 1000)   # equal share per case, balanced per case
    for cid in train_cases:
        X, y = _load_feats("val", cid)
        Xb, yb = _balanced_sample(X, y, per_case, rng)
        Xs.append(Xb); ys.append(yb)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    models, times = {}, {}
    for name in configs:
        t0 = time.time(); m = _make_model(name, n_jobs); m.fit(X, y)
        models[name] = m; times[name] = time.time() - t0
    return models, times, (len(X), int(y.sum()))


def _predict_save(models, split, cid):
    from sklearn.metrics import roc_auc_score, log_loss
    X, y = _load_feats(split, cid)
    out = {}; diag = {}
    for name, m in models.items():
        p = m.predict_proba(X)[:, 1].astype(np.float32)
        out[name] = p.astype(np.float16)
        if y.min() != y.max():
            diag[name] = {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))}
    d = os.path.join(OUT, "probs", split); os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"{cid}.npz"), **out)
    return diag


def cmd_train(a):
    import pandas as pd, pickle
    cases = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(OUT, "features", "val")))
    assert len(cases) == 30, cases
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(cases))
    folds = {cases[i]: int(k % 5) for k, i in enumerate(perm)}
    with open(os.path.join(OUT, "folds.json"), "w") as fh:
        json.dump(folds, fh, indent=1)
    rows = []
    for k in range(5):
        tr = [c for c in cases if folds[c] != k]; te = [c for c in cases if folds[c] == k]
        models, times, (n, npos) = _fit_all(tr, a.n_per_class, 100 + k, a.procs, CONFIGS)
        log(f"train fold {k}: {n} voxels ({npos} pos) from {len(tr)} cases; fit times " +
            ", ".join(f"{m}={t:.0f}s" for m, t in times.items()))
        for cid in te:
            diag = _predict_save(models, "val", cid)
            for name, dd in diag.items():
                rows.append({"fold": k, "case": cid, "config": name, **dd})
            log(f"  predicted {cid}: " + ", ".join(f"{m} auc={d['auc']:.4f}" for m, d in diag.items()))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "cv_voxel_diagnostics.csv"), index=False)
    log("CV voxel AUC by config:\n" + df.groupby("config")[["auc", "logloss"]].mean().to_string())
    # final models on all val
    models, times, (n, npos) = _fit_all(cases, a.n_per_class, 999, a.procs, CONFIGS)
    log(f"final fit on all 30 val cases: {n} voxels ({npos} pos); " + ", ".join(f"{m}={t:.0f}s" for m, t in times.items()))
    with open(os.path.join(OUT, "models.pkl"), "wb") as fh:
        pickle.dump(models, fh)
    # feature importances
    imp = pd.DataFrame({"feature": FEATURES})
    lr = models["lr"]; imp["lr_std_coef"] = lr.named_steps["logisticregression"].coef_[0]
    for name in CONFIGS[1:]:
        b = models[name].booster_
        g = b.feature_importance("gain"); imp[f"{name}_gain_pct"] = 100 * g / g.sum()
        imp[f"{name}_splits"] = b.feature_importance("split")
    imp.to_csv(os.path.join(OUT, "feature_importances.csv"), index=False)
    log("feature importances:\n" + imp.round(3).to_string())


def cmd_predict(a):
    import pickle
    with open(os.path.join(OUT, "models.pkl"), "rb") as fh:
        models = pickle.load(fh)
    for m in models.values():
        if hasattr(m, "set_params") and "n_jobs" in m.get_params():
            m.set_params(n_jobs=a.procs)
    d = os.path.join(OUT, "features", a.split)
    for f in sorted(os.listdir(d)):
        cid = os.path.splitext(f)[0]
        t0 = time.time(); diag = _predict_save(models, a.split, cid)
        log(f"predicted {a.split} {cid} in {time.time()-t0:.1f}s: " + ", ".join(f"{m} auc={dd['auc']:.4f}" for m, dd in diag.items()))


# ------------------------------------------------------------------ selection / evaluation (airmorph env)

def build_mask(split, cid, prob, thr, seed):
    import common
    with np.load(os.path.join(OUT, "features", split, f"{cid}.npz")) as f:
        idx, box, shape = f["idx"], f["box"], tuple(int(s) for s in f["shape"])
    sl = tuple(slice(int(l), int(h)) for l, h in box)
    full = np.zeros(shape, np.uint8)
    crop = np.zeros(full[sl].shape, np.uint8).ravel()
    crop[idx[prob >= thr]] = 1
    full[sl] = crop.reshape(full[sl].shape)
    return common.keep_seeded_component(full, seed)


def _sweep_case(args):
    split, cid, configs, thresholds = args
    import common
    c = common.load_cached(split, cid)
    probs = np.load(os.path.join(OUT, "probs", split, f"{cid}.npz"))
    rows = []
    for name in configs:
        p = probs[name].astype(np.float32)
        for thr in thresholds:
            t0 = time.time()
            m = build_mask(split, cid, p, thr, c["airmorph"])
            # only the two metrics the selection rule needs (same ATM'22 functions common.score calls)
            _, _, bnd = common.branch_detected_calculation(m, c["gt_parsing"], c["gt_skeleton"], thresh=0.8)
            dsc = common.dice_coefficient_score_calculation(m, c["gt"])
            rows.append({"split": split, "case": cid, "config": name, "thr": thr, "DSC": float(dsc),
                         "BND": float(bnd), "n_vox": int(m.sum()), "sec": time.time() - t0})
    return rows


def cmd_select(a):
    import common, pandas as pd
    from multiprocessing import Pool
    cases = common.cases("val")
    percase = os.path.join(OUT, "val_sweep_percase.csv")
    prev = pd.read_csv(percase) if os.path.exists(percase) else None
    thresholds = a.thresholds or THRESHOLDS
    jobs = []
    for cid in cases:
        for name in SWEEP_CONFIGS if not a.configs else a.configs:
            have = set(prev[(prev.case == cid) & (prev.config == name)].thr.round(3)) if prev is not None else set()
            todo = [t for t in thresholds if round(t, 3) not in have]
            if todo:
                jobs.append(("val", cid, [name], todo))
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(_sweep_case, jobs):
            d = pd.DataFrame(r); d.to_csv(percase, mode="a", header=not os.path.exists(percase), index=False)
            log(f"select: swept {r[0]['case']} ({sum(x['sec'] for x in r):.0f}s)")
    df = pd.read_csv(percase)
    assert set(df.case) == set(cases)
    ref = pd.read_csv(os.path.join(os.path.dirname(HERE), "results", "whole_val.csv"))
    union = ref[ref.strategy == "union_seeded"].set_index("case").loc[cases]
    summ = df.groupby(["config", "thr"])[["DSC", "BND", "n_vox"]].mean().reset_index()
    summ["BND_loss_vs_union"] = union["BND"].mean() - summ["BND"]
    summ["DSC_gain_vs_union"] = summ["DSC"] - union["DSC"].mean()
    summ["feasible"] = summ["BND_loss_vs_union"] <= 1.0
    summ.to_csv(os.path.join(OUT, "val_sweep_summary.csv"), index=False)
    log("val sweep summary (union DSC %.2f BND %.2f):\n%s" % (union["DSC"].mean(), union["BND"].mean(), summ.round(2).to_string()))
    choice = {}
    feas = summ[summ.feasible]
    for name in SWEEP_CONFIGS:
        s = feas[feas.config == name]
        if len(s):
            best = s.loc[s.DSC.idxmax()]
            choice[name] = {"thr": float(best.thr), "DSC": float(best.DSC), "BND": float(best.BND)}
    overall = feas.loc[feas.DSC.idxmax()]
    choice["selected"] = {"config": overall.config, "thr": float(overall.thr), "DSC": float(overall.DSC), "BND": float(overall.BND)}
    with open(os.path.join(OUT, "selection.json"), "w") as fh:
        json.dump(choice, fh, indent=1)
    log("selection: " + json.dumps(choice))


def _eval_case(args):
    split, cid, name, thr, save = args
    import common
    c = common.load_cached(split, cid)
    p = np.load(os.path.join(OUT, "probs", split, f"{cid}.npz"))[name].astype(np.float32)
    t0 = time.time()
    m = build_mask(split, cid, p, thr, c["airmorph"])
    sec = time.time() - t0
    s = common.score(m, c, cldice=True)
    with np.load(os.path.join(OUT, "features", split, f"{cid}.npz")) as f:
        reach = float(f["y"].sum()) / max(float(f["gt_total"]), 1)
    if save:
        d = os.path.join(OUT, "masks", split); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, f"ATM26_{cid}.npz"), mask=m.astype(np.uint8))
    return {"split": split, "case": cid, "config": name, "thr": thr, **s,
            "n_vox": int(m.sum()), "gt_reachable_frac": reach, "mask_sec": sec}


def cmd_evaluate(a):
    import common, pandas as pd
    from multiprocessing import Pool
    with open(os.path.join(OUT, "selection.json")) as fh:
        sel = json.load(fh)
    todo = [(sel["selected"]["config"], sel["selected"]["thr"], True)]
    if a.also:   # also evaluate the best operating point of the other learner family, for the comparison
        for name in a.also:
            if name != sel["selected"]["config"] and name in sel:
                todo.append((name, sel[name]["thr"], False))
    cases = common.cases(a.split)
    jobs = [(a.split, cid, name, thr, save) for name, thr, save in todo for cid in cases]
    rows = []
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(_eval_case, jobs):
            rows.append(r); log(f"evaluate {a.split} {r['case']} {r['config']}@{r['thr']}: DSC {r['DSC']:.2f} BND {r['BND']:.2f}")
    df = pd.DataFrame(rows).sort_values(["config", "case"])
    df.to_csv(os.path.join(OUT, f"percase_{a.split}.csv"), index=False)
    log(f"{a.split} means:\n" + df.groupby("config")[["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "FPR", "gt_reachable_frac", "mask_sec"]].mean().round(2).to_string())


# ------------------------------------------------------------------ report tables

COLS = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision"]


def cmd_report(a):
    import pandas as pd
    with open(os.path.join(OUT, "selection.json")) as fh:
        sel = json.load(fh)
    lines = []
    for split in ("val", "test", "pediatric"):
        f = os.path.join(OUT, f"percase_{split}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        ref = pd.read_csv(os.path.join(os.path.dirname(HERE), "results", f"whole_{split}.csv"))
        rows = []
        for strat, label in (("union_seeded", "union, seeded"), ("topofuse", "union, refined (topofuse)")):
            r = ref[ref.strategy == strat][COLS].mean(); rows.append([label] + [f"{v:.2f}" for v in r])
        for name, g in df.groupby("config"):
            tag = " **(selected)**" if name == sel["selected"]["config"] else ""
            rows.append([f"stacking {name} @ thr {g.thr.iloc[0]:.2f}{tag}"] + [f"{v:.2f}" for v in g[COLS].mean()])
        lines.append(f"\n### {split} (n={df.case.nunique()})\n")
        lines.append("| method | " + " | ".join(COLS) + " |"); lines.append("|---" * (len(COLS) + 1) + "|")
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        if split in ("test", "pediatric"):
            lines.append(f"\nPaired per-case differences on {split} (stacking minus reference):\n")
            lines.append("| stacking config | vs | metric | mean diff | improved / n |"); lines.append("|---|---|---|---|---|")
            for name, g in df.groupby("config"):
                g = g.set_index("case")
                for strat, label in (("union_seeded", "union"), ("topofuse", "refined")):
                    r = ref[ref.strategy == strat].set_index("case").loc[g.index]
                    for col in ("DSC", "clDice", "BND", "TLD"):
                        d = g[col] - r[col]
                        lines.append(f"| {name} | {label} | {col} | {d.mean():+.2f} | {(d > 0).sum()} / {len(d)} |")
            lines.append(f"\nMean candidate-region GT recall ceiling: {100*df.gt_reachable_frac.mean():.2f}%; "
                         f"mean mask build time (threshold + seeded component): {df.mask_sec.mean():.1f} s")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "report_tables.md"), "w") as fh:
        fh.write(txt)
    print(txt)



def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("features"); p.add_argument("--split", required=True); p.add_argument("--procs", type=int, default=3)
    p = sub.add_parser("train"); p.add_argument("--procs", type=int, default=3); p.add_argument("--n-per-class", type=int, default=1_000_000)
    p = sub.add_parser("predict"); p.add_argument("--split", required=True); p.add_argument("--procs", type=int, default=3)
    p = sub.add_parser("select"); p.add_argument("--procs", type=int, default=3)
    p.add_argument("--thresholds", type=float, nargs="*", help="extra grid points to add to the sweep CSV")
    p.add_argument("--configs", nargs="*", help="restrict the extra grid to these configs")
    p = sub.add_parser("report")
    p = sub.add_parser("evaluate"); p.add_argument("--split", required=True); p.add_argument("--procs", type=int, default=3)
    p.add_argument("--also", nargs="*", default=["lr", "gbt_small"])
    a = ap.parse_args()
    {"features": cmd_features, "train": cmd_train, "predict": cmd_predict, "select": cmd_select,
     "evaluate": cmd_evaluate, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
