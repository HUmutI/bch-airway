"""Ablation 2: spatially adaptive weighted ensemble / mixture-of-experts gating.

The three members are binary masks, so a weighted ensemble sum_m w_m mask_m >= t
over a region is one monotone Boolean rule on the (AirMorph, MEDPSeg, TfeNet)
vote triple; the 0.1-step weight grid x threshold yields exactly the 18
non-constant monotone rules of three inputs (plus "drop"). The gate selects a
rule per region, where regions are defined by GT-free descriptors of the union
mask M: branch radius, branch generation, branch model support, distance to
the mask boundary (EDT of M), relative distance to the skeleton (dist / r) and
the local mean vote count (5^3 box). Fitted on val only:

  global : one rule everywhere (ablation "optimised global weights")
  tree   : greedy decision tree (depth <= 4) over the descriptor bins, each leaf
           its own rule, rules chosen by Dinkelbach iteration on the pooled
           weighted Dice 2TP / (2TP + FP + beta FN)
  treebin: tree leaves refined per fine descriptor bin where the bin has support
  soft   : multinomial logistic regression w(x) = softmax(W phi(x)) trained to
           predict which member is correct at disagreement voxels; output
           sum_m w_m(x) mask_m(x) >= t

Everything ends with the paper's seeded-component rule (seed = AirMorph mask).

Usage (from evaluation/fusion_uncertainty):
  python ablation/moe_gating.py collect --split val
  python ablation/moe_gating.py fit
  python ablation/moe_gating.py evalval
  python ablation/moe_gating.py final --split test   (and pediatric)
"""
import argparse, itertools, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402

OUT = "/fileserver/external/quin/ege/fusion_uncertainty_run/ablation/moe_gating"
NPROC = 3

# ----------------------------------------------------------------- descriptors
R_EDGES = [0.75, 1.0, 1.5, 2.5]          # mm, branch mean radius -> 5 bins
G_EDGES = [4, 7, 10]                     # generation -> 4 bins
NM_EDGES = [1.5, 2.5]                    # branch n_models 1 / 2 / 3
BD_EDGES = [1.01, 2.01]                  # boundary distance in voxels -> 3 bins
REL_EDGES = [0.5, 1.0]                   # dist / radius -> 3 bins
LV_EDGES = [1.5, 2.5]                    # local mean vote count -> 3 bins
AXES = [("radius", R_EDGES), ("generation", G_EDGES), ("n_models", NM_EDGES),
        ("boundary_dist", BD_EDGES), ("rel_dist", REL_EDGES), ("local_votes", LV_EDGES)]
SHAPE = tuple(len(e) + 1 for _, e in AXES)
NBIN = int(np.prod(SHAPE))
AXIS_LABELS = {
    "radius": ["<0.75mm", "0.75-1", "1-1.5", "1.5-2.5", ">=2.5mm"],
    "generation": ["g0-3", "g4-6", "g7-9", "g10+"],
    "n_models": ["1 model", "2 models", "3 models"],
    "boundary_dist": ["surface", "2nd layer", "interior"],
    "rel_dist": ["core <0.5r", "0.5-1r", ">1r"],
    "local_votes": ["lv<1.5", "lv 1.5-2.5", "lv>=2.5"],
}


def bbox(M):
    nz = np.nonzero(M)
    return tuple(slice(max(int(a.min()) - 2, 0), int(a.max()) + 3) for a in nz)


def descriptors(c):
    """Per-voxel descriptors on the bbox of M (dict of arrays) + the bbox slices."""
    M = c["M"]; sl = bbox(M); sub = M[sl]
    sp = tuple(float(s) for s in c["spacing"]); vox = max(sp)
    bm = c["branch_map"][sl]
    tab = c["tab"]
    r = tab["r_mean_mm"].to_numpy(np.float32).copy(); r[0] = 0
    g = tab["generation"].to_numpy(np.float32).copy(); g[0] = 0
    nm = tab["n_models"].to_numpy(np.float32).copy(); nm[0] = 1
    radius = r[bm]; gen = g[bm]; nmod = nm[bm]
    bd = ndimage.distance_transform_edt(sub, sampling=sp).astype(np.float32) / vox   # voxels
    rel = c["dist"][sl] / np.maximum(radius, 0.3)
    a = c["airmorph"][sl]; m = c["medpseg"][sl]; t = c["tfenet"][sl]
    votes = (a.astype(np.float32) + m + t)
    lv = ndimage.uniform_filter(votes, size=5, mode="constant")
    p3 = (a.astype(np.int8) * 4 + m.astype(np.int8) * 2 + t.astype(np.int8))
    return {"sl": sl, "M": sub, "radius": radius, "generation": gen, "n_models": nmod,
            "boundary_dist": bd, "rel_dist": rel, "local_votes": lv, "p3": p3,
            "a": a, "m": m, "t": t}


def bin_index(d):
    """Flat fine-bin index per voxel of the bbox."""
    idx = []
    for name, edges in AXES:
        idx.append(np.digitize(d[name], edges).astype(np.int64))
    return np.ravel_multi_index(idx, SHAPE)


def features(d, sel):
    """Soft-gate feature matrix phi(x) for the selected voxels (float32)."""
    r = d["radius"][sel]; g = d["generation"][sel]; nm = d["n_models"][sel]
    bd = d["boundary_dist"][sel]; rel = d["rel_dist"][sel]; lv = d["local_votes"][sel]
    cols = [np.ones_like(r), np.log(np.maximum(r, 0.3)), np.minimum(r, 6.0) / 3.0, g / 8.0,
            (nm == 1), (nm == 2), np.minimum(bd, 5.0) / 2.0, (bd <= 1.01),
            np.minimum(rel, 3.0), (rel <= 0.5), lv / 3.0, (lv >= 2.5)]
    return np.stack([np.asarray(x, np.float32) for x in cols], 1)


# ----------------------------------------------------------------- rules
def enumerate_rules():
    """All distinct decision rules reachable by w in 0.1 steps (sum 1) and a
    threshold t, as 8-vectors over the pattern index p3 = 4a+2m+t; plus drop."""
    pats = np.array([[p >> 2 & 1, p >> 1 & 1, p & 1] for p in range(8)], np.float32)
    seen = {}
    for i in range(11):
        for j in range(11 - i):
            w = np.array([i, 10 - i - j, j], np.float32) / 10.0
            s = pats @ w
            for t in sorted(set(np.round(s[1:], 3))):
                if t <= 0:
                    continue
                rule = (s >= t - 1e-6).astype(np.int8); rule[0] = 0
                key = tuple(rule[1:])
                rep = ([round(float(x), 1) for x in w], round(float(t), 2))
                if key not in seen or max(w) < max(seen[key][0]):   # most balanced representative
                    seen[key] = rep
    rules = np.zeros((len(seen) + 1, 8), np.int8); reps = [([0, 0, 0], 2.0)]
    for k, (key, rep) in enumerate(sorted(seen.items(), key=lambda kv: (sum(kv[0]), kv[0]))):
        rules[k + 1, 1:] = key; reps.append(rep)
    return rules, reps


RULES, RULE_REPS = enumerate_rules()


def rule_name(rule):
    names = {"a": 4, "m": 2, "t": 1}
    on = [p for p in range(1, 8) if rule[p]]
    if not on:
        return "drop"
    if len(on) == 7:
        return "union"
    if on == [3, 5, 6, 7]:
        return "majority"
    # minimal true patterns (minterms) as a monotone DNF
    terms = []
    for p in sorted(on, key=lambda q: bin(q).count("1")):
        if not any((p & q) == q for q in terms):
            terms.append(p)
    return " | ".join("&".join(k for k, b in names.items() if p & b) for p in terms)


def leaf_stats(n):
    """n: (..., 8, 2) counts -> TP, FP, FN per rule (arrays over rules)."""
    pos = n[..., 1]; neg = n[..., 0]
    TP = np.einsum("rp,...p->...r", RULES.astype(np.float64), pos)
    FP = np.einsum("rp,...p->...r", RULES.astype(np.float64), neg)
    FN = pos.sum(-1)[..., None] - TP
    return TP, FP, FN


def best_rules(n_leaves, lam, beta):
    """n_leaves: (L, 8, 2). Returns rule index per leaf and the objective value."""
    TP, FP, FN = leaf_stats(n_leaves)
    val = 2 * TP - lam * (2 * TP + FP + beta * FN)
    k = val.argmax(-1)
    return k, val.max(-1)


def dinkelbach(n_leaves, beta, fn0, iters=20, fixed=(0.0, 0.0)):
    """fixed = (TP, FP) contributed by protected voxels that are always kept."""
    lam = 0.9
    for _ in range(iters):
        k, _ = best_rules(n_leaves, lam, beta)
        TP, FP, FN = leaf_stats(n_leaves)
        ar = np.arange(len(k))
        tp, fp, fn = TP[ar, k].sum() + fixed[0], FP[ar, k].sum() + fixed[1], FN[ar, k].sum() + fn0
        new = 2 * tp / (2 * tp + fp + beta * fn)
        if abs(new - lam) < 1e-9:
            break
        lam = new
    return k, lam


def dsc_from(table, leaf_of_bin, k_leaf, fn0):
    """Pooled DSC (beta = 1) of a gate on a count table (NBIN, 8, 2)."""
    rule = RULES[k_leaf[leaf_of_bin]]                         # (NBIN, 8)
    tp = (rule * table[..., 1]).sum(); fp = (rule * table[..., 0]).sum()
    fn = table[..., 1].sum() - tp + fn0
    return 100 * 2 * tp / (2 * tp + fp + fn)


# ----------------------------------------------------------------- tree
class Node:
    def __init__(self, box):
        self.box = box; self.split = None; self.kids = None; self.rule = None

    def leaves(self):
        return [self] if self.kids is None else [l for k in self.kids for l in k.leaves()]


def box_mask(box):
    idx = np.indices(SHAPE).reshape(len(SHAPE), -1)
    ok = np.ones(NBIN, bool)
    for ax, (lo, hi) in enumerate(box):
        ok &= (idx[ax] >= lo) & (idx[ax] < hi)
    return ok


def grow_tree(table, lam, beta, max_depth=4, min_frac=0.003, min_gain=0.0):
    total = table.sum()
    root = Node([(0, s) for s in SHAPE])

    def value(mask):
        n = table[mask].sum(0)[None]
        k, v = best_rules(n, lam, beta)
        return float(v[0]), int(k[0]), n[0].sum()

    def rec(node, depth):
        mask = box_mask(node.box)
        v0, k0, cnt = value(mask)
        node.rule = k0
        if depth >= max_depth:
            return
        best = (min_gain, None)
        for ax in range(len(SHAPE)):
            lo, hi = node.box[ax]
            for cut in range(lo + 1, hi):
                bl = list(node.box); bl[ax] = (lo, cut)
                br = list(node.box); br[ax] = (cut, hi)
                ml, mr = box_mask(bl), box_mask(br)
                vl, _, cl = value(ml); vr, _, cr = value(mr)
                if cl < min_frac * total or cr < min_frac * total:
                    continue
                gain = vl + vr - v0
                if gain > best[0]:
                    best = (gain, (ax, cut, bl, br))
        if best[1] is None:
            return
        ax, cut, bl, br = best[1]
        node.split = (ax, cut); node.kids = [Node(bl), Node(br)]
        for kid in node.kids:
            rec(kid, depth + 1)

    rec(root, 0)
    return root


def tree_leaf_of_bin(root):
    leaves = root.leaves()
    lob = np.zeros(NBIN, np.int64)
    for i, l in enumerate(leaves):
        lob[box_mask(l.box)] = i
    return lob, leaves


def describe_box(box):
    parts = []
    for ax, (name, _) in enumerate(AXES):
        lo, hi = box[ax]
        if (lo, hi) != (0, SHAPE[ax]):
            labs = AXIS_LABELS[name][lo:hi]
            parts.append(f"{name} in [{labs[0]} .. {labs[-1]}]" if len(labs) > 1 else f"{name}={labs[0]}")
    return "; ".join(parts) if parts else "everywhere"


UNION_K = int(np.flatnonzero(RULES[:, 1:].sum(1) == 7)[0])


def protect_bins(protect):
    """Bins forced to the union rule: C = skeleton core (rel_dist < 0.5 r),
    N = branches supported by <= 2 models' centrelines."""
    idx = np.indices(SHAPE).reshape(len(SHAPE), -1)
    ax = {name: i for i, (name, _) in enumerate(AXES)}
    prot = np.zeros(NBIN, bool)
    if "C" in protect:
        prot |= idx[ax["rel_dist"]] == 0
    if "N" in protect:
        prot |= idx[ax["n_models"]] <= 1
    return prot


def fit_gate(table, fn0, beta, kind, min_bin_frac=0.001, protect=""):
    """Returns leaf_of_bin (NBIN,), k_leaf (L,), and a description list.
    Protected bins keep the union rule and enter the objective as constants."""
    prot = protect_bins(protect)
    fixed = (float(table[prot][..., 1].sum()), float(table[prot][..., 0].sum()))
    table = table * (~prot)[:, None, None]
    if kind == "global":
        k, lam = dinkelbach(table.sum(0)[None], beta, fn0, fixed=fixed)
        lob = np.zeros(NBIN, np.int64); k = np.append(k, UNION_K); lob[prot] = 1
        return lob, k, [("everywhere" + (" (except protected)" if protect else ""), int(k[0]), float(table.sum()))]
    k0, lam = dinkelbach(table.sum(0)[None], beta, fn0, fixed=fixed)
    root = grow_tree(table, lam, beta)
    lob, leaves = tree_leaf_of_bin(root)
    n_leaves = np.stack([table[lob == i].sum(0) for i in range(len(leaves))])
    k, lam = dinkelbach(n_leaves, beta, fn0, fixed=fixed)
    desc = [(describe_box(l.box), int(k[i]), float(n_leaves[i].sum())) for i, l in enumerate(leaves)]
    if protect:
        k = np.append(k, UNION_K); lob[prot] = len(k) - 1
        desc.append((f"protected ({protect})", UNION_K, fixed[0] + fixed[1]))
    if kind == "tree":
        return lob, k, desc
    # treebin: refine bins with enough support, each becomes its own leaf
    support = table.sum((1, 2))
    big = np.nonzero((support >= min_bin_frac * table.sum()) & ~prot)[0]
    lob2 = lob.copy(); n2 = list(n_leaves) + ([np.zeros((8, 2))] if protect else [])
    for b in big:
        lob2[b] = len(n2); n2.append(table[b])
    n2 = np.stack(n2)
    k2, lam = dinkelbach(n2, beta, fn0, fixed=fixed)
    if protect:
        k2[len(n_leaves)] = UNION_K
    return lob2, k2, desc


# ----------------------------------------------------------------- collect
def collect_case(args):
    split, cid = args
    t0 = time.time()
    c = common.load_cached(split, cid)
    d = descriptors(c)
    M = d["M"]; sl = d["sl"]
    bi = bin_index(d)[M]; p3 = d["p3"][M].astype(np.int64); gt = c["gt"][sl][M].astype(np.int64)
    flat = (bi * 8 + p3) * 2 + gt
    table = np.bincount(flat, minlength=NBIN * 16).reshape(NBIN, 8, 2).astype(np.int64)
    fn0 = int(c["gt"].sum() - gt.sum())
    # soft-gate samples: disagreement voxels (p3 in 1..6), cap 150k
    dis = M & (d["p3"] > 0) & (d["p3"] < 7)
    idx = np.flatnonzero(dis)
    rng = np.random.default_rng(abs(hash(cid)) % (2 ** 32))
    if len(idx) > 150000:
        idx = rng.choice(idx, 150000, replace=False)
    sel = np.zeros(dis.shape, bool); sel.flat[idx] = True
    X = features(d, sel)
    votes = np.stack([d["a"][sel], d["m"][sel], d["t"][sel]], 1).astype(np.float32)
    g = c["gt"][sl][sel].astype(np.float32)
    frac = float(dis.sum()) / max(len(idx), 1)
    od = os.path.join(OUT, "counts", split); os.makedirs(od, exist_ok=True)
    np.savez_compressed(os.path.join(od, cid + ".npz"), table=table, fn0=fn0, X=X, votes=votes,
                        gt=g, weight=frac, n_M=int(M.sum()))
    return cid, time.time() - t0


def load_counts(split, cids):
    T, F, S = [], [], []
    for cid in cids:
        z = np.load(os.path.join(OUT, "counts", split, cid + ".npz"))
        T.append(z["table"]); F.append(int(z["fn0"]))
        S.append({"X": z["X"], "votes": z["votes"], "gt": z["gt"], "weight": float(z["weight"])})
    return np.stack(T), np.array(F), S


# ----------------------------------------------------------------- soft gate
def train_soft(samples, epochs=300, l2=1e-4, seed=0):
    import torch
    torch.manual_seed(seed)
    X = torch.from_numpy(np.concatenate([s["X"] for s in samples]))
    V = torch.from_numpy(np.concatenate([s["votes"] for s in samples]))
    G = torch.from_numpy(np.concatenate([s["gt"] for s in samples]))
    # target: members that are correct at the voxel, normalised
    correct = torch.where(G[:, None] > 0.5, V, 1 - V)
    target = correct / correct.sum(1, keepdim=True).clamp_min(1)
    W = torch.zeros(X.shape[1], 3, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=epochs, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logp = torch.log_softmax(X @ W, 1)
        loss = -(target * logp).sum(1).mean() + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    loss = closure().item()
    return W.detach().numpy().astype(np.float32), float(loss)


def soft_scores(d, W):
    M = d["M"]
    X = features(d, M)
    z = X @ W; z -= z.max(1, keepdims=True); w = np.exp(z); w /= w.sum(1, keepdims=True)
    votes = np.stack([d["a"][M], d["m"][M], d["t"][M]], 1).astype(np.float32)
    s = np.zeros(M.shape, np.float32); s[M] = (w * votes).sum(1)
    wm = np.zeros(M.shape + (3,), np.float32); wm[M] = w
    return s, wm


# ----------------------------------------------------------------- apply
def apply_hard(c, d, gate):
    lob = np.asarray(gate["leaf_of_bin"]); k = np.asarray(gate["k_leaf"])
    rule_of_bin = RULES[k[lob]]                       # (NBIN, 8)
    bi = bin_index(d)
    keep_sub = rule_of_bin[bi, d["p3"].astype(np.int64)].astype(bool) & d["M"]
    out = np.zeros(c["M"].shape, np.uint8); out[d["sl"]] = keep_sub
    return common.keep_seeded_component(out, c["airmorph"])


def protect_voxels(d, protect):
    prot = np.zeros(d["M"].shape, bool)
    if "C" in protect:
        prot |= d["rel_dist"] < REL_EDGES[0]
    if "N" in protect:
        prot |= d["n_models"] < NM_EDGES[1]
    return prot & d["M"]


def apply_soft(c, d, W, t, protect=""):
    s, _ = soft_scores(d, W)
    keep = (s >= t) & d["M"]
    if protect:
        keep |= protect_voxels(d, protect)
    out = np.zeros(c["M"].shape, np.uint8); out[d["sl"]] = keep
    return common.keep_seeded_component(out, c["airmorph"])


def eval_case(args):
    split, cid, configs, cldice, save_dir = args
    t0 = time.time()
    c = common.load_cached(split, cid)
    d = descriptors(c)
    t_desc = time.time() - t0
    rows = []
    for name, cfg in configs.items():
        t1 = time.time()
        if cfg["type"] == "hard":
            pred = apply_hard(c, d, cfg)
        else:
            pred = apply_soft(c, d, np.asarray(cfg["W"], np.float32), cfg["t"], cfg.get("protect", ""))
        t_apply = time.time() - t1
        s = common.score(pred, c, cldice=cldice)
        s.update({"split": split, "case": cid, "config": name, "sec_desc": t_desc, "sec_apply": t_apply,
                  "n_pred": int(pred.sum()), "n_M": int(c["M"].sum())})
        rows.append(s)
        if save_dir:
            os.makedirs(os.path.join(save_dir, split), exist_ok=True)
            np.savez_compressed(os.path.join(save_dir, split, f"ATM26_{cid}.npz"), mask=pred.astype(np.uint8))
    return rows


def run_eval(split, configs, cldice, tag, save_dir=None):
    cids = common.cases(split)
    with Pool(NPROC) as pool:
        res = pool.map(eval_case, [(split, cid, configs, cldice, save_dir) for cid in cids], chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(OUT, f"percase_{tag}_{split}.csv"), index=False)
    return df


# ----------------------------------------------------------------- main
def cmd_collect(a):
    cids = common.cases(a.split)
    with Pool(NPROC) as pool:
        for cid, dt in pool.imap_unordered(collect_case, [(a.split, c) for c in cids]):
            print(f"{a.split} {cid} {dt:.1f}s", flush=True)


def gate_dict(lob, k, desc, beta, kind):
    return {"type": "hard", "kind": kind, "beta": beta, "leaf_of_bin": lob.tolist(), "k_leaf": k.tolist(),
            "leaves": [{"region": r, "rule": rule_name(RULES[kk]), "w": RULE_REPS[kk][0],
                        "t": RULE_REPS[kk][1], "n_val_vox": n} for r, kk, n in desc]}


def cmd_fit(a):
    cids = common.cases("val")
    T, F, S = load_counts("val", cids)
    table = T.sum(0); fn0 = int(F.sum())
    log = []
    print("rules:", len(RULES)); log.append(f"{len(RULES)} rules incl. drop: " +
                                             ", ".join(rule_name(r) for r in RULES))
    base = dsc_from(table, np.zeros(NBIN, np.int64), np.array([RULES.shape[0] - 1]), fn0)
    print("pooled val DSC union:", base)
    # 5-fold CV within val over (kind, beta) on pooled held-out DSC (beta=1 scoring)
    folds = [cids[i::5] for i in range(5)]
    betas = [1.0, 1.5, 2.0, 3.0]
    cv = []
    for kind in (() if os.path.exists(os.path.join(OUT, "cv_val_pooled.csv")) else ("global", "tree", "treebin")):
        for beta in betas:
            tps = fps = fns = 0.0
            for f in folds:
                tr = [i for i, c in enumerate(cids) if c not in f]; te = [i for i, c in enumerate(cids) if c in f]
                lob, k, _ = fit_gate(T[tr].sum(0), int(F[tr].sum()), beta, kind)
                rule = RULES[k[lob]]
                tt = T[te].sum(0)
                tp = (rule * tt[..., 1]).sum(); fp = (rule * tt[..., 0]).sum(); fn = tt[..., 1].sum() - tp + F[te].sum()
                tps += tp; fps += fp; fns += fn
            dsc = 100 * 2 * tps / (2 * tps + fps + fns)
            lob, k, _ = fit_gate(table, fn0, beta, kind)
            ins = dsc_from(table, lob, k, fn0)
            cv.append({"kind": kind, "beta": beta, "cv_pooled_DSC": dsc, "insample_pooled_DSC": ins,
                       "n_leaves": int(len(k))})
            print(cv[-1], flush=True)
    if cv:
        pd.DataFrame(cv).to_csv(os.path.join(OUT, "cv_val_pooled.csv"), index=False)
    # final gates on all val
    gates = {}
    for kind in ("global", "tree", "treebin"):
        for beta in betas:
            for prot in ("", "C", "N", "CN"):
                lob, k, desc = fit_gate(table, fn0, beta, kind, protect=prot)
                g = gate_dict(lob, k, desc, beta, kind); g["protect"] = prot
                gates[f"{kind}_b{beta}" + (f"_p{prot}" if prot else "")] = g
    # soft gate
    t0 = time.time(); W, loss = train_soft(S); print("soft LR loss", loss, time.time() - t0)
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        gates[f"soft_t{t}"] = {"type": "soft", "W": W.tolist(), "t": t, "loss": loss}
        for prot in ("C", "N", "CN"):
            gates[f"soft_t{t}_p{prot}"] = {"type": "soft", "W": W.tolist(), "t": t, "loss": loss, "protect": prot}
    with open(os.path.join(OUT, "gates_val.json"), "w") as fh:
        json.dump(gates, fh)
    # human-readable tree descriptions
    with open(os.path.join(OUT, "gates_val.txt"), "w") as fh:
        for name, g in gates.items():
            if g["type"] != "hard" or g["kind"] == "treebin" or g.get("protect"):
                continue
            fh.write(f"## {name}\n")
            for l in g["leaves"]:
                fh.write(f"  {l['region']}: {l['rule']}  (w={l['w']}, t={l['t']}, n={l['n_val_vox']:.0f})\n")
        fh.write("## soft W (rows: " + ", ".join(
            ["1", "log r", "r/3", "gen/8", "nm==1", "nm==2", "bd/2", "surface", "rel", "core", "lv/3", "lv>=2.5"]) +
            "; cols a,m,t)\n" + np.array2string(W, precision=3) + "\n")


def cmd_evalval(a):
    with open(os.path.join(OUT, "gates_val.json")) as fh:
        gates = json.load(fh)
    names = a.configs.split(",") if a.configs else list(gates)
    df = run_eval("val", {n: gates[n] for n in names}, cldice=False, tag=a.tag)
    print(df.groupby("config")[["DSC", "TLD", "BND", "Sensitivity", "Precision"]].mean().round(2))


def cmd_final(a):
    with open(os.path.join(OUT, "gates_val.json")) as fh:
        gates = json.load(fh)
    names = a.configs.split(",")
    df = run_eval(a.split, {n: gates[n] for n in names}, cldice=True, tag=a.tag,
                  save_dir=os.path.join(OUT, "masks") if a.save else None)
    print(df.groupby("config")[["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision"]].mean().round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect"); p.add_argument("--split", default="val")
    sub.add_parser("fit")
    p = sub.add_parser("evalval"); p.add_argument("--configs", default=""); p.add_argument("--tag", default="sweep")
    p = sub.add_parser("final"); p.add_argument("--split", required=True); p.add_argument("--configs", required=True)
    p.add_argument("--tag", default="final"); p.add_argument("--save", action="store_true")
    a = ap.parse_args()
    {"collect": cmd_collect, "fit": cmd_fit, "evalval": cmd_evalval, "final": cmd_final}[a.cmd](a)
