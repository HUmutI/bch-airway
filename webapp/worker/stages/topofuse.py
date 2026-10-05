"""Uncertainty-aware fusion for the web pipeline, self-contained.

Mirrors evaluation/fusion_uncertainty (tree.py, sweep.py, refine.py) so the
site runs exactly what the study measured, without importing the study. Three
pieces, each usable alone:

  confidence(votes, medpseg, tfenet)   P(airway | vote pattern) per voxel, from
                                       confidence_lut.json (fitted on the ATM'22
                                       validation split).
  branch_tree(mask, spacing, ...)      skeleton -> branches -> tree, with per-
                                       branch evidence and the logistic
                                       probability from branch_logit.json.
  prune(...) / refine(...)             leaf pruning of improbable branches, then
                                       branch-relative trimming of the boundary.

Runs in the AirMorph environment (scikit-image, scipy).
"""
import json, os
from collections import deque

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
LUT_PATH = os.path.join(HERE, "..", "confidence_lut.json")
LOGIT_PATH = os.path.join(HERE, "..", "branch_logit.json")
CONN26 = np.ones((3, 3, 3), dtype=np.uint8)
MEMBERS = ["member_model1", "member_model2", "member_model3", "member_wingsnet", "member_break"]


# ----------------------------------------------------------------- confidence

def confidence_lut():
    t = json.load(open(LUT_PATH))["table"]
    lut = np.full(256, 0.0, np.float32)
    for r in t:
        if r["p_airway"] is not None:
            lut[r["pattern"]] = r["p_airway"]
    return lut


def confidence(votes, medpseg, tfenet, mask=None):
    """votes: AirMorph member votes 0..5 (uint8); medpseg, tfenet: binary.
    Returns float32 P(airway) per voxel (0 outside `mask` if given)."""
    pat = (np.clip(votes, 0, 5).astype(np.uint8) * 4 + 2 * (medpseg > 0).astype(np.uint8)
           + (tfenet > 0).astype(np.uint8))
    c = confidence_lut()[pat]
    if mask is not None:
        c[~(mask > 0)] = 0.0
    return c


# ----------------------------------------------------------------- tree

def skeleton_parsing(skeleton):
    """Cut a skeleton at its branch points (ATM'22 tree_parse_skelbased)."""
    nb = ndimage.generate_binary_structure(3, 3)
    filt = ndimage.convolve(skeleton, nb) * skeleton
    parse = skeleton.copy(); parse[filt > 3] = 0
    lab, _ = ndimage.label(parse, structure=nb)
    sizes = np.bincount(lab.ravel()); keep = sizes > 1; keep[0] = False
    parse = keep[lab]
    cd, num = ndimage.label(parse, structure=nb)
    return parse, cd, num


def _bbox(mask, pad=2):
    sl = ndimage.find_objects(mask.astype(np.uint8))[0]
    return tuple(slice(max(0, s.start - pad), min(d, s.stop + pad)) for s, d in zip(sl, mask.shape))


def _adjacency(skeleton, parse, cd, num):
    removed, rnum = ndimage.label((skeleton > 0) & (parse == 0), structure=CONN26)
    zc, yc, xc = np.nonzero(skeleton)
    Z, Y, X = skeleton.shape
    lin = (zc.astype(np.int64) * Y + yc) * X + xc
    seg = cd[zc, yc, xc].astype(np.int64); rem = removed[zc, yc, xc].astype(np.int64)
    node = np.where(seg > 0, seg, num + rem)
    adj = {i: set() for i in range(1, num + 1)}
    touch = {r: set() for r in range(num + 1, num + rnum + 1)}
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue
                nz, ny, nx = zc + dz, yc + dy, xc + dx
                ok = (nz >= 0) & (nz < Z) & (ny >= 0) & (ny < Y) & (nx >= 0) & (nx < X)
                nlin = (nz[ok].astype(np.int64) * Y + ny[ok]) * X + nx[ok]
                pos = np.searchsorted(lin, nlin); good = pos < lin.size; pos = pos[good]
                hit = lin[pos] == nlin[good]
                src = np.flatnonzero(ok)[good][hit]
                for u, v in zip(node[src].tolist(), node[pos[hit]].tolist()):
                    if u == v:
                        continue
                    if u <= num and v <= num:
                        adj[u].add(v); adj[v].add(u)
                    elif u <= num:
                        touch[v].add(u)
                    elif v <= num:
                        touch[u].add(v)
    for segs in touch.values():
        segs = sorted(segs)
        for i, u in enumerate(segs):
            for v in segs[i + 1:]:
                adj[u].add(v); adj[v].add(u)
    return adj


def branch_tree(mask, spacing, conf, airmorph, medpseg, tfenet, votes):
    """Returns (branch_map int32, dist_to_skeleton float32 mm, table dict)."""
    mask = mask.astype(bool)
    sl = _bbox(mask)
    m = mask[sl]
    skel = skeletonize(m).astype(np.uint8)
    parse, cd, num = skeleton_parsing(skel)
    if num == 0:
        raise RuntimeError("no branches in the fused mask")
    inds = ndimage.distance_transform_edt(1 - parse, return_distances=False, return_indices=True)
    near = cd[inds[0], inds[1], inds[2]].astype(np.int32); del inds
    near[~m] = 0
    bm = np.zeros(mask.shape, np.int32); bm[sl] = near
    dist = np.zeros(mask.shape, np.float32)
    dist[sl] = ndimage.distance_transform_edt(~skel.astype(bool), sampling=tuple(spacing)) * m

    edt = ndimage.distance_transform_edt(m, sampling=tuple(spacing))
    zc, yc, xc = np.nonzero(parse)
    sid = cd[zc, yc, xc].astype(np.int64)
    n_skel = np.bincount(sid, minlength=num + 1).astype(float)
    r_mean = np.bincount(sid, weights=edt[zc, yc, xc], minlength=num + 1) / np.maximum(n_skel, 1)
    n_vox = np.bincount(near.ravel(), minlength=num + 1).astype(float)
    adj = _adjacency(skel, parse, cd, num)
    root = int(np.argmax(np.where(n_skel > 0, r_mean, -1)[1:]) + 1)
    parent = np.full(num + 1, -1, np.int64); gen = np.zeros(num + 1, np.int64)
    gen[root] = 1; parent[root] = 0
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if gen[v] == 0:
                gen[v] = gen[u] + 1; parent[v] = u; q.append(v)
    gen[1:][gen[1:] == 0] = -1; gen[0] = 0
    n_children = np.bincount(parent[parent > 0], minlength=num + 1)
    vox_mm3 = float(np.prod(spacing))
    length_mm = n_skel * float(np.mean(spacing))
    tube = np.pi * np.maximum(r_mean, 1e-3) ** 2 * np.maximum(length_mm, 1e-3)
    tubularity = n_vox * vox_mm3 / tube
    pr = np.where(parent > 0, r_mean[np.maximum(parent, 0)], r_mean)

    flat = near.ravel(); nz = np.flatnonzero(flat)

    def frac(vol, on_skel):
        v = vol[sl]
        if on_skel:
            return np.bincount(sid, weights=v[zc, yc, xc].astype(float), minlength=num + 1) / np.maximum(n_skel, 1)
        return np.bincount(flat[nz], weights=v.ravel()[nz].astype(float), minlength=num + 1) / np.maximum(n_vox, 1)

    a5_sk = frac(votes.astype(np.float32) / 5.0, True); a5_vx = frac(votes.astype(np.float32) / 5.0, False)
    md_sk, md_vx = frac(medpseg > 0, True), frac(medpseg > 0, False)
    tf_sk, tf_vx = frac(tfenet > 0, True), frac(tfenet > 0, False)
    am_sk = frac(airmorph > 0, True)
    tab = {"parent": parent, "generation": gen, "n_children": n_children, "root": np.zeros(num + 1, bool),
           "n_skel": n_skel, "n_vox": n_vox, "r_mean_mm": r_mean, "r_ratio_parent": r_mean / np.maximum(pr, 1e-3),
           "tubularity": tubularity,
           "conf_skel": (a5_sk + md_sk + tf_sk) / 3, "conf_vox": (a5_vx + md_vx + tf_vx) / 3,
           "n_models": (am_sk >= 0.5).astype(int) + (md_sk >= 0.5) + (tf_sk >= 0.5),
           "skel_airmorph5": a5_sk, "skel_medpseg": md_sk, "skel_tfenet": tf_sk,
           "conf_skel_cal": frac(conf, True), "conf_vox_cal": frac(conf, False)}
    tab["root"][root] = True
    return bm, dist, tab


# ----------------------------------------------------------------- branch probability

def branch_prob(tab):
    """Logistic probability that a branch is airway (fitted on ATM'22 validation)."""
    m = json.load(open(LOGIT_PATH))
    gen = tab["generation"]
    X = np.c_[tab["conf_skel"], tab["conf_vox"], tab["n_models"], tab["skel_airmorph5"], tab["skel_medpseg"],
              tab["skel_tfenet"], np.log1p(tab["n_skel"]), np.clip(tab["r_mean_mm"], 0, 10),
              np.clip(tab["r_ratio_parent"], 0, 3), np.log(np.clip(tab["tubularity"], 0.1, 50)),
              np.clip(np.where(gen > 0, gen, 12), 1, 15), (tab["n_children"] == 0).astype(float)]
    Z = np.c_[np.ones(len(X)), (X - np.array(m["mu"])) / np.array(m["sd"])]
    p = 1 / (1 + np.exp(-Z @ np.array(m["w"])))
    p[0] = 1.0
    return p


def prune(tab, p, tau=0.5, protect_gen=3):
    """Alive flag per branch after repeatedly cutting improbable leaves."""
    num = len(p) - 1
    parent, gen, root = tab["parent"], tab["generation"], tab["root"]
    nchild = np.zeros(num + 1, np.int64)
    for i in range(1, num + 1):
        if parent[i] > 0:
            nchild[parent[i]] += 1
    alive = np.ones(num + 1, bool)
    frontier = [i for i in range(1, num + 1) if nchild[i] == 0]
    while frontier:
        nxt = []
        for i in frontier:
            if not alive[i] or root[i] or nchild[i] > 0 or (gen[i] != -1 and gen[i] <= protect_gen):
                continue
            if p[i] < tau:
                alive[i] = False
                if parent[i] > 0:
                    nchild[parent[i]] -= 1
                    if nchild[parent[i]] == 0:
                        nxt.append(parent[i])
        frontier = nxt
    return alive


def seeded_component(mask, seed, connectivity=6):
    st = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    lab, n = ndimage.label(mask.astype(bool), structure=st)
    if n == 0:
        return np.zeros(mask.shape, np.uint8)
    hit = np.bincount(lab[seed.astype(bool)], minlength=n + 1); hit[0] = 0
    if hit.sum() == 0:
        hit = np.bincount(lab.ravel(), minlength=n + 1); hit[0] = 0
    return (lab == int(hit.argmax())).astype(np.uint8)


def refine(mask, conf, bm, dist, tab, spacing, kappa=0.5, core_frac=0.5, core_min_vox=1.5, max_tau=0.6):
    """Keep a voxel when its probability is at least kappa times the probability
    along its branch's centreline; always keep the core around the centreline."""
    M = mask.astype(bool)
    cs = tab["conf_skel_cal"]
    tau_b = np.clip(kappa * cs, 0.0, max_tau); tau_b[0] = 0.0
    keep = conf >= tau_b.astype(np.float32)[bm]
    r = np.asarray(tab["r_mean_mm"], np.float32).copy(); r[0] = 0
    vox = float(np.max(spacing))
    keep |= dist <= np.maximum(core_frac * r[bm], core_min_vox * vox)
    out = (keep & M).astype(np.uint8)
    seed = (M & (bm > 0) & (conf >= 0.95)).astype(np.uint8)
    return seeded_component(out, seed)
