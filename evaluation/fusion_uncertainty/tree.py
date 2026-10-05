"""Branch graph of a binary airway mask, and per-branch evidence.

The fused mask is skeletonised, the skeleton is cut at its branch points into
branches (the same `skeleton_parsing` the benchmark scores with), every mask
voxel is assigned to its nearest branch, the branches are joined into a tree
rooted at the widest branch (the trachea), and for every branch we record what
the seven predictors say about it and how it sits in the tree.

All work is done inside the mask's bounding box.
"""
from collections import deque

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from training.eval_patch import skeleton_parsing
from common import AIRMORPH_MEMBERS

CONN26 = np.ones((3, 3, 3), dtype=np.uint8)


def bbox_slices(mask, pad=2):
    objs = ndimage.find_objects(mask.astype(np.uint8))
    sl = objs[0]
    return tuple(slice(max(0, s.start - pad), min(d, s.stop + pad)) for s, d in zip(sl, mask.shape))


def branch_adjacency(skeleton, skeleton_parse, cd, num):
    """Branch-level adjacency (26-connectivity), via the removed branch-point
    voxels. Verbatim from benchmark_adult/per_generation.py."""
    removed_vol, rnum = ndimage.label((skeleton > 0) & (skeleton_parse == 0), structure=CONN26)
    zc, yc, xc = np.nonzero(skeleton)
    Z, Y, X = skeleton.shape
    lin = (zc.astype(np.int64) * Y + yc) * X + xc
    seg = cd[zc, yc, xc].astype(np.int64)
    rem = removed_vol[zc, yc, xc].astype(np.int64)
    node = np.where(seg > 0, seg, num + rem)
    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) > (0, 0, 0)]
    pairs = []
    for dz, dy, dx in offsets:
        nz, ny, nx = zc + dz, yc + dy, xc + dx
        ok = (nz >= 0) & (nz < Z) & (ny >= 0) & (ny < Y) & (nx >= 0) & (nx < X)
        nlin = (nz[ok].astype(np.int64) * Y + ny[ok]) * X + nx[ok]
        pos = np.searchsorted(lin, nlin)
        good = pos < lin.size
        pos = pos[good]
        hit = lin[pos] == nlin[good]
        src = np.flatnonzero(ok)[good][hit]
        pairs.append((node[src], node[pos[hit]]))
    adj = {i: set() for i in range(1, num + 1)}
    touch = {r: set() for r in range(num + 1, num + rnum + 1)}
    for a, b in pairs:
        for u, v in zip(a.tolist(), b.tolist()):
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


def bfs_tree(adj, num, root):
    """Parent and generation of every branch (root = generation 1); -1 when
    not reachable from the root."""
    parent = np.full(num + 1, -1, np.int64)
    gen = np.zeros(num + 1, np.int64)
    gen[root] = 1; parent[root] = 0
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if gen[v] == 0:
                gen[v] = gen[u] + 1; parent[v] = u; q.append(v)
    gen[1:][gen[1:] == 0] = -1
    gen[0] = 0
    return parent, gen


def build_tree(mask, spacing, members, gt=None):
    """Return (branch_map, table). branch_map: int32 volume, branch id per mask
    voxel (0 elsewhere). table: dict of per-branch arrays indexed by branch id
    (index 0 unused)."""
    mask = mask.astype(bool)
    sl = bbox_slices(mask)
    m = mask[sl]
    skel = skeletonize(m).astype(np.uint8)
    skel_parse, cd, num = skeleton_parsing(skel)
    if num == 0:
        raise RuntimeError("no branches")
    # nearest branch for every mask voxel
    inds = ndimage.distance_transform_edt(1 - skel_parse, return_distances=False, return_indices=True)
    near = cd[inds[0], inds[1], inds[2]].astype(np.int32)
    del inds
    near[~m] = 0
    branch_map = np.zeros(mask.shape, np.int32)
    branch_map[sl] = near

    # geometry: radius from the EDT of the mask, in mm
    edt = ndimage.distance_transform_edt(m, sampling=tuple(spacing))
    zc, yc, xc = np.nonzero(skel_parse)
    sid = cd[zc, yc, xc].astype(np.int64)
    n_skel = np.bincount(sid, minlength=num + 1).astype(float)
    r_mean = np.bincount(sid, weights=edt[zc, yc, xc], minlength=num + 1) / np.maximum(n_skel, 1)
    r_max = np.zeros(num + 1); np.maximum.at(r_max, sid, edt[zc, yc, xc])
    n_vox = np.bincount(near.ravel(), minlength=num + 1).astype(float)
    # mean of z (array axis 0) per branch, for a cranio-caudal cue
    z_mean = np.bincount(sid, weights=zc.astype(float), minlength=num + 1) / np.maximum(n_skel, 1)

    adj = branch_adjacency(skel, skel_parse, cd, num)
    root = int(np.argmax(np.where(n_skel > 0, r_mean, -1)[1:]) + 1)
    parent, gen = bfs_tree(adj, num, root)
    n_children = np.bincount(parent[parent > 0], minlength=num + 1)
    # subtree size (branches) by reverse BFS order
    order = sorted(range(1, num + 1), key=lambda i: -gen[i])
    sub = np.ones(num + 1, np.int64)
    for i in order:
        if parent[i] > 0:
            sub[parent[i]] += sub[i]

    # evidence per branch: fraction of skeleton voxels and of all voxels each
    # predictor covers, and the voxel confidence
    tab = {"id": np.arange(num + 1), "root": np.zeros(num + 1, bool),
           "parent": parent, "generation": gen, "n_children": n_children,
           "subtree": sub, "n_skel": n_skel, "n_vox": n_vox,
           "r_mean_mm": r_mean, "r_max_mm": r_max, "z_mean": z_mean}
    tab["root"][root] = True
    vox_mm3 = float(np.prod(spacing))
    tab["length_mm"] = n_skel * float(np.mean(spacing))       # coarse but monotone
    tab["volume_mm3"] = n_vox * vox_mm3
    tube = np.pi * np.maximum(r_mean, 1e-3) ** 2 * np.maximum(tab["length_mm"], 1e-3)
    tab["tubularity"] = tab["volume_mm3"] / tube                 # ~1 for a tube, >>1 for a blob
    pr = np.where(parent > 0, r_mean[np.maximum(parent, 0)], r_mean)
    tab["r_ratio_parent"] = r_mean / np.maximum(pr, 1e-3)

    a5_sk = np.zeros(num + 1); a5_vx = np.zeros(num + 1)
    flat_near = near.ravel()
    nz_idx = np.flatnonzero(flat_near)
    for k in AIRMORPH_MEMBERS + ["medpseg", "tfenet"]:
        mk = members[k][sl]
        f_sk = np.bincount(sid, weights=mk[zc, yc, xc].astype(float), minlength=num + 1) / np.maximum(n_skel, 1)
        f_vx = np.bincount(flat_near[nz_idx], weights=mk.ravel()[nz_idx].astype(float), minlength=num + 1) / np.maximum(n_vox, 1)
        tab[f"skel_{k}"] = f_sk; tab[f"vox_{k}"] = f_vx
        if k in AIRMORPH_MEMBERS:
            a5_sk += f_sk / 5; a5_vx += f_vx / 5
    tab["skel_airmorph5"] = a5_sk; tab["vox_airmorph5"] = a5_vx
    tab["conf_skel"] = (a5_sk + tab["skel_medpseg"] + tab["skel_tfenet"]) / 3
    tab["conf_vox"] = (a5_vx + tab["vox_medpseg"] + tab["vox_tfenet"]) / 3
    # how many of the three models cover at least half of the centreline
    am_sk = np.bincount(sid, weights=members["airmorph"][sl][zc, yc, xc].astype(float), minlength=num + 1) / np.maximum(n_skel, 1)
    tab["skel_airmorph"] = am_sk
    tab["n_models"] = (am_sk >= 0.5).astype(int) + (tab["skel_medpseg"] >= 0.5) + (tab["skel_tfenet"] >= 0.5)

    if gt is not None:
        g = gt[sl].astype(bool)
        tab["gt_skel_frac"] = np.bincount(sid, weights=g[zc, yc, xc].astype(float), minlength=num + 1) / np.maximum(n_skel, 1)
        tab["gt_vox_frac"] = np.bincount(flat_near[nz_idx], weights=g.ravel()[nz_idx].astype(float), minlength=num + 1) / np.maximum(n_vox, 1)
        # tolerant version: within 1 voxel of the GT
        gd = ndimage.binary_dilation(g, structure=CONN26)
        tab["gt_skel_frac_dil"] = np.bincount(sid, weights=gd[zc, yc, xc].astype(float), minlength=num + 1) / np.maximum(n_skel, 1)
    return branch_map, tab


def prune(mask, branch_map, tab, keep_prob, tau, protect_gen=3):
    """Topology-preserving pruning: repeatedly remove leaf branches whose
    probability of being airway is below tau. Branches of generation <=
    protect_gen and the root are never removed. Returns the pruned mask and
    the set of removed branch ids."""
    num = len(tab["id"]) - 1
    parent = tab["parent"].copy()
    children = {i: set() for i in range(num + 1)}
    for i in range(1, num + 1):
        if parent[i] > 0:
            children[parent[i]].add(i)
    alive = np.ones(num + 1, bool); alive[0] = False
    removed = set()
    changed = True
    while changed:
        changed = False
        for i in range(1, num + 1):
            if not alive[i] or tab["root"][i] or children[i]:
                continue
            if tab["generation"][i] != -1 and tab["generation"][i] <= protect_gen:
                continue
            if keep_prob[i] < tau:
                alive[i] = False; removed.add(i); changed = True
                if parent[i] > 0:
                    children[parent[i]].discard(i)
    # branches not connected to the root (generation -1) with low prob go too
    out = mask.astype(bool).copy()
    if removed:
        lut = np.ones(num + 1, bool); lut[list(removed)] = False
        out &= lut[branch_map]
    return out.astype(np.uint8), removed
