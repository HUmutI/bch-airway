#!/usr/bin/env python
"""Per-generation segmentation accuracy of the or_all3_lcc fusion on the ATM26 test split.

Builds the ground-truth branch tree with the same parsing the benchmark uses
(training.eval_patch.skeleton_parsing / airway_parsing), assigns a generation to
every ground-truth branch, and scores the or_all3_lcc fusion restricted to each
generation.

Generation assignment
---------------------
* branches           = 26-connected components of the branch-point-pruned GT
                       centreline (`skeleton_parsing`), i.e. the same branch ids
                       `branch_detected_calculation` scores against.
* topology           = two branches are adjacent if their skeleton voxels are
                       26-adjacent, or if both are 26-adjacent to the same
                       connected component of the removed (branch-point / dust)
                       skeleton voxels.
* root               = branch with the largest mean radius, the radius being the
                       Euclidean distance transform of the GT airway (in mm,
                       anisotropic spacing) averaged over the branch's skeleton
                       voxels.  The trachea is the widest airway by a wide
                       margin, and this criterion needs no assumption about the
                       image orientation.  The most-superior-voxel criterion is
                       computed as well and logged for cross-checking.
* generation         = BFS depth from the root + 1 (trachea = generation 1).
  Branches in skeleton components that do not reach the root are reported as
  generation -1 and are never silently dropped.

Restriction of the metrics to a generation
------------------------------------------
`airway_parsing` assigns every voxel of the volume to its nearest branch
(the ground truth restricts that map to the airway).  The same nearest-branch
map is used to assign predicted voxels to a generation, so for generation g

    GT_g   = {airway voxels whose branch has generation g}
    PRED_g = {predicted voxels whose nearest GT branch has generation g}
    DSC_g  = 2|PRED_g n GT_g| / (|PRED_g| + |GT_g|), etc.

The GT_g partition the airway and the PRED_g partition the prediction, and for
any voxel inside the airway the nearest-branch id is its own branch id, so
pooling every generation reproduces the whole-tree DSC / sensitivity /
precision exactly.  That identity is the sanity check printed per case.

The metric formulas (including the 1e-5 smoothing and the rounding) are those of
util.airwayatlas_eval; the branch-detection criterion is the >=80% centreline
coverage of branch_detected_calculation, applied branch by branch.
"""
import argparse, csv, json, os, sys, time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "shim"))

BUNDLE = "/path/to/airway-data/AIRWAY_METHOD_PAPER/airway_paper_bundle"
MASKS = os.path.join(BUNDLE, "08_masks_and_gt")
GT_LABELS = os.path.join(MASKS, "gt_labelsTr")
CACHE = os.path.join(MASKS, "gt_cache_centerlines")

from scipy import ndimage
from training.eval_patch import skeleton_parsing
from util.utils import load_itk_image, restore_image_via_box

MASK_DIRS = {"airmorph": "airmorph_test_masks",
             "medpseg": "medpseg_test_masks",
             "tfenet": "tfenet_BAS_test_masks"}

CONN26 = np.ones((3, 3, 3), dtype=np.uint8)
MAX_GEN_TABLE = 10


# ----------------------------------------------------------------- fusion (verbatim)

def keep_largest_component(mask):
    structure = ndimage.generate_binary_structure(3, 1)
    labelled, count = ndimage.label(mask.astype(bool), structure=structure)
    if count == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = np.bincount(labelled.ravel()); sizes[0] = 0
    return (labelled == sizes.argmax()).astype(np.uint8)


def keep_seeded_component(mask, seed):
    structure = ndimage.generate_binary_structure(3, 1)
    labelled, count = ndimage.label(mask.astype(bool), structure=structure)
    if count == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    seed_labels = labelled[seed.astype(bool)]
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return keep_largest_component(mask)
    return (labelled == int(np.bincount(seed_labels).argmax())).astype(np.uint8)


# ----------------------------------------------------------------- metric formulas

def dsc(inter, n_pred, n_gt, smooth=1e-5):
    return round(((2.0 * inter + smooth) / (n_pred + n_gt + smooth)) * 100, 2)


def sensitivity(inter, n_gt, smooth=1e-5):
    fnr = round((n_gt - inter + smooth) * 100 / (n_gt + smooth), 3)
    return round(100 - fnr, 3)


def precision(inter, n_pred, smooth=1e-5):
    return round((inter + smooth) * 100 / (n_pred + smooth), 3)


# ----------------------------------------------------------------- tree topology

def branch_adjacency(skeleton, skeleton_parse, cd, num):
    """Segment-level adjacency of the parsed skeleton (26-connectivity)."""
    removed_vol, rnum = ndimage.label((skeleton > 0) & (skeleton_parse == 0),
                                      structure=CONN26)
    zc, yc, xc = np.nonzero(skeleton)
    Z, Y, X = skeleton.shape
    lin = (zc.astype(np.int64) * Y + yc) * X + xc          # ascending
    seg = cd[zc, yc, xc].astype(np.int64)
    rem = removed_vol[zc, yc, xc].astype(np.int64)
    node = np.where(seg > 0, seg, num + rem)               # removed -> num+1..num+rnum

    offsets = [(dz, dy, dx)
               for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) > (0, 0, 0)]                # 13 half-offsets

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
        dst = pos[hit]
        pairs.append((node[src], node[dst]))

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


def generations_from_root(adj, num, root):
    gen = np.zeros(num + 1, dtype=np.int64)       # 0 = not reached yet
    gen[root] = 1
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if gen[v] == 0:
                gen[v] = gen[u] + 1
                q.append(v)
    gen[0] = 0
    unreached = [i for i in range(1, num + 1) if gen[i] == 0]
    for i in unreached:
        gen[i] = -1
    return gen, len(unreached)


# ----------------------------------------------------------------- per case

def run_case(case):
    t0 = time.time()
    with open(os.path.join(CACHE, case, "meta.json")) as fh:
        meta = json.load(fh)

    label, _, spacing, _ = load_itk_image(os.path.join(GT_LABELS, case + ".mha"))
    label = (label > 0).astype(np.uint8)

    skel_crop = np.load(os.path.join(CACHE, case, "centerline.npy"))
    skeleton = restore_image_via_box(tuple(meta["original_shape"]),
                                     skel_crop.astype(np.uint8),
                                     np.array(meta["bbox"]))
    skeleton = (skeleton > 0).astype(np.uint8)

    skeleton_parse, cd, num = skeleton_parsing(skeleton)

    # nearest-branch map over the whole volume; * label reproduces airway_parsing
    inds = ndimage.distance_transform_edt(1 - skeleton_parse,
                                          return_distances=False,
                                          return_indices=True)
    full_parsing = cd[inds[0], inds[1], inds[2]]
    del inds

    # ---- root: largest mean radius (mm), computed inside the airway bbox
    zc, yc, xc = np.nonzero(skeleton_parse)
    seg_of_vox = cd[zc, yc, xc].astype(np.int64)
    objs = ndimage.find_objects(label)
    sl = objs[0]
    pad = tuple(slice(max(0, s.start - 1), min(d, s.stop + 1))
                for s, d in zip(sl, label.shape))
    edt_crop = ndimage.distance_transform_edt(label[pad], sampling=tuple(spacing))
    edt_vals = np.zeros(seg_of_vox.shape, dtype=np.float64)
    inside = ((zc >= pad[0].start) & (zc < pad[0].stop) & (yc >= pad[1].start)
              & (yc < pad[1].stop) & (xc >= pad[2].start) & (xc < pad[2].stop))
    edt_vals[inside] = edt_crop[zc[inside] - pad[0].start,
                                yc[inside] - pad[1].start,
                                xc[inside] - pad[2].start]
    del edt_crop
    n_vox = np.bincount(seg_of_vox, minlength=num + 1).astype(np.float64)
    r_sum = np.bincount(seg_of_vox, weights=edt_vals, minlength=num + 1)
    mean_radius = r_sum / np.maximum(n_vox, 1)
    mean_radius[0] = -1
    root_radius = int(np.argmax(mean_radius))
    # cross-check: branch holding the most superior (largest z index) skeleton voxel
    zmax = zc.max()
    root_superior = int(np.bincount(seg_of_vox[zc == zmax]).argmax())

    adj = branch_adjacency(skeleton, skeleton_parse, cd, num)
    gen_of_branch, n_unreached = generations_from_root(adj, num, root_radius)

    # ---- generation map over the volume
    lut = gen_of_branch.astype(np.int16)
    gen_map = lut[full_parsing]
    del full_parsing

    # ---- prediction: or_all3_lcc (OR of the three raw masks, seeded LCC on AirMorph)
    masks = {}
    for name, sub in MASK_DIRS.items():
        with np.load(os.path.join(MASKS, sub, f"ATM26_{case}.npz")) as payload:
            masks[name] = payload["raw"].astype(np.uint8)
        if masks[name].shape != label.shape:
            raise ValueError(f"{case} {name}: {masks[name].shape} != {label.shape}")
    union = masks["airmorph"] | masks["medpseg"] | masks["tfenet"]
    pred = keep_seeded_component(union, masks["airmorph"])
    del union, masks

    # ---- voxel counts per generation
    flat_gen = gen_map.ravel()
    gt_idx = np.flatnonzero(label.ravel())
    pred_idx = np.flatnonzero(pred.ravel())
    inter_idx = np.flatnonzero((label & pred).ravel())
    gens_gt = flat_gen[gt_idx]
    gens_pred = flat_gen[pred_idx]
    gens_int = flat_gen[inter_idx]
    del flat_gen, gt_idx, pred_idx, inter_idx

    def hist(v):
        v = v.astype(np.int64)
        neg = int((v < 0).sum())
        pos = np.bincount(v[v >= 0])
        return pos, neg

    gt_h, gt_neg = hist(gens_gt)
    pr_h, pr_neg = hist(gens_pred)
    in_h, in_neg = hist(gens_int)

    # ---- branch detection per branch (>=80% centreline coverage)
    # label_branch = label_skeleton * label_parsing; on a skeleton_parse voxel the
    # nearest-branch id is the voxel's own branch, so this is cd * label there.
    b_vals = (seg_of_vox * label[zc, yc, xc]).astype(np.int64)
    p_vals = pred[zc, yc, xc]
    lb_counts = np.bincount(b_vals, minlength=num + 1)
    pb_counts = np.bincount(b_vals[p_vals > 0], minlength=num + 1)
    ratio = np.zeros(num + 1, dtype=np.float64)
    nz = lb_counts > 0
    ratio[nz] = pb_counts[nz] / lb_counts[nz]
    detected = (ratio >= 0.8) & nz

    # ---- assemble rows
    gens_present = sorted(set(int(g) for g in gen_of_branch[1:]))
    rows = []
    for g in gens_present:
        sel = np.zeros(num + 1, dtype=bool)
        sel[1:] = gen_of_branch[1:] == g
        n_branch = int(sel.sum())
        # branch_detected_calculation counts branches that have centreline voxels
        sel_sk = sel & nz
        n_branch_sk = int(sel_sk.sum())
        n_det = int((detected & sel).sum())
        if g >= 0:
            n_gt = int(gt_h[g]) if g < gt_h.size else 0
            n_pr = int(pr_h[g]) if g < pr_h.size else 0
            n_in = int(in_h[g]) if g < in_h.size else 0
        else:
            n_gt, n_pr, n_in = gt_neg, pr_neg, in_neg
        rows.append({"case": f"ATM26/{case}", "generation": g,
                     "gt_branches": n_branch, "gt_voxels": n_gt,
                     "DSC": dsc(n_in, n_pr, n_gt),
                     "Sensitivity": sensitivity(n_in, n_gt),
                     "Precision": precision(n_in, n_pr),
                     "branches_detected": n_det,
                     "branches_total": n_branch_sk})

    tot_gt = int(label.sum()); tot_pr = int(pred.sum()); tot_in = int((label & pred).sum())
    pooled_gt = sum(r["gt_voxels"] for r in rows)
    diag = {"case": case,
            "n_branches": num,
            "root_radius": root_radius,
            "root_superior": root_superior,
            "root_agree": root_radius == root_superior,
            "root_mean_radius_mm": round(float(mean_radius[root_radius]), 3),
            "n_unreached_branches": n_unreached,
            "max_gen": int(max(gens_present)),
            "whole_DSC": dsc(tot_in, tot_pr, tot_gt),
            "whole_Sensitivity": sensitivity(tot_in, tot_gt),
            "whole_Precision": precision(tot_in, tot_pr),
            "gt_voxels_total": tot_gt,
            "gt_voxels_pooled": pooled_gt,
            "seconds": round(time.time() - t0, 1)}
    print(f"{case}: {num} branches, root={root_radius} (superior={root_superior}, "
          f"agree={diag['root_agree']}), maxgen={diag['max_gen']}, "
          f"unreached={n_unreached}, whole DSC={diag['whole_DSC']} "
          f"Sens={diag['whole_Sensitivity']} Prec={diag['whole_Precision']} "
          f"({diag['seconds']}s)", flush=True)
    return rows, diag


# ----------------------------------------------------------------- aggregation

def aggregate(rows, out_summary, out_tex, n_cases):
    by_gen = {}
    for r in rows:
        by_gen.setdefault(r["generation"], []).append(r)

    summary = []
    for g in sorted(by_gen):
        rs = by_gen[g]
        det_pct = [100.0 * r["branches_detected"] / r["branches_total"]
                   for r in rs if r["branches_total"] > 0]

        def ms(key):
            v = np.array([r[key] for r in rs], dtype=float)
            return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else float("nan"))
        dm, ds = ms("DSC"); sm, ss = ms("Sensitivity"); pm, ps = ms("Precision")
        dpm = float(np.mean(det_pct)) if det_pct else float("nan")
        dps = float(np.std(det_pct, ddof=1)) if len(det_pct) > 1 else float("nan")
        bm, bs = ms("gt_branches")
        summary.append({"generation": g,
                        "n_scans_with_generation": len(rs),
                        "n_scans_total": n_cases,
                        "gt_branches_total": sum(r["gt_branches"] for r in rs),
                        "gt_branches_mean": round(bm, 2), "gt_branches_sd": round(bs, 2),
                        "gt_voxels_total": sum(r["gt_voxels"] for r in rs),
                        "DSC_mean": round(dm, 2), "DSC_sd": round(ds, 2),
                        "Sensitivity_mean": round(sm, 2), "Sensitivity_sd": round(ss, 2),
                        "Precision_mean": round(pm, 2), "Precision_sd": round(ps, 2),
                        "BranchesDetectedPct_mean": round(dpm, 2),
                        "BranchesDetectedPct_sd": round(dps, 2)})

    with open(out_summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    os.makedirs(os.path.dirname(os.path.abspath(out_tex)), exist_ok=True)
    with open(out_tex, "w") as fh:
        for s in summary:
            g = s["generation"]
            if g < 1 or g > MAX_GEN_TABLE:
                continue
            fh.write("%d & %d & %d & $%.2f \\pm %.2f$ & $%.2f \\pm %.2f$ & "
                     "$%.2f \\pm %.2f$ & $%.2f \\pm %.2f$ \\\\\n"
                     % (g, s["gt_branches_total"], s["n_scans_with_generation"],
                        s["DSC_mean"], s["DSC_sd"],
                        s["Sensitivity_mean"], s["Sensitivity_sd"],
                        s["Precision_mean"], s["Precision_sd"],
                        s["BranchesDetectedPct_mean"], s["BranchesDetectedPct_sd"]))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "per_generation_atm26.csv"))
    ap.add_argument("--summary", default=os.path.join(HERE, "per_generation_atm26_summary.csv"))
    ap.add_argument("--tex", default="/path/to/airway-data/AIRWAY_METHOD_PAPER/"
                                     "paper_fragments/table7_gen_public.tex")
    ap.add_argument("--diag", default=os.path.join(HERE, "per_generation_atm26_diagnostics.csv"))
    args = ap.parse_args()

    cases = args.cases or sorted(d for d in os.listdir(CACHE)
                                 if os.path.isdir(os.path.join(CACHE, d)))
    print(f"{len(cases)} cases, {args.workers} workers", flush=True)

    rows, diags = [], []
    if args.workers > 1 and len(cases) > 1:
        import multiprocessing as mp
        with mp.Pool(min(args.workers, 12)) as pool:
            for r, d in pool.imap_unordered(run_case, cases):
                rows.extend(r); diags.append(d)
    else:
        for c in cases:
            r, d = run_case(c)
            rows.extend(r); diags.append(d)

    rows.sort(key=lambda r: (r["case"], r["generation"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "generation", "gt_branches",
                                           "gt_voxels", "DSC", "Sensitivity",
                                           "Precision", "branches_detected",
                                           "branches_total"])
        w.writeheader(); w.writerows(rows)

    diags.sort(key=lambda d: d["case"])
    with open(args.diag, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diags[0].keys()))
        w.writeheader(); w.writerows(diags)

    summary = aggregate(rows, args.summary, args.tex, len(cases))
    print(f"wrote {len(rows)} rows -> {args.out}")
    for s in summary:
        print(s)


if __name__ == "__main__":
    raise SystemExit(main())
