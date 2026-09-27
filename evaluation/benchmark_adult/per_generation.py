#!/usr/bin/env python
"""Per-generation segmentation accuracy on the ATM26 test split: the or_all3_lcc
fusion and each member alone (raw, as the fusion consumes it, before the
model's own post-processing; the same convention as Table 1).

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

BUNDLE = os.path.join(os.environ.get("AIRWAY_ROOT", "/path/to/airway-data"),
                      "paper", "airway_paper_bundle")
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

# What is scored, from the three raw member masks. Order is the table order.
STRATEGIES = {
    "airmorph_raw": lambda m: m["airmorph"],
    "medpseg_raw":  lambda m: m["medpseg"],
    "tfenet_raw":   lambda m: m["tfenet"],
    "or_all3_lcc":  lambda m: keep_seeded_component(m["airmorph"] | m["medpseg"] | m["tfenet"],
                                                    m["airmorph"]),
}
LABEL = {"airmorph_raw": "AirMorph", "medpseg_raw": "MEDPSeg", "tfenet_raw": "TfeNet",
         "or_all3_lcc": "All three"}


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

strategies = list(STRATEGIES)          # narrowed by --strategies; read by the workers


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

    # ---- the three raw member masks; every strategy is built from them
    masks = {}
    for name, sub in MASK_DIRS.items():
        with np.load(os.path.join(MASKS, sub, f"ATM26_{case}.npz")) as payload:
            masks[name] = payload["raw"].astype(np.uint8)
        if masks[name].shape != label.shape:
            raise ValueError(f"{case} {name}: {masks[name].shape} != {label.shape}")

    # GT-side quantities are strategy independent
    flat_gen = gen_map.ravel()
    gens_gt = flat_gen[np.flatnonzero(label.ravel())]

    def hist(v):
        v = v.astype(np.int64)
        neg = int((v < 0).sum())
        pos = np.bincount(v[v >= 0])
        return pos, neg

    gt_h, gt_neg = hist(gens_gt)
    # label_branch = label_skeleton * label_parsing; on a skeleton_parse voxel the
    # nearest-branch id is the voxel's own branch, so this is cd * label there.
    b_vals = (seg_of_vox * label[zc, yc, xc]).astype(np.int64)
    lb_counts = np.bincount(b_vals, minlength=num + 1)
    nz = lb_counts > 0
    # generation of every GT skeleton voxel, for tree length per generation
    # (tree_length_calculation: covered centreline voxels / centreline voxels)
    sk_gen = gen_of_branch[b_vals]
    sk_gen[b_vals == 0] = 0
    gens_present = sorted(set(int(g) for g in gen_of_branch[1:]))
    tot_gt = int(label.sum())

    rows, diags = [], []
    for strategy in strategies:
        pred = STRATEGIES[strategy](masks)

        # ---- voxel counts per generation
        gens_pred = flat_gen[np.flatnonzero(pred.ravel())]
        gens_int = flat_gen[np.flatnonzero((label & pred).ravel())]
        pr_h, pr_neg = hist(gens_pred)
        in_h, in_neg = hist(gens_int)

        # ---- branch detection per branch (>=80% centreline coverage)
        p_vals = pred[zc, yc, xc]
        pb_counts = np.bincount(b_vals[p_vals > 0], minlength=num + 1)
        ratio = np.zeros(num + 1, dtype=np.float64)
        ratio[nz] = pb_counts[nz] / lb_counts[nz]
        detected = (ratio >= 0.8) & nz
        on_skel = b_vals > 0

        # ---- assemble rows
        for g in gens_present:
            sel = np.zeros(num + 1, dtype=bool)
            sel[1:] = gen_of_branch[1:] == g
            n_branch = int(sel.sum())
            # branch_detected_calculation counts branches that have centreline voxels
            n_branch_sk = int((sel & nz).sum())
            n_det = int((detected & sel).sum())
            sk_sel = on_skel & (sk_gen == g)
            n_sk = int(sk_sel.sum())
            tld = round(100.0 * int((p_vals[sk_sel] > 0).sum()) / n_sk, 3) if n_sk else float("nan")
            if g >= 0:
                n_gt = int(gt_h[g]) if g < gt_h.size else 0
                n_pr = int(pr_h[g]) if g < pr_h.size else 0
                n_in = int(in_h[g]) if g < in_h.size else 0
            else:
                n_gt, n_pr, n_in = gt_neg, pr_neg, in_neg
            rows.append({"case": f"ATM26/{case}", "strategy": strategy, "generation": g,
                         "gt_branches": n_branch, "gt_voxels": n_gt,
                         "DSC": dsc(n_in, n_pr, n_gt),
                         "Sensitivity": sensitivity(n_in, n_gt),
                         "Precision": precision(n_in, n_pr),
                         "TLD": tld, "skeleton_voxels": n_sk,
                         "branches_detected": n_det,
                         "branches_total": n_branch_sk})

        tot_pr = int(pred.sum()); tot_in = int((label & pred).sum())
        diag = {"case": case, "strategy": strategy,
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
                "whole_TLD": round(100.0 * int((p_vals[on_skel] > 0).sum()) / int(on_skel.sum()), 3),
                "whole_BND": round(100.0 * int(detected.sum()) / int(nz.sum()), 3),
                "gt_voxels_total": tot_gt,
                "gt_voxels_pooled": sum(r["gt_voxels"] for r in rows if r["strategy"] == strategy),
                "seconds": round(time.time() - t0, 1)}
        diags.append(diag)
        print(f"{case} {strategy}: {num} branches, root={root_radius} (superior={root_superior}, "
              f"agree={diag['root_agree']}), maxgen={diag['max_gen']}, "
              f"unreached={n_unreached}, whole DSC={diag['whole_DSC']} TLD={diag['whole_TLD']} "
              f"BND={diag['whole_BND']} Sens={diag['whole_Sensitivity']} Prec={diag['whole_Precision']} "
              f"({diag['seconds']}s)", flush=True)
    return rows, diags


# ----------------------------------------------------------------- aggregation

def summarise(rows, n_cases):
    """One row per (strategy, generation): mean and SD over the volumes that
    contain the generation, plus branch counts."""
    by = {}
    for r in rows:
        by.setdefault((r["strategy"], r["generation"]), []).append(r)
    summary = []
    for (strategy, g) in sorted(by, key=lambda k: (list(STRATEGIES).index(k[0]), k[1])):
        rs = by[(strategy, g)]
        det_pct = [100.0 * r["branches_detected"] / r["branches_total"]
                   for r in rs if r["branches_total"] > 0]

        def ms(key):
            v = np.array([r[key] for r in rs], dtype=float)
            return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else float("nan"))
        dm, ds = ms("DSC"); sm, ss = ms("Sensitivity"); pm, ps = ms("Precision")
        tl = np.array([r["TLD"] for r in rs if r["skeleton_voxels"] > 0], dtype=float)
        tm, ts = (float(tl.mean()), float(tl.std(ddof=1)) if tl.size > 1 else float("nan")) if tl.size else (float("nan"),) * 2
        dpm = float(np.mean(det_pct)) if det_pct else float("nan")
        dps = float(np.std(det_pct, ddof=1)) if len(det_pct) > 1 else float("nan")
        bm, bs = ms("gt_branches")
        summary.append({"strategy": strategy, "generation": g,
                        "n_scans_with_generation": len(rs),
                        "n_scans_total": n_cases,
                        "gt_branches_total": sum(r["gt_branches"] for r in rs),
                        "gt_branches_mean": round(bm, 2), "gt_branches_sd": round(bs, 2),
                        "gt_voxels_total": sum(r["gt_voxels"] for r in rs),
                        "DSC_mean": round(dm, 2), "DSC_sd": round(ds, 2),
                        "Sensitivity_mean": round(sm, 2), "Sensitivity_sd": round(ss, 2),
                        "Precision_mean": round(pm, 2), "Precision_sd": round(ps, 2),
                        "TLD_mean": round(tm, 2), "TLD_sd": round(ts, 2),
                        "BranchesDetectedPct_mean": round(dpm, 2),
                        "BranchesDetectedPct_sd": round(dps, 2)})
    return summary


def whole_tree(diags):
    """Mean and SD of the whole-tree metrics per strategy, from the diagnostics."""
    out = {}
    for strategy in STRATEGIES:
        ds = [d for d in diags if d["strategy"] == strategy]
        if not ds:
            continue
        out[strategy] = {k: (float(np.mean([d[k] for d in ds])), float(np.std([d[k] for d in ds], ddof=1)))
                         for k in ("whole_DSC", "whole_TLD", "whole_BND", "whole_Sensitivity", "whole_Precision")}
    return out


def write_tex(summary, whole, out_tex):
    """Table body: one block per generation, one line per strategy. The
    caller's tabular is  l l c c c c c  (Gen, Strategy, DSC, TLD, BND,
    Sensitivity, Precision); the generation is printed on the first line of
    each block only, and the branch / volume counts per generation are
    printed as a comment line for the caption. TLD and BND
    are the per-generation forms of Table 1's tree length detected and branch
    detected: centreline voxels covered, and branches with >=80% of their
    centreline covered, restricted to the branches of that generation."""
    order = [k for k in STRATEGIES if any(r["strategy"] == k for r in summary)]
    gens = sorted({r["generation"] for r in summary if 1 <= r["generation"] <= MAX_GEN_TABLE})
    os.makedirs(os.path.dirname(os.path.abspath(out_tex)), exist_ok=True)

    def best(rows, key):
        return max(r[key] for r in rows)

    with open(out_tex, "w") as fh:
        first = [next(r for r in summary if r["strategy"] == order[0] and r["generation"] == g) for g in gens]
        fh.write("%% reference branches per generation: %s; volumes containing it: %s\n"
                 % (", ".join(str(r["gt_branches_total"]) for r in first),
                    ", ".join(str(r["n_scans_with_generation"]) for r in first)))
        for g in gens:
            block = [next(r for r in summary if r["strategy"] == k and r["generation"] == g) for k in order]
            for i, (k, r) in enumerate(zip(order, block)):
                cells = []
                for key in ("DSC", "TLD", "BranchesDetectedPct", "Sensitivity", "Precision"):
                    v = "$%.2f \\pm %.2f$" % (r[key + "_mean"], r[key + "_sd"])
                    if r[key + "_mean"] == best(block, key + "_mean"):
                        v = "$\\mathbf{%.2f} \\pm %.2f$" % (r[key + "_mean"], r[key + "_sd"])
                    cells.append(v)
                head = ("%d & %s" % (g, LABEL[k])) if i == 0 else " & %s" % LABEL[k]
                fh.write("%s & %s \\\\\n" % (head, " & ".join(cells)))
            fh.write("\\addlinespace[2pt]\n" if g != gens[-1] else "")
        if whole:
            fh.write("\\midrule\n")
            for i, k in enumerate(order):
                if k not in whole:
                    continue
                w = whole[k]
                cells = []
                for key in ("whole_DSC", "whole_TLD", "whole_BND", "whole_Sensitivity", "whole_Precision"):
                    m, sd = w[key]
                    b = max(whole[j][key][0] for j in order if j in whole)
                    cells.append(("$\\mathbf{%.2f} \\pm %.2f$" if m == b else "$%.2f \\pm %.2f$") % (m, sd))
                head = ("\\multicolumn{1}{@{}l}{Whole tree} & %s" % LABEL[k]) if i == 0 else " & %s" % LABEL[k]
                fh.write("%s & %s \\\\\n" % (head, " & ".join(cells)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES), choices=list(STRATEGIES))
    ap.add_argument("--out", default=os.path.join(HERE, "per_generation_atm26_members.csv"))
    ap.add_argument("--summary", default=os.path.join(HERE, "per_generation_atm26_members_summary.csv"))
    ap.add_argument("--tex", default=os.path.join(HERE, "per_generation_atm26_members.tex"))
    ap.add_argument("--diag", default=os.path.join(HERE, "per_generation_atm26_members_diagnostics.csv"))
    args = ap.parse_args()
    global strategies
    strategies = [k for k in STRATEGIES if k in args.strategies]

    cases = args.cases or sorted(d for d in os.listdir(CACHE)
                                 if os.path.isdir(os.path.join(CACHE, d)))
    print(f"{len(cases)} cases, {args.workers} workers", flush=True)

    rows, diags = [], []
    if args.workers > 1 and len(cases) > 1:
        import multiprocessing as mp
        with mp.Pool(min(args.workers, 12)) as pool:
            for r, d in pool.imap_unordered(run_case, cases):
                rows.extend(r); diags.extend(d)
    else:
        for c in cases:
            r, d = run_case(c)
            rows.extend(r); diags.extend(d)

    rows.sort(key=lambda r: (r["case"], list(STRATEGIES).index(r["strategy"]), r["generation"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "strategy", "generation", "gt_branches",
                                           "gt_voxels", "DSC", "Sensitivity",
                                           "Precision", "TLD", "skeleton_voxels",
                                           "branches_detected", "branches_total"])
        w.writeheader(); w.writerows(rows)

    diags.sort(key=lambda d: (d["case"], list(STRATEGIES).index(d["strategy"])))
    with open(args.diag, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diags[0].keys()))
        w.writeheader(); w.writerows(diags)

    summary = summarise(rows, len(cases))
    with open(args.summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    write_tex(summary, whole_tree(diags), args.tex)
    print(f"wrote {len(rows)} rows -> {args.out}; table body -> {args.tex}")
    for s in summary:
        if 1 <= s["generation"] <= MAX_GEN_TABLE:
            print(f'{s["strategy"]:14s} gen {s["generation"]:2d}  DSC {s["DSC_mean"]:6.2f}  TLD {s["TLD_mean"]:6.2f}  '
                  f'BND {s["BranchesDetectedPct_mean"]:6.2f}  Sens {s["Sensitivity_mean"]:6.2f}  Prec {s["Precision_mean"]:6.2f}  n={s["n_scans_with_generation"]}')


if __name__ == "__main__":
    raise SystemExit(main())
