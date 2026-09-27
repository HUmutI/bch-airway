#!/usr/bin/env python
"""Model-versus-model benchmark on pediatric CT with expert ground truth.

This is the pediatric counterpart of the adult benchmark table. Same six
metrics, same metric code, same combination strategies, so the two tables can be
read side by side. The adult table says which combination wins on adults. This
one is the question the paper actually asks, and it needs ground truth that
adults' public benchmarks cannot supply.

GEOMETRY. The annotations are drawn on the source scan, whose y axis runs
opposite to the grid the member masks were written on. The two differ by exactly
one array flip, which is asserted per case from the direction cosines rather
than assumed, because every number here is meaningless if the two volumes do not
sit on the same voxels.

CENTRELINE. The reference centreline is the TEASAR skeleton, the same function
that built the adult benchmark's cached centrelines, so tree length and branch
detection mean the same thing in both tables.

Usage:
    python bench_pediatric.py --gt-dir /path/to/airway-data/pediatric_gts \\
                              --out pediatric_benchmark.csv
"""
import argparse, csv, glob, os, re, sys, time
import numpy as np
import SimpleITK as sitk

VAL_RUN = "/path/to/airway-data/ATM26_VAL_RUN"
RECOMPUTE = "/path/to/airway-data/paper/recompute_2026-09-16"
sys.path.insert(0, os.path.join(RECOMPUTE, "shim"))
sys.path.insert(0, RECOMPUTE)
sys.path.insert(0, VAL_RUN)

from scipy import ndimage
from training.eval_patch import parse_ground_truth
from recompute import score, keep_seeded_component
from util.airwayatlas_utils import IMRAirwayAtlas_SkeletonExtraction

SEG_ROOT = "/path/to/airway-data/airway-seg"
COHORT = "/path/to/airway-data/cohort90_results/cohort_90.csv"
MEMBERS = {"airmorph": "airway_bin.nii.gz",
           "medpseg": "medpseg_airway.nii.gz",
           "tfenet": "tfenet_airway.nii.gz"}


def case_dir(case_id):
    hits = glob.glob(os.path.join(SEG_ROOT, "*", "*", case_id))
    if len(hits) != 1:
        raise FileNotFoundError(f"{case_id}: {len(hits)} candidate directories")
    return hits[0]


def case_id_of(gt_path):
    """Pull sub-XXXX_acc-YYYY out of whatever the annotator named the file."""
    m = re.search(r"(sub-\d+_acc-\d+)", os.path.basename(gt_path))
    if not m:
        raise ValueError(f"cannot read a case id from {gt_path}")
    return m.group(1)


def aligned_gt(gt_path, ref_path):
    """Return the ground truth as an array on the reference mask's voxel grid."""
    g, r = sitk.ReadImage(gt_path), sitk.ReadImage(ref_path)
    if g.GetSize() != r.GetSize():
        raise ValueError(f"size {g.GetSize()} vs {r.GetSize()}")
    if not np.allclose(g.GetSpacing(), r.GetSpacing(), atol=1e-3):
        raise ValueError(f"spacing {g.GetSpacing()} vs {r.GetSpacing()}")
    dg = np.array(g.GetDirection()).reshape(3, 3)
    dr = np.array(r.GetDirection()).reshape(3, 3)
    arr = sitk.GetArrayFromImage(g) > 0
    diff = np.diag(dg) / np.diag(dr)                 # +1 same, -1 reversed
    if not np.allclose(np.abs(dg), np.abs(dr)) or not np.allclose(np.abs(diff), 1):
        raise ValueError(f"unsupported direction change\n  gt {dg}\n  ref {dr}")
    for axis_sitk, s in enumerate(diff):             # sitk x,y,z -> array z,y,x
        if s < 0:
            arr = np.flip(arr, 2 - axis_sitk)
    return arr


def strategies(a, m, t):
    """The rows of the adult benchmark table, built the same way."""
    v = {"AirMorph": (a, None), "MEDPSeg": (m, None), "TfeNet": (t, None),
         "AirMorph u MEDPSeg": (a | m, a),
         "AirMorph u TfeNet": (a | t, a),
         "MEDPSeg u TfeNet": (m | t, m),
         "All three members": (a | m | t, a)}
    vote = (a.view(np.uint8) + m.view(np.uint8) + t.view(np.uint8)) >= 2
    v["Majority vote"] = (vote, None)
    v["Majority vote, selected"] = (vote, a)
    return v


def run_case(gt_path):
    t0 = time.time()
    cid = case_id_of(gt_path)
    d = case_dir(cid)
    ref = os.path.join(d, MEMBERS["airmorph"])
    gt = aligned_gt(gt_path, ref).astype(np.uint8)
    if not gt.any():
        raise ValueError(f"{cid}: empty ground truth")

    spacing = np.array(sitk.ReadImage(ref).GetSpacing()[::-1])   # to z,y,x
    skel = IMRAirwayAtlas_SkeletonExtraction(gt, spacing)
    if skel is None or not np.asarray(skel).any():
        raise ValueError(f"{cid}: skeletonisation of the ground truth failed")
    parsing, lskel = parse_ground_truth(gt, np.asarray(skel).astype(np.uint8))

    masks = {}
    for k, f in MEMBERS.items():
        img = sitk.ReadImage(os.path.join(d, f))
        masks[k] = sitk.GetArrayFromImage(img) > 0
        if masks[k].shape != gt.shape:
            raise ValueError(f"{cid} {k}: {masks[k].shape} != {gt.shape}")

    rows = []
    for name, (mask, seed) in strategies(masks["airmorph"], masks["medpseg"],
                                         masks["tfenet"]).items():
        mask = np.asarray(mask, dtype=np.uint8)
        pred = mask if seed is None else keep_seeded_component(mask, seed)
        rows.append({"case": cid, "strategy": name,
                     **score(pred, gt, parsing, lskel)})
    print(f"{cid} done  {len(rows)} rows  {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    gts = sorted(glob.glob(os.path.join(args.gt_dir, "**", "*.nii*"), recursive=True))
    if not gts:
        sys.exit(f"no .nii or .nii.gz under {args.gt_dir}")
    print(f"{len(gts)} ground-truth volumes, {args.workers} workers", flush=True)

    rows, failed = [], []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for gt, res in zip(gts, pool.imap(_safe, gts)):
                rows.extend(res) if isinstance(res, list) else failed.append((gt, res))
    else:
        for gt in gts:
            res = _safe(gt)
            rows.extend(res) if isinstance(res, list) else failed.append((gt, res))

    for gt, err in failed:
        print(f"FAILED {os.path.basename(gt)}: {err}", file=sys.stderr)
    if not rows:
        sys.exit("no case succeeded")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows from {len(rows)//9} cases -> {args.out}")
    if failed:
        print(f"{len(failed)} volume(s) failed, listed above")


def _safe(gt):
    try:
        return run_case(gt)
    except Exception as exc:                      # one bad volume must not
        return f"{type(exc).__name__}: {exc}"     # lose the others


if __name__ == "__main__":
    raise SystemExit(main())
