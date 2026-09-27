#!/usr/bin/env python
"""Surface distances for the strategies of the two benchmark tables.

The overlap and topology metrics say how much of the tree is found. They say
nothing about how far a wrong voxel is from the truth, which is what a surface
distance adds.

Three are computed. HD is the largest distance from either surface to the other
and is decided by a single voxel, so on an unpruned mask a slab in the chest
wall sets it and the number stops describing the airway. HD95 is the same
quantity at the 95th percentile and is the one to read. ASSD is the mean, which
is dominated by the proximal airway because that is where most of the surface
is.

Distances are in millimetres, from the voxel spacing, not in voxels.

    python hausdorff.py --split test --out hd_test.csv
    python hausdorff.py --split pediatric --gt-dir <dir> --out hd_ped.csv
"""
import argparse, csv, json, os, sys, time
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
RECOMPUTE = "/path/to/airway-data/paper/recompute_2026-09-16"
sys.path.insert(0, os.path.join(RECOMPUTE, "shim"))
sys.path.insert(0, RECOMPUTE)
sys.path.insert(0, HERE)
from recompute import keep_seeded_component, MASKS, GT_LABELS, CACHE, MASK_DIRS
from util.utils import load_itk_image

ST6 = ndimage.generate_binary_structure(3, 1)


def surface(mask):
    """Voxels of the mask that touch the background under 6-connectivity."""
    return mask & ~ndimage.binary_erosion(mask, ST6, border_value=0)


def distances(pred, gt, spacing):
    """HD, HD95 and ASSD in millimetres between the two surfaces."""
    sp, sg = surface(pred.astype(bool)), surface(gt.astype(bool))
    if not sp.any() or not sg.any():
        return {"HD": float("nan"), "HD95": float("nan"), "ASSD": float("nan")}
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d_pg, d_gp = dt_g[sp], dt_p[sg]
    both = np.concatenate([d_pg, d_gp])
    return {"HD": float(max(d_pg.max(), d_gp.max())),
            "HD95": float(np.percentile(both, 95)),
            "ASSD": float(both.mean())}


def strategies(a, m, t):
    v = {"AirMorph": (a, None), "MEDPSeg": (m, None), "TfeNet": (t, None),
         "AirMorph u MEDPSeg": (a | m, a),
         "AirMorph u TfeNet": (a | t, a),
         "MEDPSeg u TfeNet": (m | t, m),
         "All three members": (a | m | t, a)}
    vote = (a.view(np.uint8) + m.view(np.uint8) + t.view(np.uint8)) >= 2
    v["Majority vote"] = (vote, None)
    v["Majority vote, selected"] = (vote, a)
    return v


def run_test(case):
    t0 = time.time()
    with open(os.path.join(CACHE, case, "meta.json")) as fh:
        spacing = np.array(json.load(fh)["spacing"], dtype=float)
    gt, _, _, _ = load_itk_image(os.path.join(GT_LABELS, case + ".mha"))
    gt = (gt > 0).astype(np.uint8)
    masks = {}
    for name, sub in MASK_DIRS.items():
        with np.load(os.path.join(MASKS, sub, f"ATM26_{case}.npz")) as p:
            masks[name] = p["raw"].astype(bool)
    rows = []
    for name, (mask, seed) in strategies(masks["airmorph"], masks["medpseg"],
                                         masks["tfenet"]).items():
        mask = np.asarray(mask, dtype=np.uint8)
        pred = mask if seed is None else keep_seeded_component(mask, seed)
        rows.append({"case": f"ATM26/{case}", "strategy": name,
                     **distances(pred, gt, spacing)})
    print(f"{case} done  {time.time()-t0:.0f}s", flush=True)
    return rows


# The pediatric helpers are restated here rather than imported from
# bench_pediatric. That module also pulls in the skeletoniser, whose package
# shares the name `util` with the metric shim on this path, and the shim wins.
SEG_ROOT = "/path/to/airway-data/airway-seg"
MEMBERS = {"airmorph": "airway_bin.nii.gz",
           "medpseg": "medpseg_airway.nii.gz",
           "tfenet": "tfenet_airway.nii.gz"}


def case_id_of(gt_path):
    import re
    m = re.search(r"(sub-\d+_acc-\d+)", os.path.basename(gt_path))
    if not m:
        raise ValueError(f"cannot read a case id from {gt_path}")
    return m.group(1)


def case_dir(case_id):
    import glob
    hits = glob.glob(os.path.join(SEG_ROOT, "*", "*", case_id))
    if len(hits) != 1:
        raise FileNotFoundError(f"{case_id}: {len(hits)} candidate directories")
    return hits[0]


def aligned_gt(gt_path, ref_path):
    """Ground truth on the reference mask's voxel grid, flips checked not assumed."""
    import SimpleITK as sitk
    g, r = sitk.ReadImage(gt_path), sitk.ReadImage(ref_path)
    if g.GetSize() != r.GetSize():
        raise ValueError(f"size {g.GetSize()} vs {r.GetSize()}")
    dg = np.array(g.GetDirection()).reshape(3, 3)
    dr = np.array(r.GetDirection()).reshape(3, 3)
    arr = sitk.GetArrayFromImage(g) > 0
    diff = np.diag(dg) / np.diag(dr)
    if not np.allclose(np.abs(dg), np.abs(dr)) or not np.allclose(np.abs(diff), 1):
        raise ValueError("unsupported direction change")
    for axis_sitk, sgn in enumerate(diff):
        if sgn < 0:
            arr = np.flip(arr, 2 - axis_sitk)
    return arr


def run_ped(gt_path):
    import SimpleITK as sitk
    t0 = time.time()
    cid = case_id_of(gt_path)
    d = case_dir(cid)
    ref = os.path.join(d, MEMBERS["airmorph"])
    gt = aligned_gt(gt_path, ref).astype(np.uint8)
    spacing = np.array(sitk.ReadImage(ref).GetSpacing()[::-1], dtype=float)
    masks = {k: sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(d, f))) > 0
             for k, f in MEMBERS.items()}
    rows = []
    for name, (mask, seed) in strategies(masks["airmorph"], masks["medpseg"],
                                         masks["tfenet"]).items():
        mask = np.asarray(mask, dtype=np.uint8)
        pred = mask if seed is None else keep_seeded_component(mask, seed)
        rows.append({"case": cid, "strategy": name,
                     **distances(pred, gt, spacing)})
    print(f"{cid} done  {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("test", "pediatric"), required=True)
    ap.add_argument("--gt-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    if args.split == "test":
        items = sorted(d for d in os.listdir(CACHE)
                       if os.path.isdir(os.path.join(CACHE, d)))
        fn = run_test
    else:
        import glob
        items = sorted(glob.glob(os.path.join(args.gt_dir, "**", "*.nii*"),
                                 recursive=True))
        fn = run_ped
    print(f"{len(items)} cases, {args.workers} workers", flush=True)

    rows = []
    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(fn, items):
            rows.extend(r)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
