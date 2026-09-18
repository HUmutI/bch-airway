#!/usr/bin/env python
"""Score candidate combination rules on the ATM26 VALIDATION split.

WHY A VALIDATION SPLIT. The rule the paper adopts has to be chosen somewhere
other than the split it is reported on. Only test masks survived from the
original machine, so the 30 validation volumes were re-inferred locally with
the same seven checkpoints (all MD5-matched against the originals). Whatever
wins here is then run once on test and reported, with no further tuning.

WHAT IS AVAILABLE TO FUSE. Seven binary maps per case, not three. AirMorph
ships as a five-member ensemble and the dump keeps each member, so the usable
predictors are model1, model2, model3, wingsnet, break, plus MEDPSeg and
TfeNet. A union of three is only one point on a much longer agreement curve,
and the point of this sweep is to find where on that curve the six reported
metrics are jointly best.

Metric functions and the component-selection rules are imported from the
recompute driver, unmodified, so validation and test numbers are commensurable.
"""
import argparse, csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECOMPUTE = "/path/to/airway-data/AIRWAY_METHOD_PAPER/recompute_2026-09-16"
sys.path.insert(0, os.path.join(RECOMPUTE, "shim"))
sys.path.insert(0, RECOMPUTE)

from scipy import ndimage
from training.eval_patch import parse_ground_truth
from util.utils import load_itk_image, restore_image_via_box
from recompute import score, keep_seeded_component, keep_largest_component

VAL_ROOT = "/path/to/airway-data/ATM26_VAL_RUN"
TEST_ROOT = ("/path/to/airway-data/AIRWAY_METHOD_PAPER/"
             "airway_paper_bundle/08_masks_and_gt")

# The two splits live in different trees because the test masks came from the
# original machine and the validation masks were made here. The checkpoints
# behind them are the same seven files, MD5-matched, so the numbers compare.
LAYOUT = {
    "val": {"root": VAL_ROOT,
            "cache": os.path.join(VAL_ROOT, "data/cache/ATM26"),
            "labels": os.path.join(VAL_ROOT, "data/ATM26/labelsTr"),
            "dirs": {"airmorph": "outputs/airmorph/val_masks",
                     "medpseg": "outputs/medpseg/val_masks",
                     "tfenet": "outputs/tfenet/BAS_val_masks"}},
    "test": {"root": TEST_ROOT,
             "cache": os.path.join(TEST_ROOT, "gt_cache_centerlines"),
             "labels": os.path.join(TEST_ROOT, "gt_labelsTr"),
             "dirs": {"airmorph": "airmorph_test_masks",
                      "medpseg": "medpseg_test_masks",
                      "tfenet": "tfenet_BAS_test_masks"}},
}
ROOT = CACHE = GT_LABELS = None
OUT_DIRS = {}


def use_split(split):
    """Point the module at one split. Called once in each worker at import."""
    global ROOT, CACHE, GT_LABELS, OUT_DIRS
    cfg = LAYOUT[split]
    ROOT, CACHE, GT_LABELS, OUT_DIRS = (cfg["root"], cfg["cache"],
                                        cfg["labels"], cfg["dirs"])


MEMBERS = ["member_model1", "member_model2", "member_model3",
           "member_wingsnet", "member_break"]


def load_case_masks(case):
    """Return the three fused member maps plus AirMorph's five raw members."""
    out = {}
    with np.load(os.path.join(ROOT, OUT_DIRS["airmorph"], f"ATM26_{case}.npz")) as p:
        out["airmorph"] = p["raw"].astype(bool)
        for m in MEMBERS:
            out[m] = p[m].astype(bool)
    for name in ("medpseg", "tfenet"):
        with np.load(os.path.join(ROOT, OUT_DIRS[name], f"ATM26_{case}.npz")) as p:
            out[name] = p["raw"].astype(bool)
    return out


def near(anchor, radius_list):
    """Euclidean distance to the anchor, thresholded at each radius.

    One transform serves every radius, which matters because the transform runs
    over the whole volume and the sweep asks for several radii per anchor.
    """
    dist = ndimage.distance_transform_edt(~anchor)
    return {r: dist <= r for r in radius_list}


def opening_by_ball(mask, radius):
    """Morphological opening by a ball, via two distance transforms.

    A voxel survives if some ball of this radius fits inside the mask and covers
    it. Structures too narrow to hold the ball vannotator entirely, which is exactly
    the distinction wanted here: a distal airway is a thin tube, a leak into the
    parenchyma is a fat blob. scipy's binary_opening with an explicit ball is far
    slower at these radii, and this formulation is the same operation.
    """
    inside = ndimage.distance_transform_edt(mask)
    core = inside > radius
    if not core.any():
        return np.zeros_like(mask, dtype=bool)
    return (ndimage.distance_transform_edt(~core) <= radius) & mask


def build(m):
    """Every candidate rule, as {name: (mask, seed_for_component_selection)}."""
    a, md, tf = m["airmorph"], m["medpseg"], m["tfenet"]
    seven = [m[k] for k in MEMBERS] + [md, tf]
    votes = np.zeros(a.shape, dtype=np.uint8)
    for s in seven:
        votes += s.view(np.uint8)

    v = {}
    # reference points: the three members on their own, no selection applied
    v["solo_airmorph"] = (a, None)
    v["solo_medpseg"] = (md, None)
    v["solo_tfenet"] = (tf, None)

    # the published family
    v["or_airmorph_medpseg"] = (a | md, a)
    v["or_airmorph_tfenet"] = (a | tf, a)
    v["or_medpseg_tfenet"] = (md | tf, md)
    v["or_all3"] = (a | md | tf, a)
    v["vote2of3"] = ((a.view(np.uint8) + md.view(np.uint8) + tf.view(np.uint8)) >= 2, a)

    # graded consensus over all seven predictors
    for k in range(2, 7):
        v[f"vote{k}of7"] = (votes >= k, a)

    # agreement-anchored union: start from where several predictors agree, then
    # grow back along the union, but only within reach of that agreement. Distal
    # extensions seen by one model stay, isolated blobs do not.
    union3 = a | md | tf
    for anchor_name, anchor in (("v2of3", (a.view(np.uint8) + md.view(np.uint8)
                                           + tf.view(np.uint8)) >= 2),
                                ("v3of7", votes >= 3),
                                ("v4of7", votes >= 4)):
        masks = near(anchor, [1, 2, 3, 5])
        for r, within in masks.items():
            v[f"anchored_{anchor_name}_r{r}"] = (anchor | (union3 & within), a)

    # the same anchoring, but grown back along the two-of-seven map instead of
    # the union. The union admits anything one of three sees, which is where the
    # isolated false positives come from. Two of seven is already one agreement
    # step tighter, so anchoring on top of it trades less precision away.
    cand2of7 = votes >= 2
    for anchor_name, anchor in (("v3of7", votes >= 3), ("v4of7", votes >= 4)):
        for r, within in near(anchor, [1, 2, 3]).items():
            v[f"anchored_{anchor_name}_c2of7_r{r}"] = (anchor | (cand2of7 & within), a)

    # thickness gate. The union's false positives are not detached, they are
    # fused to the real tree, so no distance-to-agreement rule can reach them.
    # What separates them from a true distal branch is calibre. Keep everything
    # two of three models agree on, and of the rest keep only what is too narrow
    # to contain a ball of radius T.
    v2of3 = (a.view(np.uint8) + md.view(np.uint8) + tf.view(np.uint8)) >= 2
    for T in (1.0, 1.5, 2.0, 2.5, 3.0):
        thick = opening_by_ball(union3, T)
        v[f"thingate_v2of3_T{T}"] = (v2of3 | (union3 & ~thick), a)
    for T in (1.5, 2.5):
        thick = opening_by_ball(union3, T)
        v[f"thingate_v3of7_T{T}"] = ((votes >= 3) | (union3 & ~thick), a)

    # spatially smoothed agreement. Averaging the vote fraction over a small
    # neighbourhood costs a thin shell of disagreement at every surface, which
    # is where the leak lives, while the interior of a genuine tube keeps its
    # support from the neighbours along the axis.
    frac = ndimage.uniform_filter((votes / 7.0).astype(np.float32), size=3)
    for t in (0.15, 0.20, 0.25, 0.30, 0.40):
        v[f"smoothagree_t{t:.2f}"] = (frac >= t, a)

    # equal weight per model. Counting seven predictors lets AirMorph outvote
    # the other two on its own, since five of the seven are its members. Here
    # each model contributes one unit and AirMorph's internal agreement only
    # grades its own vote.
    amf = (m["member_model1"].view(np.uint8) + m["member_model2"].view(np.uint8)
           + m["member_model3"].view(np.uint8) + m["member_wingsnet"].view(np.uint8)
           + m["member_break"].view(np.uint8)).astype(np.float32) / 5.0
    wsum = amf + md.astype(np.float32) + tf.astype(np.float32)
    for t in (0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0):
        v[f"wvote_t{t}"] = (wsum >= t, a)

    # a morphological alternative to the anchoring idea
    st = ndimage.generate_binary_structure(3, 3)
    v["or_all3_opened"] = (ndimage.binary_closing(
        ndimage.binary_opening(union3, st, 1), st, 1), a)
    return v


def run_case(case):
    t0 = time.time()
    with open(os.path.join(CACHE, case, "meta.json")) as fh:
        meta = json.load(fh)
    label, _, _, _ = load_itk_image(os.path.join(GT_LABELS, case + ".mha"))
    label = (label > 0).astype(np.uint8)
    skel = restore_image_via_box(tuple(meta["original_shape"]),
                                 np.load(os.path.join(CACHE, case, "centerline.npy")
                                         ).astype(np.uint8),
                                 np.array(meta["bbox"]))
    parsing, lskel = parse_ground_truth(label, skel)

    masks = load_case_masks(case)
    for name, mk in masks.items():
        if mk.shape != label.shape:
            raise ValueError(f"{case} {name}: {mk.shape} != {label.shape}")

    rows = []
    for name, (mask, seed) in build(masks).items():
        mask = np.asarray(mask, dtype=np.uint8)
        pred = mask if seed is None else keep_seeded_component(mask, seed)
        rows.append({"case": f"ATM26/{case}", "model": name,
                     **score(pred, label, parsing, lskel)})
    print(f"{case} done  {len(rows)} rows  {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    args = ap.parse_args()
    use_split(args.split)

    cases = args.cases or sorted(d for d in os.listdir(CACHE)
                                 if os.path.isdir(os.path.join(CACHE, d)))
    ready = [c for c in cases
             if all(os.path.exists(os.path.join(ROOT, d, f"ATM26_{c}.npz"))
                    for d in OUT_DIRS.values())]
    if len(ready) != len(cases):
        print(f"WARNING {len(cases)-len(ready)} cases lack all three masks and are "
              f"skipped: {sorted(set(cases)-set(ready))}", flush=True)
    print(f"{len(ready)} cases, {args.workers} workers", flush=True)

    print(f"split={args.split}  masks under {ROOT}", flush=True)
    rows = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_case, ready):
                rows.extend(r)
    else:
        for c in ready:
            rows.extend(run_case(c))

    rows.sort(key=lambda r: (r["case"], r["model"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
