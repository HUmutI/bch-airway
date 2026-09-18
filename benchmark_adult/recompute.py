#!/usr/bin/env python
"""Recompute the ATM26 test-split benchmark from stored masks and ground truth.

Reproduces every row of the fusion study without depending on the driver that
emitted the shipped CSVs (that version is lost; the one on disk never registers
--lcc-rule and would raise). Metric functions and the fusion rules are the
originals, imported unmodified.

Per case it scores:
  * 3 solo members, raw (pre-LCC) masks as the fusion consumes them
  * 6 fusion variants under the seeded rule
  * 6 fusion variants under the largest-component rule
"""
import argparse, csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "shim"))

BUNDLE = "/path/to/airway-data/AIRWAY_METHOD_PAPER/airway_paper_bundle"
MASKS = os.path.join(BUNDLE, "08_masks_and_gt")
GT_LABELS = os.path.join(MASKS, "gt_labelsTr")
CACHE = os.path.join(MASKS, "gt_cache_centerlines")

from scipy import ndimage
from training.eval_patch import cldice_fixed, parse_ground_truth
from util.airwayatlas_eval import (branch_detected_calculation,
                                   dice_coefficient_score_calculation,
                                   false_positive_rate_calculation,
                                   precision_calculation,
                                   sensitivity_calculation,
                                   tree_length_calculation)
from util.utils import load_itk_image, restore_image_via_box

MASK_DIRS = {"airmorph": "airmorph_test_masks",
             "medpseg": "medpseg_test_masks",
             "tfenet": "tfenet_BAS_test_masks"}


def score(pred, label, label_parsing, label_skeleton):
    """Verbatim from training/evaluate.py:119-141."""
    pred = pred.astype(np.uint8); label = label.astype(np.uint8)
    _, _, bnd = branch_detected_calculation(pred, label_parsing, label_skeleton,
                                            thresh=0.8)
    return {"DSC": float(dice_coefficient_score_calculation(pred, label)),
            "clDice": float(cldice_fixed(v_p_pred=pred, v_l_gt=label)),
            "TLD": float(tree_length_calculation(pred, label_skeleton)),
            "BND": float(bnd),
            "Sensitivity": float(sensitivity_calculation(pred, label)),
            "Precision": float(precision_calculation(pred, label)),
            "FPR": float(false_positive_rate_calculation(pred, label))}


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


def build_variants(masks, lcc_rule):
    airmorph, medpseg, tfenet = (masks[n] for n in ("airmorph", "medpseg", "tfenet"))
    unions = {"or_airmorph_tfenet_lcc": (airmorph | tfenet, airmorph),
              "or_airmorph_medpseg_lcc": (airmorph | medpseg, airmorph),
              "or_medpseg_tfenet_lcc": (medpseg | tfenet, medpseg),
              "or_all3_lcc": (airmorph | medpseg | tfenet, airmorph)}
    out = {}
    for name, (union, seed) in unions.items():
        out[name] = (keep_seeded_component(union, seed) if lcc_rule == "seeded"
                     else keep_largest_component(union))
    vote2 = ((airmorph.astype(np.uint8) + medpseg.astype(np.uint8)
              + tfenet.astype(np.uint8)) >= 2).astype(np.uint8)
    out["vote2of3"] = vote2
    out["vote2of3_lcc"] = (keep_seeded_component(vote2, airmorph) if lcc_rule == "seeded"
                           else keep_largest_component(vote2))
    return out


def run_case(case):
    t0 = time.time()
    with open(os.path.join(CACHE, case, "meta.json")) as fh:
        meta = json.load(fh)

    label_full, _, _, _ = load_itk_image(os.path.join(GT_LABELS, case + ".mha"))
    label_full = (label_full > 0).astype(np.uint8)

    skeleton_crop = np.load(os.path.join(CACHE, case, "centerline.npy"))
    skeleton_full = restore_image_via_box(tuple(meta["original_shape"]),
                                          skeleton_crop.astype(np.uint8),
                                          np.array(meta["bbox"]))
    label_parsing, label_skeleton = parse_ground_truth(label_full, skeleton_full)

    masks = {}
    for name, sub in MASK_DIRS.items():
        with np.load(os.path.join(MASKS, sub, f"ATM26_{case}.npz")) as payload:
            masks[name] = payload["raw"].astype(np.uint8)
        if masks[name].shape != label_full.shape:
            raise ValueError(f"{case} {name}: {masks[name].shape} != {label_full.shape}")

    rows = []
    for name, mask in masks.items():
        rows.append({"case": f"ATM26/{case}", "model": f"{name}_raw", "lcc_rule": "none",
                     **score(mask, label_full, label_parsing, label_skeleton)})
    for rule in ("seeded", "largest"):
        for variant, pred in build_variants(masks, rule).items():
            rows.append({"case": f"ATM26/{case}", "model": variant, "lcc_rule": rule,
                         **score(pred, label_full, label_parsing, label_skeleton)})
    print(f"{case} done  {len(rows)} rows  {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cases = args.cases or sorted(
        d for d in os.listdir(CACHE) if os.path.isdir(os.path.join(CACHE, d)))
    print(f"{len(cases)} cases, {args.workers} workers", flush=True)

    rows = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_case, cases):
                rows.extend(r)
    else:
        for c in cases:
            rows.extend(run_case(c))

    rows.sort(key=lambda r: (r["case"], r["lcc_rule"], r["model"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
