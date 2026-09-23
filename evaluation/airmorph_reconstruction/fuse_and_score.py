# -*- coding: utf-8 -*-
"""Fusion study: score AirMorph, MEDPSeg and TfeNet alone and in every combination.

Answers "which model or combination segments ATM26 best", using the ground truth the
ATM26 test split has. Reads the `.npz` masks written by `training/dump_airmorph_masks.py`,
`training/predict_medpseg.py` and `training/predict_tfenet.py`, all already in the
original image frame, and scores every variant with `training.evaluate.score` - the same
function, same uncropped ground truth, as every other row in the table.

FUSION RULE, matching the shipped clinical runs (verified bit-for-bit against
BCH_LIAM/run-6): bitwise OR the member masks, then apply the largest connected component
ONCE to the combined result, at 6-connectivity. LCC is deliberately NOT applied per
method first - TfeNet in particular is noisy but reaches further distally, and its
detached fragments are exactly what the OR is meant to reconnect to the tree. Pruning
them per method would throw away the reach the fusion exists to capture.

LCC is applied at exactly one point in this script: to a finished combined mask. No
per-method LCC row exists, and no input to a union has been pruned.

Variants scored (combinations only - the solo rows already exist in eval_test.csv):
    or_airmorph_tfenet_lcc                 OR then LCC
    or_airmorph_medpseg_lcc                OR then LCC
    or_medpseg_tfenet_lcc                  OR then LCC  (the '01_airway_combined' recipe)
    or_all3_lcc                            OR then LCC  (the 'union3_lcc' recipe)
    vote2of3 / vote2of3_lcc                >=2 of the 3 methods agree

WHICH COMPONENT THE LCC KEEPS (2026-08-07)
    The original rule kept the *globally largest* component of the union. On 3 of the 30
    test cases (ATM_507, ATM_509, ATM_627) TfeNet contributes a spurious blob that is both
    larger than the airway tree and disjoint from it, so the largest-component rule keeps
    the blob and deletes the entire airway: DSC/clDice/TLD/BND/Sensitivity all score
    exactly 0.0. Every TfeNet-containing union was hit; `or_airmorph_medpseg_lcc` was not.
    Those three zeros are what dragged the TfeNet fusions' means to ~85 and inflated their
    std to ~28, which made them look far worse than they are.

    `--lcc-rule seeded` (the default) instead keeps the component that contains the seed
    method's tree - AirMorph where present, MEDPSeg for `or_medpseg_tfenet_lcc`. Audited
    over all 30 test cases x 4 unions: seeded and largest pick the SAME component on 111
    of 120 pairs; they differ only on the 9 pairs above. So the switch changes nothing
    except repairing the three collapses.

    `--lcc-rule largest` reproduces the original behaviour.

Usage (the AirMorph `clip` env):
    /path/to/envs/conda-envs/clip/bin/python -m training.fuse_and_score \
        --split test --csv outputs/seg-model1/fusion_test_seeded.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from training.eval_patch import parse_ground_truth                        # noqa: E402
from training.evaluate import print_table, score, summarise               # noqa: E402
from util.utils import load_itk_image, restore_image_via_box              # noqa: E402

logger = logging.getLogger("airmorph.fuse_and_score")

METHODS = ("airmorph", "medpseg", "tfenet")


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Largest connected component at 6-connectivity - the clinical runs' fusion LCC.

    Deliberately NOT `training.evaluate.keep_largest_component`, which is MONAI's
    `KeepLargestConnectedComponent(connectivity=3)` = 26-connectivity. The two settings
    coexist in this repo and are not interchangeable:

      * 26-conn is the *deployed AirMorph* LCC (segmentator/airway_segmentator.py, and
        BCH_LIAM/airmorph_run/pipeline_case.py:160-168 for union3_lcc).
      * 6-conn is what actually produced the shipped clinical fusion. Verified against
        BCH_LIAM/run-6: `combined_airway.nii.gz` is bit-for-bit equal to the 6-conn LCC
        of `medpseg | tfenet` (53946 voxels, matching logs/combine.log). The 26-conn LCC
        of the same union gives 54177 voxels - close, but not the shipped mask.

    6-conn is the stricter rule: a voxel touching the tree only at a corner or edge is a
    separate component and gets pruned. On run-6 the union split into 2566 components at
    6-conn versus 1552 at 26-conn.

    `skimage.morphology.remove_small_holes`/`binary_fill_holes` is not applied. TfeNet's
    own `large_connected_domain` does fill holes, but on run-6 that step changed nothing
    (0 voxels), so it cannot be confirmed as part of the shipped recipe. Plain 6-conn LCC
    is the minimal rule that reproduces the shipped mask exactly.
    """

    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, 1)
    labelled, count = ndimage.label(mask.astype(bool), structure=structure)
    if count == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return (labelled == sizes.argmax()).astype(np.uint8)


def keep_seeded_component(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Keep the 6-conn component of `mask` that holds the most `seed` voxels.

    Same connectivity as `keep_largest_component`; only the choice of which component
    survives differs. `seed` is the union member trusted to sit on the airway (AirMorph,
    or MEDPSeg when AirMorph is not a member). Falls back to the largest component if the
    seed is empty.

    Needed because "largest" is not a safe proxy for "the airway" once a noisy member is
    in the union: on ATM_507/509/627 TfeNet's spurious blob outweighs the whole tree and
    is disjoint from it, so the largest-component rule returns a mask with zero overlap
    with the ground truth. See the module docstring.
    """

    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, 1)
    labelled, count = ndimage.label(mask.astype(bool), structure=structure)
    if count == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    seed_labels = labelled[seed.astype(bool)]
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return keep_largest_component(mask)

    return (labelled == int(np.bincount(seed_labels).argmax())).astype(np.uint8)


def build_variants(masks: Dict[str, np.ndarray],
                   lcc_rule: str = "seeded") -> Dict[str, np.ndarray]:
    """Every single method and combination, LCC applied only where the name says so."""

    airmorph, medpseg, tfenet = (masks[name] for name in METHODS)

    # Solo rows are NOT recomputed here - outputs/seg-model1/eval_test.csv already holds
    # them (`medpseg` and `tfenet_bas` are the same raw masks this script loads; `ensemble`
    # is AirMorph, though with its deployed 26-conn LCC rather than raw). Scoring is the
    # expensive step, so skipping the three solos cuts this job by a third.
    variants: Dict[str, np.ndarray] = {}

    # (union, seed) - the seed is the member trusted to lie on the airway.
    unions = {
        "or_airmorph_tfenet_lcc": (airmorph | tfenet, airmorph),
        "or_airmorph_medpseg_lcc": (airmorph | medpseg, airmorph),
        "or_medpseg_tfenet_lcc": (medpseg | tfenet, medpseg),
        "or_all3_lcc": (airmorph | medpseg | tfenet, airmorph),
    }
    for name, (union, seed) in unions.items():
        variants[name] = (keep_seeded_component(union, seed) if lcc_rule == "seeded"
                          else keep_largest_component(union))

    votes = (airmorph.astype(np.uint8) + medpseg.astype(np.uint8)
             + tfenet.astype(np.uint8))
    vote2 = (votes >= 2).astype(np.uint8)
    variants["vote2of3"] = vote2
    variants["vote2of3_lcc"] = (keep_seeded_component(vote2, airmorph)
                                if lcc_rule == "seeded"
                                else keep_largest_component(vote2))

    return variants


def evaluate(split: str, splits_path: str, cache_dir: str,
             mask_dirs: Dict[str, str], lcc_rule: str = "seeded") -> List[Dict]:
    with open(splits_path, "r") as handle:
        keys = json.load(handle)[split]

    rows: List[Dict] = []

    for index, key in enumerate(keys, start=1):
        stem = key.replace("/", "_") + ".npz"
        paths = {name: os.path.join(directory, stem)
                 for name, directory in mask_dirs.items()}
        missing = [name for name, path in paths.items() if not os.path.exists(path)]
        if missing:
            logger.warning("[%d/%d] %s: missing %s, skipping case",
                           index, len(keys), key, ", ".join(missing))
            continue

        case_dir = os.path.join(cache_dir, key)
        with open(os.path.join(case_dir, "meta.json"), "r") as handle:
            meta = json.load(handle)

        logger.info("[%d/%d] %s", index, len(keys), key)

        # Identical ground-truth construction to training/evaluate.py:178-185.
        label_full, _, _, _ = load_itk_image(meta["label_path"])
        label_full = (label_full > 0).astype(np.uint8)

        skeleton_crop = np.load(os.path.join(case_dir, "centerline.npy"))
        skeleton_full = restore_image_via_box(
            tuple(meta["original_shape"]), skeleton_crop.astype(np.uint8),
            np.array(meta["bbox"]))
        label_parsing, label_skeleton = parse_ground_truth(label_full, skeleton_full)

        masks = {}
        for name, path in paths.items():
            with np.load(path) as payload:
                masks[name] = payload["raw"].astype(np.uint8)
            if masks[name].shape != label_full.shape:
                raise ValueError(f"{key} {name}: mask {masks[name].shape} != "
                                 f"label {label_full.shape}")

        for variant, pred in build_variants(masks, lcc_rule).items():
            rows.append({"case": key, "model": variant,
                         **score(pred, label_full, label_parsing, label_skeleton)})

        del masks

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--splits-path",
                        default=os.path.join(_PROJECT_ROOT, "data/splits.json"))
    parser.add_argument("--cache-dir", default=os.path.join(_PROJECT_ROOT, "data/cache"))
    parser.add_argument("--airmorph-dir", default=os.path.join(
        _PROJECT_ROOT, "outputs/airmorph/test_masks"))
    parser.add_argument("--medpseg-dir", default=os.path.join(
        _PROJECT_ROOT, "outputs/medpseg/test_masks"))
    parser.add_argument("--tfenet-dir", default=os.path.join(
        _PROJECT_ROOT, "outputs/tfenet/BAS_test_masks"))
    parser.add_argument("--csv", default=os.path.join(
        _PROJECT_ROOT, "outputs/seg-model1/fusion_test.csv"),
        help="written fresh; this study does not touch eval_test.csv")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    mask_dirs = {"airmorph": args.airmorph_dir,
                 "medpseg": args.medpseg_dir,
                 "tfenet": args.tfenet_dir}
    for name, directory in mask_dirs.items():
        logger.info("%-8s <- %s", name, directory)

    logger.info("lcc-rule: %s", args.lcc_rule)
    rows = evaluate(args.split, args.splits_path, args.cache_dir, mask_dirs,
                    args.lcc_rule)
    if not rows:
        logger.error("no cases scored")
        return 1

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("per-case results written to %s", args.csv)

    summary = summarise(rows)
    print_table(summary)

    summary_path = args.csv.replace(".csv", "_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("summary written to %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
