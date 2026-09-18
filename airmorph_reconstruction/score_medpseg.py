# -*- coding: utf-8 -*-
"""Stage 2 of the MEDPSeg baseline: score the saved masks against ATM26 ground truth.

Reads the `.npz` masks written by `training/predict_medpseg.py` and scores them with
`training.evaluate.score` - the *same* function, on the *same* uncropped ground truth,
that produced the numbers for the five AirMorph models. Rows are appended to the
existing per-case CSV so a single table covers all methods.

MEDPSeg's own copy of the ATM'22 metric code (`medpseg/atm_evaluation.py`) is
line-for-line the same official implementation as `util/airwayatlas_eval.py`, so the
comparison is like-for-like.

Two rows are emitted per case:
    medpseg       - the thresholded airway head, no post-processing (MEDPSeg default)
    medpseg_post  - largest connected component only (MEDPSeg's `--post` flag)

Usage (the AirMorph `clip` env):
    /path/to/envs/conda-envs/clip/bin/python -m training.score_medpseg \
        --split test --mask-dir outputs/medpseg/test_masks \
        --csv outputs/seg-model1/eval_test.csv
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

from training.eval_patch import parse_ground_truth          # noqa: E402
from training.evaluate import score, summarise, print_table  # noqa: E402
from util.utils import load_itk_image, restore_image_via_box  # noqa: E402

logger = logging.getLogger("airmorph.score_medpseg")

VARIANTS = {"raw": "medpseg", "post": "medpseg_post"}


def evaluate_masks(split: str, splits_path: str, cache_dir: str,
                   mask_dir: str) -> List[Dict]:
    """Score every saved MEDPSeg mask exactly as `training/evaluate.py` does."""

    with open(splits_path, "r") as handle:
        keys = json.load(handle)[split]

    rows: List[Dict] = []

    for index, key in enumerate(keys, start=1):
        mask_path = os.path.join(mask_dir, key.replace("/", "_") + ".npz")
        if not os.path.exists(mask_path):
            logger.warning("missing mask, skipping: %s", mask_path)
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

        with np.load(mask_path) as payload:
            for variant, model_name in VARIANTS.items():
                pred = payload[variant].astype(np.uint8)
                if pred.shape != label_full.shape:
                    raise ValueError(
                        f"{key} {variant}: mask {pred.shape} != label {label_full.shape}")
                rows.append({
                    "case": key, "model": model_name,
                    **score(pred, label_full, label_parsing, label_skeleton)})

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--splits-path",
                        default=os.path.join(_PROJECT_ROOT, "data/splits.json"))
    parser.add_argument("--cache-dir", default=os.path.join(_PROJECT_ROOT, "data/cache"))
    parser.add_argument("--mask-dir",
                        default=os.path.join(_PROJECT_ROOT, "outputs/medpseg/test_masks"))
    parser.add_argument("--csv",
                        default=os.path.join(_PROJECT_ROOT,
                                             "outputs/seg-model1/eval_test.csv"),
                        help="existing per-case CSV; MEDPSeg rows are appended to it")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    rows = evaluate_masks(args.split, args.splits_path, args.cache_dir, args.mask_dir)
    if not rows:
        logger.error("no masks scored")
        return 1

    with open(args.csv, "r", newline="") as handle:
        existing = list(csv.DictReader(handle))

    # Drop any earlier MEDPSeg rows so re-running does not duplicate them.
    merged = [row for row in existing if row["model"] not in VARIANTS.values()] + rows

    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(merged)
    logger.info("per-case results written to %s", args.csv)

    # Re-read so the summary is over floats parsed the same way for every method.
    typed = [{**row, **{k: float(v) for k, v in row.items()
                        if k not in ("case", "model")}} for row in merged]
    summary = summarise(typed)
    print_table(summary)

    summary_path = args.csv.replace(".csv", "_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("summary written to %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
