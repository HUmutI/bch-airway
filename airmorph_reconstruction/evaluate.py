# -*- coding: utf-8 -*-
"""Evaluate the trained models, individually and as the 5-model ensemble.

Reproduces the deployed inference path exactly (`segmentator/airway_segmentator.py`):

    sliding_window_inference(roi=(128,224,304), sw_batch_size=1, overlap=0.75,
                             mode='gaussian', sigma_scale=0.25)
    -> threshold each model at 0.5
    -> sum the 5 binary maps, keep voxels with >= 3 votes
    -> KeepLargestConnectedComponent
    -> restore to the full frame via the lung bounding box

Predictions are compared against the **uncropped** ground truth, so the pipeline is
correctly charged for any airway (typically upper trachea) that the lung bounding box
cut off - the same penalty the deployed system incurs.

Metrics come from `util/airwayatlas_eval.py`, except clDice which is taken from
`training/eval_patch.py` because the shipped one raises on 3D input.

IMPORTANT on comparability: the paper's Table 1 reports TLD/BND/DSC/clDice/Sensitivity
separately under "Lobar", "Segmental" and "Subsegmental" headings. Those are binary
metrics restricted to branches at each anatomical level, which requires the manual
anatomical labels that are out of scope here. The numbers this script produces are
**whole-tree** and are systematically higher than the paper's level-stratified columns -
they are not directly comparable. See training/README.md.

Usage:
    python -m training.evaluate --split test
    python -m training.evaluate --split test --models model1 wingsnet
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from training.configs.train_config_models import MODEL_NAMES, get_config   # noqa: E402
from training.eval_patch import cldice_fixed, parse_ground_truth           # noqa: E402
from training.nets import ProbabilityPredictor, build_network             # noqa: E402
from util.airwayatlas_eval import (                                       # noqa: E402
    branch_detected_calculation,
    dice_coefficient_score_calculation,
    false_positive_rate_calculation,
    precision_calculation,
    sensitivity_calculation,
    tree_length_calculation,
)
from util.utils import load_itk_image, restore_image_via_box              # noqa: E402

logger = logging.getLogger("airmorph.evaluate")

#: Votes required for a voxel to be airway (segmentator/airway_segmentator.py:67-68).
ENSEMBLE_VOTE_THRESHOLD = 3


def load_model(model_name: str, device: str, checkpoint_dir: Optional[str] = None):
    """Rebuild a network and load its exported checkpoint."""

    config = get_config(model_name)
    directory = checkpoint_dir or config["checkpoint_dir"]
    path = os.path.join(directory, config["checkpoint_name"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")

    net = build_network(config, device)
    payload = torch.load(path, map_location=device)
    state_dict = payload["state_dict"] if config["checkpoint_format"] == "state_dict" \
        else payload
    net.load_state_dict(state_dict)
    net.eval()

    return net, config


@torch.no_grad()
def predict_volume(net, config: Dict, image: np.ndarray, device: str) -> np.ndarray:
    """Sliding-window prediction over one cached (lung-cropped) volume."""

    from monai.inferers import sliding_window_inference

    predictor = ProbabilityPredictor(net, config)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None].to(device)

    prob = sliding_window_inference(
        inputs=tensor,
        roi_size=config["patch_size"],
        sw_batch_size=config["sw_batch_size"],
        predictor=predictor,
        overlap=config["overlap"],
        mode=config["sw_mode"],
        sigma_scale=config["sigma_scale"])

    result = prob.float()[0, 0].cpu().numpy()
    del tensor, prob
    torch.cuda.empty_cache()
    return result


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """The `KeepLargestConnectedComponent(connectivity=3)` step of the deployed path."""

    from monai.transforms import KeepLargestConnectedComponent

    transform = KeepLargestConnectedComponent(applied_labels=1, connectivity=3)
    tensor = torch.from_numpy(mask.astype(np.uint8))[None]
    return np.asarray(transform(tensor)[0], dtype=np.uint8)


def score(
    pred: np.ndarray,
    label: np.ndarray,
    label_parsing: np.ndarray,
    label_skeleton: np.ndarray,
) -> Dict[str, float]:
    """Compute the paper's binary metrics for one case."""

    pred = pred.astype(np.uint8)
    label = label.astype(np.uint8)

    _, _, bnd = branch_detected_calculation(
        pred, label_parsing, label_skeleton, thresh=0.8)

    return {
        "DSC": float(dice_coefficient_score_calculation(pred, label)),
        "clDice": float(cldice_fixed(v_p_pred=pred, v_l_gt=label)),
        "TLD": float(tree_length_calculation(pred, label_skeleton)),
        "BND": float(bnd),
        "Sensitivity": float(sensitivity_calculation(pred, label)),
        "Precision": float(precision_calculation(pred, label)),
        "FPR": float(false_positive_rate_calculation(pred, label)),
    }


def evaluate(
    split: str,
    model_names: List[str],
    device: str,
    checkpoint_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Evaluate each model and the ensemble over one split."""

    base = get_config(model_names[0])

    with open(base["split_path"], "r") as handle:
        keys = json.load(handle)[split]
    if limit:
        keys = keys[:limit]

    models = {}
    for name in model_names:
        models[name] = load_model(name, device, checkpoint_dir)
        logger.info("loaded %s", name)

    rows: List[Dict] = []

    for index, key in enumerate(keys, start=1):
        case_dir = os.path.join(base["cache_dir"], key)
        with open(os.path.join(case_dir, "meta.json"), "r") as handle:
            meta = json.load(handle)

        logger.info("[%d/%d] %s", index, len(keys), key)

        image = np.load(os.path.join(case_dir, "image.npy"))
        bbox = np.array(meta["bbox"])
        original_shape = tuple(meta["original_shape"])

        # Ground truth in the ORIGINAL frame, so the bbox crop is charged for.
        label_full, _, _, _ = load_itk_image(meta["label_path"])
        label_full = (label_full > 0).astype(np.uint8)

        skeleton_crop = np.load(os.path.join(case_dir, "centerline.npy"))
        skeleton_full = restore_image_via_box(
            original_shape, skeleton_crop.astype(np.uint8), bbox)
        label_parsing, label_skeleton = parse_ground_truth(label_full, skeleton_full)

        votes = np.zeros(image.shape, dtype=np.uint8)
        for name, (net, config) in models.items():
            prob = predict_volume(net, config, image, device)
            binary = (prob > 0.5).astype(np.uint8)
            votes += binary

            pred_full = restore_image_via_box(original_shape, binary, bbox)
            rows.append({
                "case": key, "model": name,
                **score(pred_full, label_full, label_parsing, label_skeleton)})

        if len(models) == len(MODEL_NAMES):
            ensemble = (votes >= ENSEMBLE_VOTE_THRESHOLD).astype(np.uint8)
            ensemble = keep_largest_component(ensemble)
            ensemble_full = restore_image_via_box(original_shape, ensemble, bbox)
            rows.append({
                "case": key, "model": "ensemble",
                **score(ensemble_full, label_full, label_parsing, label_skeleton)})

    return rows


def summarise(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Mean +/- sd per model."""

    metrics = ["TLD", "BND", "DSC", "clDice", "Sensitivity", "Precision", "FPR"]
    summary: Dict[str, Dict[str, float]] = {}

    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        entry = {}
        for metric in metrics:
            values = np.array([row[metric] for row in subset], dtype=np.float64)
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std())
        entry["n"] = len(subset)
        summary[model] = entry

    return summary


def print_table(summary: Dict[str, Dict[str, float]]) -> None:
    metrics = ["TLD", "BND", "DSC", "clDice", "Sensitivity"]
    header = f"{'model':<12}" + "".join(f"{m:>18}" for m in metrics) + f"{'n':>6}"
    print("\n" + header)
    print("-" * len(header))
    for model, entry in summary.items():
        line = f"{model:<12}"
        for metric in metrics:
            line += f"{entry[f'{metric}_mean']:>11.2f} ± {entry[f'{metric}_std']:<4.2f}"
        line += f"{entry['n']:>6}"
        print(line)
    print("\nWhole-tree metrics. NOT directly comparable to the paper's Table 1, whose "
          "columns are stratified by anatomical level (see module docstring).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES),
                        choices=list(MODEL_NAMES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="path for the per-case CSV")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    rows = evaluate(args.split, args.models, args.device,
                    args.checkpoint_dir, args.limit)
    if not rows:
        logger.error("no cases evaluated")
        return 1

    summary = summarise(rows)
    print_table(summary)

    out_path = args.out or os.path.join(
        get_config(args.models[0])["work_dir"], f"eval_{args.split}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("per-case results written to %s", out_path)

    with open(out_path.replace(".csv", "_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
