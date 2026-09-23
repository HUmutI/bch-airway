# -*- coding: utf-8 -*-
"""Evaluate the trained anatomical-labeling transformer on a split.

Reports the paper's graph-node-level labeling metrics per anatomical level
(lobar / segmental / subsegmental):
  - accuracy (fraction of labelled nodes classified correctly),
  - macro precision / sensitivity (mean over classes present in the ground truth),
  - TreeCons (predicted-subtree consistency, %), via the provided
    `models/evaluation_matrics.py::calculate_CS`.

Uses the STAGE-2 heads, exactly as `models/labeling_model.py::predict` does at inference.

IMPORTANT on comparability: the paper's Table 1 labeling columns are computed on its large
multi-cohort data with the authors' full annotations. Numbers here are on the ATM26 118-case
split only and are not directly comparable; see training/README.md.

Usage:
    python -m training.evaluate_labeling --split test
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from training.configs.train_config_labeling import get_config          # noqa: E402
from training.data.labeling_dataset import LabelingGraphDataset        # noqa: E402
from training.train_labeling import build_model                        # noqa: E402
from models.evaluation_matrics import calculate_CS                     # noqa: E402

logger = logging.getLogger("airmorph.evaluate_labeling")

_LEVELS = (("lob", "y_lob", "num_classes1"),
           ("seg", "y_seg", "num_classes2"),
           ("sub", "y_sub", "num_classes3"))


def load_model(config: Dict, device: str, checkpoint_dir: str = None):
    directory = checkpoint_dir or config["checkpoint_dir"]
    path = os.path.join(directory, config["checkpoint_name"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    net = build_model(config, device)
    net.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    net.eval()
    return net


def _macro_pr_se(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Macro precision / sensitivity over classes present in `target`."""
    precisions, sensitivities = [], []
    for c in np.unique(target):
        tp = np.sum((pred == c) & (target == c))
        fp = np.sum((pred == c) & (target != c))
        fn = np.sum((pred != c) & (target == c))
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
        sensitivities.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return float(np.mean(precisions)) if precisions else 0.0, \
        float(np.mean(sensitivities)) if sensitivities else 0.0


@torch.no_grad()
def evaluate(split: str, device: str, checkpoint_dir: str = None,
             limit: int = None) -> List[Dict]:
    config = get_config()
    net = load_model(config, device, checkpoint_dir)
    dataset = LabelingGraphDataset(config, split=split)
    ignore = config["ignore_index"]

    rows: List[Dict] = []
    for index in range(len(dataset) if limit is None else min(limit, len(dataset))):
        item = dataset[index]
        x = item["x"].to(device)
        spd = item["spd"].to(device)
        mask_top_t = item["mask_top"].to(device)
        outputs = net(x, mask_top_t, spd, 0.0)
        _, _, _, x1_2, x2_2, x3_2 = outputs[:6]
        preds = {"lob": x1_2.argmax(1).cpu().numpy(),
                 "seg": x2_2.argmax(1).cpu().numpy(),
                 "sub": x3_2.argmax(1).cpu().numpy()}

        row = {"case": item["key"], "num_nodes": int(x.shape[0])}
        for name, ykey, nkey in _LEVELS:
            y = item[ykey].numpy()
            m = y != ignore
            pred = preds[name]
            acc = float(np.mean(pred[m] == y[m])) if m.any() else float("nan")
            prec, sens = _macro_pr_se(pred[m], y[m], config[nkey])
            row[f"acc_{name}"] = acc
            row[f"precision_{name}"] = prec
            row[f"sensitivity_{name}"] = sens

        mask_top = item["mask_top"].numpy()
        try:
            cs_seg, cs_sub = calculate_CS(
                preds["seg"], preds["sub"], mask_top, config["label_old2new_path"])
            row["treecons_seg"] = float(cs_seg)
            row["treecons_sub"] = float(cs_sub)
        except Exception:
            row["treecons_seg"] = float("nan")
            row["treecons_sub"] = float("nan")

        rows.append(row)
        logger.info("[%d/%d] %s acc lob/seg/sub %.3f/%.3f/%.3f",
                    index + 1, len(dataset), item["key"],
                    row["acc_lob"], row["acc_seg"], row["acc_sub"])
    return rows


def summarise(rows: List[Dict]) -> Dict[str, float]:
    metrics = [k for k in rows[0] if k not in ("case", "num_nodes")]
    summary = {}
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        summary[f"{m}_mean"] = float(vals.mean()) if vals.size else float("nan")
        summary[f"{m}_std"] = float(vals.std()) if vals.size else float("nan")
    summary["n"] = len(rows)
    return summary


def print_table(summary: Dict[str, float]) -> None:
    print("\nAnatomical labeling (graph-node level), mean over cases:")
    print(f"{'level':<6}{'accuracy':>12}{'precision':>12}{'sensitivity':>13}{'TreeCons':>11}")
    print("-" * 54)
    for name, label in (("lob", "lobar"), ("seg", "segmental"), ("sub", "subseg")):
        tc_key = f"treecons_{name}_mean" if name != "lob" else None
        tc = summary.get(tc_key, float("nan")) if tc_key else float("nan")
        print(f"{label:<6}"
              f"{summary[f'acc_{name}_mean']*100:>11.2f}"
              f"{summary[f'precision_{name}_mean']*100:>12.2f}"
              f"{summary[f'sensitivity_{name}_mean']*100:>13.2f}"
              f"{(tc if not np.isnan(tc) else float('nan')):>11.2f}")
    print(f"\nn = {summary['n']} cases. ATM26 split only; NOT directly comparable to the "
          f"paper's multi-cohort Table 1 (see module docstring).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    config = get_config()
    device = args.device or config["device"]
    rows = evaluate(args.split, device, args.checkpoint_dir, args.limit)
    if not rows:
        logger.error("no cases evaluated")
        return 1

    summary = summarise(rows)
    print_table(summary)

    out_path = args.out or os.path.join(config["work_dir"], f"eval_labeling_{args.split}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_path.replace(".csv", "_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
