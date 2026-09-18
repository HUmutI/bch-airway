# -*- coding: utf-8 -*-
"""Leak-safe split handling shared by the labeling prepare / train / eval scripts.

The segmentation stage partitions the 299 ATM26 case ids once (seeded 80/10/10). The
labeling stage (118 annotated cases, a subset of those 299) **inherits** that partition:
a labeling case lands in whichever fold its case id already belongs to. This guarantees no
case is segmentation-train but labeling-test, which matters for the decoupled "both" run.
"""

import json
import os
from typing import Dict, List


def _case_id(key: str) -> str:
    """"ATM26/ATM_001" -> "ATM_001"."""
    return key.split("/", 1)[-1]


def load_or_build_seg_splits(config: Dict) -> Dict[str, List[str]]:
    """Return the segmentation split manifest, building it from the case list if absent.

    Building here (rather than requiring the heavy seg preprocessing to have run) lets the
    labeling-only training path stand alone while still using the identical seeded split.
    """

    path = config["split_path"]
    if os.path.exists(path):
        with open(path, "r") as handle:
            return json.load(handle)

    # Build from the discovered seg case ids using the same seeded logic as prepare_dataset.
    from training.data.prepare_dataset import build_splits, discover_cases

    cases = discover_cases(config["seg_images"], config["seg_labels"],
                           config["dataset_name"])
    splits = build_splits(cases, config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(splits, handle, indent=2)
    print(f"[labeling_splits] built seg splits at {path}")
    return splits


def derive_labeling_splits(
    seg_splits: Dict[str, List[str]],
    label_case_ids: List[str],
    dataset_name: str,
) -> Dict[str, List[str]]:
    """Map each labeling case id into the fold its seg case id already occupies."""

    fold_of = {}
    for fold in ("train", "val", "test"):
        for key in seg_splits.get(fold, []):
            fold_of[_case_id(key)] = fold

    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    orphans = []
    for cid in sorted(label_case_ids):
        fold = fold_of.get(cid)
        if fold is None:
            orphans.append(cid)
            fold = "train"          # unseen by the seg split: default to train, warn
        out[fold].append(f"{dataset_name}/{cid}")

    if orphans:
        print(f"[labeling_splits] {len(orphans)} labeling cases not in the seg split "
              f"(added to train): {orphans[:5]}{' ...' if len(orphans) > 5 else ''}")
    return out


def write_labeling_splits(config: Dict, label_case_ids: List[str]) -> Dict[str, List[str]]:
    """Derive and persist the labeling split manifest; return it."""

    seg_splits = load_or_build_seg_splits(config)
    label_splits = derive_labeling_splits(
        seg_splits, label_case_ids, config["dataset_name"])
    path = config["label_split_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(label_splits, handle, indent=2)
    print(f"[labeling_splits] wrote {path}: "
          f"train={len(label_splits['train'])} val={len(label_splits['val'])} "
          f"test={len(label_splits['test'])}")
    return label_splits
