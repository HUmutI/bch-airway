# -*- coding: utf-8 -*-
"""Save AirMorph's ensemble mask per case so it can be fused with other methods.

`training/evaluate.py` predicts and scores in one pass and never writes the masks out,
so there is nothing on disk to OR against MEDPSeg and TfeNet. This script runs the same
inference path and persists the result in the layout `training/predict_medpseg.py` and
`training/predict_tfenet.py` already use, which `training/fuse_and_score.py` reads.

The saved ensemble is **pre-LCC** on purpose. In the fusion study the largest-connected-
component step runs once, on the final combined mask - applying it per method first would
delete detached distal branches before the other methods get a chance to reconnect them.
`training/evaluate.py:200` applies LCC because it scores AirMorph as the deployed
standalone product; that is a different question from this one.

Arrays written per case (all uint8, all in the ORIGINAL image frame):
    raw            3-of-5 vote, no LCC          <- what the fusion consumes
    post           3-of-5 vote + LCC            <- the deployed AirMorph output
    member_<name>  each of the 5 models at 0.5  <- kept so member-level fusion needs no re-run

Usage (the AirMorph `clip` env):
    /path/to/envs/conda-envs/clip/bin/python -m training.dump_airmorph_masks \
        --split test --device cuda:1 --out-dir outputs/airmorph/test_masks
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from training.configs.train_config_models import MODEL_NAMES, get_config   # noqa: E402
from training.evaluate import (                                            # noqa: E402
    ENSEMBLE_VOTE_THRESHOLD,
    keep_largest_component,
    load_model,
    predict_volume,
)
from util.utils import restore_image_via_box                               # noqa: E402

logger = logging.getLogger("airmorph.dump_airmorph_masks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--out-dir", default=os.path.join(
        _PROJECT_ROOT, "outputs/airmorph/test_masks"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    base = get_config(MODEL_NAMES[0])
    with open(base["split_path"], "r") as handle:
        keys = json.load(handle)[args.split]
    if args.limit:
        keys = keys[:args.limit]

    models = {}
    for name in MODEL_NAMES:
        models[name] = load_model(name, args.device, args.checkpoint_dir)
        logger.info("loaded %s", name)

    for index, key in enumerate(keys, start=1):
        out_path = os.path.join(out_dir, key.replace("/", "_") + ".npz")
        if os.path.exists(out_path):
            logger.info("[%d/%d] %s (cached)", index, len(keys), key)
            continue

        case_dir = os.path.join(base["cache_dir"], key)
        with open(os.path.join(case_dir, "meta.json"), "r") as handle:
            meta = json.load(handle)

        logger.info("[%d/%d] %s", index, len(keys), key)

        image = np.load(os.path.join(case_dir, "image.npy"))
        bbox = np.array(meta["bbox"])
        original_shape = tuple(meta["original_shape"])

        # Identical to training/evaluate.py:187-200, minus the scoring.
        payload = {}
        votes = np.zeros(image.shape, dtype=np.uint8)
        for name, (net, config) in models.items():
            binary = (predict_volume(net, config, image, args.device) > 0.5).astype(np.uint8)
            votes += binary
            payload[f"member_{name}"] = restore_image_via_box(
                original_shape, binary, bbox).astype(np.uint8)

        ensemble = (votes >= ENSEMBLE_VOTE_THRESHOLD).astype(np.uint8)
        payload["raw"] = restore_image_via_box(
            original_shape, ensemble, bbox).astype(np.uint8)
        payload["post"] = restore_image_via_box(
            original_shape, keep_largest_component(ensemble), bbox).astype(np.uint8)
        payload["original_shape"] = np.array(original_shape)

        np.savez_compressed(out_path, **payload)
        del payload, votes, image

    logger.info("masks written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
