# -*- coding: utf-8 -*-
"""Stage 1 of the MEDPSeg baseline: run MEDPSeg over a split and save airway masks.

MEDPSeg lives in its own conda env (`/path/to/envs/conda-envs/medpseg`,
torch 2.5.1+cu121) which does not have the AirMorph package, and the AirMorph env
(`clip`) does not have MEDPSeg. So prediction and scoring are split into two scripts:
this one writes masks, `training/score_medpseg.py` reads them back and scores them with
the exact metric code used for the 5 AirMorph models.

The prediction path mirrors `medpseg/pipeline.py::pipeline` for the airway branch only -
lung / lobe / GGO / consolidation heads are skipped since they play no part in the
airway comparison:

    read_preprocess(path, norm=True)      # clip to [-1024, 600], min-max, orientation fix
    -> PolySegmentationPipeline(...)      # 2.5D stacked slice prediction
    -> airway = (atm_head[:, 1:] > 0.5)

Two variants are saved per case:
    raw   - the thresholded head, no post-processing (MEDPSeg's default)
    post  - largest connected component only (MEDPSeg's `--post` flag)

Masks are written back in the ORIGINAL image orientation (the read_preprocess
orientation flip is undone) so they line up with the untouched ground truth.

Usage (must be the medpseg env):
    /path/to/envs/conda-envs/medpseg/bin/python -m training.predict_medpseg \
        --split test --out-dir outputs/medpseg/test_masks
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

logger = logging.getLogger("airmorph.predict_medpseg")

#: Where the MEDPSeg checkout keeps its released weights.
# The original path pointed at the source checkout on the machine this was
# written on. Resolve the installed package instead, so the weights are found
# wherever medpseg happens to be installed, and keep the old path as a fallback.
def _medpseg_pkg():
    try:
        import importlib.util
        spec = importlib.util.find_spec("medpseg")
        if spec and spec.submodule_search_locations:
            return list(spec.submodule_search_locations)[0]
    except Exception:
        pass
    return "/path/to/airway-data"

MEDPSEG_PKG = _medpseg_pkg()
POLY_WEIGHT = os.path.join(MEDPSEG_PKG, "poly_medseg_25d_fix.ckpt")


class QueueShim:
    """Stands in for the GUI's `mp.Queue`.

    `PolySegmentationPipeline` wraps whatever it is handed in `PrintInterface`, which
    only ever calls `.put((tag, payload))`. Progress chatter is dropped; `write`
    messages go to the log so a stuck case is visible.
    """

    def put(self, item):
        tag, payload = item
        if tag == "write":
            logger.debug("medpseg: %s", str(payload).strip())


def unflip(array, dir_array):
    """Undo the axial orientation fix `read_preprocess` applies.

    `np.flip` is its own inverse, so replaying the same axes restores the original
    frame. Identity-direction scans (all of ATM26) are unaffected.
    """

    dir_array = np.asarray(dir_array)
    if len(dir_array) != 9:
        return array
    axes = np.where(dir_array[[0, 4, 8]][::-1] < 0)[0]
    if len(axes) == 0:
        return array
    return np.flip(array, axes).copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--splits-path",
                        default=os.path.join(_PROJECT_ROOT, "data/splits.json"))
    parser.add_argument("--cache-dir", default=os.path.join(_PROJECT_ROOT, "data/cache"))
    parser.add_argument("--out-dir",
                        default=os.path.join(_PROJECT_ROOT, "outputs/medpseg/test_masks"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    from medpseg.image_read import read_preprocess
    from medpseg.poly_pipeline import PolySegmentationPipeline, get_connected_components

    with open(args.splits_path, "r") as handle:
        keys = json.load(handle)[args.split]
    if args.limit:
        keys = keys[:args.limit]

    os.makedirs(args.out_dir, exist_ok=True)

    model = PolySegmentationPipeline(
        weight=POLY_WEIGHT,
        batch_size=args.batch_size,
        cpu=False,
        output_dir=args.out_dir,
        post=False,       # both variants are derived here instead
        cli=True)
    logger.info("MEDPSeg loaded from %s", POLY_WEIGHT)

    for index, key in enumerate(keys, start=1):
        out_path = os.path.join(args.out_dir, key.replace("/", "_") + ".npz")
        if os.path.exists(out_path):
            logger.info("[%d/%d] %s (cached)", index, len(keys), key)
            continue

        with open(os.path.join(args.cache_dir, key, "meta.json"), "r") as handle:
            meta = json.load(handle)

        logger.info("[%d/%d] %s", index, len(keys), key)

        data, _, _, _, directions, original_image = read_preprocess(
            meta["image_path"], norm=True)
        dir_array = np.asarray(directions)

        # [Z, H, W] -> [1, 1, Z, H, W], the shape the pipeline expects.
        volume = data.unsqueeze(0).unsqueeze(0)

        _, airway, _ = model(volume, QueueShim(), None, original_image,
                             os.path.basename(key), dir_array, False)

        airway = airway.astype(np.uint8)
        airway_post = get_connected_components(
            airway, return_largest=1)[0].astype(np.uint8)

        np.savez_compressed(
            out_path,
            raw=unflip(airway, dir_array),
            post=unflip(airway_post, dir_array),
            original_shape=np.array(meta["original_shape"]))

        del data, volume, airway, airway_post
        torch.cuda.empty_cache()

    logger.info("masks written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
