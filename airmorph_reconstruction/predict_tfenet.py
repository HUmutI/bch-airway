# -*- coding: utf-8 -*-
"""Stage 1 of the TfeNet baseline: run TfeNet over a split and save airway masks.

TfeNet lives in its own conda env (`/path/to/envs/conda-envs/tfenet`,
torch 2.5.1+cu121) with the compiled DAConv/DSConv extensions, so - as with MEDPSeg -
prediction and scoring are split. This script writes masks; `training/score_external.py`
scores them with the metric code used for the five AirMorph models.

The prediction path replicates `TfeNet/run_pipeline.py`, which is three of TfeNet's own
scripts chained:

    evaluation.py     TfeNet and TfeNetSmall each predict the full volume with a
                      128^3 / stride-64 sliding window, overlap-averaged, threshold 0.5
    concat.py         union of the two binary maps
    postprocessing.py largest connected component (connectivity=1) + fill holes

Both variants are saved per case:
    raw   - the union, before post-processing
    post  - largest connected component, TfeNet's shipped final output

CHECKPOINT CHOICE: `TfeNet/evaluation.py` hardcodes `checkpoint/ATM22/`, which is trained
on the ATM22 public training set - the same 299 cases `data/ATM26/imagesTr` holds and our
splits are carved from. Every one of our test cases was almost certainly in its training
data, so that checkpoint is NOT usable as a held-out baseline. This script therefore takes
`--ckpt-set` and defaults to BAS. Use BAS or AIIB23; both are trained on other datasets and
are genuinely held out on our split, at the cost of a domain shift.

Usage (must be the tfenet env):
    /path/to/envs/conda-envs/tfenet/bin/python -m training.predict_tfenet \
        --split test --ckpt-set BAS --out-dir outputs/tfenet/BAS_test_masks
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

logger = logging.getLogger("airmorph.predict_tfenet")

# The original path pointed at the checkout on the machine this was written on.
# Take it from the environment when set, otherwise fall back to the local clone,
# then to the original, so the script runs unchanged on either machine.
TFENET_ROOT = os.environ.get(
    "TFENET_ROOT",
    "/path/to/airway-data/or_all3_run/code/TfeNet"
    if os.path.isdir("/path/to/airway-data/or_all3_run/code/TfeNet")
    else "/path/to/airway-data")

#: TfeNet/evaluation.py:29-30 - the sliding window it ships with.
CUBE_SIZE = 128
STEP = 64


def load_net(ckpt_path: str, device: str):
    """Build TfeNet and load one of its released checkpoints."""

    from importlib import import_module

    _, net = import_module("TfeNet").get_model()
    payload = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(payload["state_dict"])
    return net.to(device).eval()


@torch.no_grad()
def predict_volume(net, volume: torch.Tensor, device: str) -> np.ndarray:
    """The sliding window of `TfeNet/evaluation.py:31-70`, verbatim in behaviour.

    Overlapping windows are summed and divided by their hit count, then thresholded at
    0.5. Accumulators are float32 rather than the original float64 purely to halve RAM;
    the averaging is unaffected at this precision.
    """

    shape = volume.shape
    total = np.zeros(shape[2:], dtype=np.float32)
    counts = np.zeros(shape[2:], dtype=np.float32)
    volume = volume.to(device)

    def starts(extent: int) -> list:
        n = (extent - CUBE_SIZE) // STEP + (1 if (extent - CUBE_SIZE) % STEP == 0 else 2)
        offsets = []
        for i in range(n):
            left = STEP * i
            if left + CUBE_SIZE > extent:
                left = extent - CUBE_SIZE
            offsets.append(left)
        return offsets

    for xl in starts(shape[2]):
        for yl in starts(shape[3]):
            for zl in starts(shape[4]):
                patch = volume[:, :,
                               xl:xl + CUBE_SIZE,
                               yl:yl + CUBE_SIZE,
                               zl:zl + CUBE_SIZE].contiguous()
                out = net(patch).cpu().numpy()[0, 0]
                total[xl:xl + CUBE_SIZE,
                      yl:yl + CUBE_SIZE,
                      zl:zl + CUBE_SIZE] += out
                counts[xl:xl + CUBE_SIZE,
                       yl:yl + CUBE_SIZE,
                       zl:zl + CUBE_SIZE] += 1

    del volume
    torch.cuda.empty_cache()
    return (total / counts >= 0.5).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt-set", default="BAS", choices=["BAS", "AIIB23", "ATM22"],
                        help="ATM22 is train-set-contaminated for our splits; see docstring")
    parser.add_argument("--splits-path",
                        default=os.path.join(_PROJECT_ROOT, "data/splits.json"))
    parser.add_argument("--cache-dir", default=os.path.join(_PROJECT_ROOT, "data/cache"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    if args.ckpt_set == "ATM22":
        logger.warning("ATM22 checkpoint saw our test cases during training - "
                       "its scores are not held-out performance")

    out_dir = args.out_dir or os.path.join(
        _PROJECT_ROOT, f"outputs/tfenet/{args.ckpt_set}_{args.split}_masks")
    # Resolved before the chdir below, or relative paths would land under TFENET_ROOT.
    out_dir = os.path.abspath(out_dir)
    args.splits_path = os.path.abspath(args.splits_path)
    args.cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(out_dir, exist_ok=True)

    # TfeNet imports its model and compiled ops by relative path.
    os.chdir(TFENET_ROOT)
    sys.path.insert(0, TFENET_ROOT)

    from data_ATM22 import lumTrans_hu
    from postprocessing import large_connected_domain
    from utils import load_itk_image

    with open(args.splits_path, "r") as handle:
        keys = json.load(handle)[args.split]
    if args.limit:
        keys = keys[:args.limit]

    nets = {
        "big": load_net(
            os.path.join(TFENET_ROOT, "checkpoint", args.ckpt_set,
                         "TfeNet_checkpoint.ckpt"), args.device),
        "small": load_net(
            os.path.join(TFENET_ROOT, "checkpoint", args.ckpt_set,
                         "TfeNetSmall_checkpoint.ckpt"), args.device),
    }
    logger.info("loaded TfeNet + TfeNetSmall from checkpoint/%s", args.ckpt_set)

    for index, key in enumerate(keys, start=1):
        out_path = os.path.join(out_dir, key.replace("/", "_") + ".npz")
        if os.path.exists(out_path):
            logger.info("[%d/%d] %s (cached)", index, len(keys), key)
            continue

        with open(os.path.join(args.cache_dir, key, "meta.json"), "r") as handle:
            meta = json.load(handle)

        logger.info("[%d/%d] %s", index, len(keys), key)

        # data_ATM22.SegValData.__getitem__: HU window to uint8, then scale to [0, 1].
        image, _, _ = load_itk_image(meta["image_path"])
        image = lumTrans_hu(image).astype(np.float32) / 255.0
        volume = torch.from_numpy(image)[None, None]
        del image

        # concat.py: union of the two models' binary outputs.
        union = np.zeros(volume.shape[2:], dtype=np.uint8)
        for name, net in nets.items():
            union |= predict_volume(net, volume, args.device)
            logger.debug("%s %s done", key, name)
        del volume

        # postprocessing.py: largest connected component, holes filled.
        post = large_connected_domain(union).astype(np.uint8)

        np.savez_compressed(out_path, raw=union, post=post,
                            original_shape=np.array(meta["original_shape"]))
        del union, post

    logger.info("masks written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
