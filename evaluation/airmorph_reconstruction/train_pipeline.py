# -*- coding: utf-8 -*-
"""Decoupled full-pipeline training for AirMorph ("both segmentation and labeling").

The seg -> skeleton -> graph -> labeling pipeline is not differentiable end-to-end
(skeletonisation and the pycuda graph build break the gradient), so "both" is an
orchestration, not a joint-gradient model:

    Stage A  train the 5-model binary segmentation ensemble on the 299 ATM26 cases
    Stage B  build the 118 labeling graphs from the GROUND-TRUTH binary masks, then train
             the labeling transformer on them

producing all six checkpoints (`airway_model{1,2,3}.pth`, `wingsnet.ckpt`, `break1.ckpt`,
`airway_cls.ckpt`) = the complete deployable pipeline. This wraps the standalone entry
points as subprocesses so each trains with clean process state.

Usage:
    python -m training.train_pipeline                     # sequential, cuda:0
    python -m training.train_pipeline --parallel-seg --gpus 0 1 2 3
    python -m training.train_pipeline --skip-seg          # labeling only (seg already done)
    python -m training.train_pipeline --skip-labeling     # seg only
"""

import argparse
import logging
import os
import subprocess
import sys
from typing import List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from training.configs.train_config_models import MODEL_NAMES            # noqa: E402

logger = logging.getLogger("airmorph.train_pipeline")
PY = sys.executable


def _run(cmd: List[str], env_gpu: Optional[str] = None) -> int:
    env = os.environ.copy()
    if env_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = env_gpu
    logger.info("RUN %s%s", f"[CUDA={env_gpu}] " if env_gpu else "", " ".join(cmd))
    return subprocess.run(cmd, env=env, cwd=_PROJECT_ROOT).returncode


def train_segmentation(parallel: bool, gpus: List[str], extra: List[str]) -> None:
    models = list(MODEL_NAMES)
    if parallel and gpus:
        # One model per GPU, cycling through the GPU pool; wait for each wave.
        pending = list(models)
        while pending:
            wave, pending = pending[: len(gpus)], pending[len(gpus):]
            procs = []
            for model, gpu in zip(wave, gpus):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                cmd = [PY, "-m", "training.train", "--model", model,
                       "--device", "cuda:0", *extra]
                logger.info("RUN [CUDA=%s] %s", gpu, " ".join(cmd))
                procs.append((model, subprocess.Popen(cmd, env=env, cwd=_PROJECT_ROOT)))
            for model, proc in procs:
                if proc.wait() != 0:
                    raise RuntimeError(f"segmentation model '{model}' failed")
    else:
        for model in models:
            cmd = [PY, "-m", "training.train", "--model", model,
                   "--device", "cuda:0", *extra]
            if _run(cmd, env_gpu=gpus[0] if gpus else None) != 0:
                raise RuntimeError(f"segmentation model '{model}' failed")
    logger.info("segmentation stage complete")


def train_labeling(gpu: Optional[str], prepare: bool, extra: List[str]) -> None:
    if prepare:
        if _run([PY, "-m", "training.data.prepare_labeling"], env_gpu=gpu) != 0:
            raise RuntimeError("labeling graph preparation failed")
    if _run([PY, "-m", "training.train_labeling", *extra], env_gpu=gpu) != 0:
        raise RuntimeError("labeling training failed")
    logger.info("labeling stage complete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="*", default=["0"],
                        help="GPU ids to use (first is used for sequential/labeling)")
    parser.add_argument("--parallel-seg", action="store_true",
                        help="train seg models concurrently, one per GPU")
    parser.add_argument("--skip-seg", action="store_true")
    parser.add_argument("--skip-labeling", action="store_true")
    parser.add_argument("--skip-prepare-labeling", action="store_true",
                        help="assume labeling graph caches already exist")
    parser.add_argument("--seg-args", nargs="*", default=[],
                        help="extra args forwarded to training.train")
    parser.add_argument("--labeling-args", nargs="*", default=[],
                        help="extra args forwarded to training.train_labeling")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    if not args.skip_seg:
        logger.info("=== Stage A: segmentation ensemble ===")
        train_segmentation(args.parallel_seg, args.gpus, args.seg_args)
    if not args.skip_labeling:
        logger.info("=== Stage B: anatomical labeling ===")
        train_labeling(args.gpus[0] if args.gpus else None,
                       prepare=not args.skip_prepare_labeling,
                       extra=args.labeling_args)

    logger.info("pipeline complete; checkpoints in the configured checkpoint_dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
