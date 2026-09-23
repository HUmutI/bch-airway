# -*- coding: utf-8 -*-
"""Trainer for one member of the AirMorph binary airway segmentation ensemble.

Run one model per GPU. Batch size 1 is a paper-specified hyperparameter, so DDP is
deliberately not used: sharding across 4 GPUs would make the effective batch 4 and
change the gradient-noise scale that lr = 2e-3 was tuned against. Since five independent
models have to be trained anyway, the embarrassingly-parallel axis is free:

    CUDA_VISIBLE_DEVICES=0 python -m training.train --model model1 &
    CUDA_VISIBLE_DEVICES=1 python -m training.train --model model2 &
    CUDA_VISIBLE_DEVICES=2 python -m training.train --model model3 &
    CUDA_VISIBLE_DEVICES=3 python -m training.train --model wingsnet &
    # then, when a GPU frees up:
    CUDA_VISIBLE_DEVICES=0 python -m training.train --model break

Isolating one GPU per process also makes `droplayer`'s bare `.cuda()` call
(`networks/airway_network_wingsnet.py:101`) correct by construction.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from training.configs.train_config_models import MODEL_NAMES, get_config  # noqa: E402
from training.data.airway_dataset import AirwayPatchDataset               # noqa: E402
from training.losses import DeepSupervisionLoss, build_loss               # noqa: E402
from training.nets import (                                              # noqa: E402
    ProbabilityPredictor,
    build_network,
    forward_probability,
    save_checkpoint,
)

logger = logging.getLogger("airmorph.train")


# ------------------------------------------------------------------- lr schedule

def learning_rate_at(step: int, total_steps: int, config: Dict) -> float:
    """Warmup then decay.

    The paper states the initial LR (0.002) and the optimiser (Adam) but neither the
    epoch count nor the schedule. lr 2e-3 at batch size 1 is aggressive enough that a
    short warmup is close to mandatory for stability, so we add one. Documented
    deviation.
    """

    base_lr = config["lr"]
    min_lr = config["lr_min"]
    warmup = config.get("warmup_iters", 0)

    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup

    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)

    schedule = config["lr_schedule"]
    if schedule == "cosine":
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    if schedule == "poly":
        return max(min_lr, base_lr * (1.0 - progress) ** config.get("poly_power", 0.9))
    if schedule == "constant":
        return base_lr
    raise KeyError(f"unknown lr_schedule '{schedule}'")


# ------------------------------------------------------------------------ metrics

@torch.no_grad()
def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    denominator = pred_bin.sum() + target.sum()
    return float((2.0 * intersection / (denominator + eps)).item())


@torch.no_grad()
def centerline_recall(
    pred: torch.Tensor, centerline: torch.Tensor, eps: float = 1e-6
) -> float:
    """Fraction of ground-truth centreline voxels covered by the prediction.

    A cheap in-loop proxy for the paper's tree-length detected (TLD); the real TLD is
    computed by training/evaluate.py using util/airwayatlas_eval.py.
    """

    pred_bin = (pred > 0.5).float()
    return float(((pred_bin * centerline).sum() / (centerline.sum() + eps)).item())


# ------------------------------------------------------------------------ trainer

class Trainer:

    def __init__(self, config: Dict, device: str):
        self.config = config
        self.device = device
        self.model_name = config["model_name"]

        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        torch.backends.cudnn.benchmark = True

        self.run_dir = os.path.join(config["work_dir"], self.model_name)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(config["checkpoint_dir"], exist_ok=True)

        self.net = build_network(config, device)
        self.base_loss = build_loss(config["loss"])
        self.deep_supervision = (
            DeepSupervisionLoss(self.base_loss)
            if config.get("deep_supervision") else None)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"])

        self.train_set = AirwayPatchDataset(config, split="train", augment=True)
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=config["batch_size"],
            shuffle=False,          # the dataset samples randomly itself
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=config["num_workers"] > 0)

        self.val_set = AirwayPatchDataset(
            config, split="val", augment=False,
            length=config.get("val_patches", 40))
        self.val_loader = DataLoader(
            self.val_set, batch_size=1, shuffle=False,
            num_workers=min(2, config["num_workers"]))

        self.amp_enabled = bool(config["amp"])
        # bfloat16 rather than float16: the losses reduce over ~8.7M voxels and GUL
        # raises p to a fractional power, both of which overflow or lose precision in
        # fp16's narrow range. bf16 needs no GradScaler.
        self.amp_dtype = torch.bfloat16

        self.global_step = 0
        self.best_metric = -1.0
        self.total_steps = config["max_epochs"] * config["iters_per_epoch"]

    # ---------------------------------------------------------------- curriculum

    def _stage(self, epoch: int) -> str:
        return "stage1" if epoch < self.config["stage1_epochs"] else "stage2"

    def _loss_meta(self, batch: Dict, epoch: int) -> Dict:
        """Assemble the auxiliary maps the losses consume."""

        meta = {
            "centerline": batch["centerline"],
            "dist_to_centerline": batch["dist_to_centerline"],
        }

        # Stage-1 hard attention. The paper uses the lung segmentation "to suppress
        # irrelevant areas and reduce false positives". We apply it to the *loss* rather
        # than the input: the deployed pipeline never multiplies the CT by the lung mask
        # (it only crops to the lung bounding box), so masking the input would introduce
        # a train/test mismatch, whereas masking the loss does not.
        if (self.config["use_lung_attention"]
                and self._stage(epoch) == "stage1"):
            meta["loss_mask"] = batch["lung"]

        return meta

    # ------------------------------------------------------------------- training

    def _run_step(self, batch: Dict, epoch: int) -> Dict[str, float]:
        images = batch["image"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)

        meta = self._loss_meta(batch, epoch)
        meta = {key: value.to(self.device, non_blocking=True)
                for key, value in meta.items()}

        lr = learning_rate_at(self.global_step, self.total_steps, self.config)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            prob, aux, sides = forward_probability(self.net, images, self.config)

        # Reductions in fp32: summing millions of small values in bf16 discards exactly
        # the faint peripheral-branch contributions these losses exist to preserve.
        prob = prob.float()
        if self.deep_supervision is not None:
            aux = aux.float() if aux is not None else None
            sides = [s.float() for s in sides] if sides else None
            loss = self.deep_supervision(prob, aux, sides, labels, meta)
        else:
            loss = self.base_loss(prob, labels, meta)

        loss.backward()
        if self.config["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.config["grad_clip"])
        self.optimizer.step()
        self.global_step += 1

        return {
            "loss": float(loss.detach().item()),
            "lr": lr,
            "dice": dice_score(prob.detach(), labels),
        }

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.net.train()
        stage = self._stage(epoch)

        if stage == "stage2" and self.config["hard_mining_enabled"]:
            self.train_set.set_hard_ratio(self.config["hard_mining_ratio"])
        else:
            self.train_set.set_hard_ratio(0.0)

        totals = {"loss": 0.0, "dice": 0.0}
        count = 0
        start = time.time()

        for batch in self.train_loader:
            stats = self._run_step(batch, epoch)
            totals["loss"] += stats["loss"]
            totals["dice"] += stats["dice"]
            count += 1

            if count % 50 == 0:
                logger.info(
                    "epoch %d [%s] step %d/%d loss %.4f dice %.4f lr %.2e",
                    epoch, stage, count, len(self.train_loader),
                    totals["loss"] / count, totals["dice"] / count, stats["lr"])

        elapsed = time.time() - start
        return {
            "loss": totals["loss"] / max(1, count),
            "dice": totals["dice"] / max(1, count),
            "seconds": elapsed,
        }

    # ----------------------------------------------------------------- validation

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Patch-level validation.

        Cheap by design - it runs every `val_interval` epochs purely for model
        selection. Whole-volume sliding-window evaluation with the paper's metrics is
        training/evaluate.py's job.
        """

        self.net.eval()
        dice_total = 0.0
        recall_total = 0.0
        count = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            centerline = batch["centerline"].to(self.device)

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                prob, _, _ = forward_probability(self.net, images, self.config)
            prob = prob.float()

            dice_total += dice_score(prob, labels)
            recall_total += centerline_recall(prob, centerline)
            count += 1

        return {
            "dice": dice_total / max(1, count),
            "centerline_recall": recall_total / max(1, count),
        }

    # -------------------------------------------------------------- hard mining

    @torch.no_grad()
    def refresh_hard_patches(self) -> None:
        """Stage-2 curriculum: mine patches the current model gets wrong.

        Paper: "early-stage model predictions are used to identify and resample hard
        patches in later epochs." The criterion and schedule are unspecified; we score
        each ground-truth centreline voxel by the local prediction error, weighting false
        negatives above false positives because breakage is what the topology-preserving
        losses target.
        """

        from monai.inferers import sliding_window_inference
        from scipy import ndimage

        self.net.eval()
        predictor = ProbabilityPredictor(self.net, self.config)

        rng = np.random.default_rng(self.config["seed"] + self.global_step)
        keys = list(self.train_set.keys)
        subset_size = max(1, int(round(self.config.get("hard_mining_subset", 0.25)
                                       * len(keys))))
        subset = rng.choice(keys, size=subset_size, replace=False)

        top_k = self.config.get("hard_mining_top_k", 256)
        w_fn = self.config.get("hard_mining_fn_weight", 3.0)
        w_fp = self.config.get("hard_mining_fp_weight", 1.0)

        for key in subset:
            case = self.train_set.cases[key]
            image = np.asarray(case.array("image"), dtype=np.float32)
            label = np.asarray(case.array("label"), dtype=np.float32)

            tensor = torch.from_numpy(image)[None, None].to(self.device)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                prob = sliding_window_inference(
                    inputs=tensor,
                    roi_size=self.config["patch_size"],
                    sw_batch_size=self.config["sw_batch_size"],
                    predictor=predictor,
                    overlap=0.25,          # coarse; this is only for ranking
                    mode=self.config["sw_mode"],
                    sigma_scale=self.config["sigma_scale"])
            pred = (prob.float()[0, 0].cpu().numpy() > 0.5).astype(np.float32)

            error = w_fn * np.clip(label - pred, 0, 1) + w_fp * np.clip(pred - label, 0, 1)
            # Local error density around each centreline voxel, in one pass.
            density = ndimage.uniform_filter(error, size=32, mode="constant")

            points = case.sampler.points
            if points is None or len(points) == 0:
                continue
            scores = density[points[:, 0], points[:, 1], points[:, 2]]
            order = np.argsort(scores)[::-1][:top_k]
            self.train_set.set_hard_centres(key, points[order])

            del tensor, prob
            torch.cuda.empty_cache()

        logger.info("refreshed hard patches on %d cases", len(subset))

    # ---------------------------------------------------------------- checkpoints

    def export(self, tag: Optional[str] = None) -> str:
        """Write the checkpoint in the format the released inference code loads."""

        name = self.config["checkpoint_name"]
        if tag:
            path = os.path.join(self.run_dir, f"{tag}_{name}")
        else:
            path = os.path.join(self.config["checkpoint_dir"], name)

        save_checkpoint(self.net, path, self.config["checkpoint_format"])
        return path

    def save_train_state(self, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(self.run_dir, "train_state.pt"))

    # ---------------------------------------------------------------------- loop

    def run(self) -> None:
        logger.info("training %s on %s", self.model_name, self.device)
        logger.info("loss=%s network=%s checkpoint=%s (%s)",
                    self.config["loss"], self.config["network"],
                    self.config["checkpoint_name"], self.config["checkpoint_format"])
        logger.info("train cases=%d val cases=%d",
                    len(self.train_set.keys), len(self.val_set.keys))

        history = []
        for epoch in range(self.config["max_epochs"]):
            stats = self.train_one_epoch(epoch)
            logger.info("epoch %d done: loss %.4f dice %.4f (%.1f s)",
                        epoch, stats["loss"], stats["dice"], stats["seconds"])

            if (self.config["hard_mining_enabled"]
                    and self._stage(epoch) == "stage2"
                    and epoch % self.config["hard_mining_interval"] == 0):
                self.refresh_hard_patches()

            if (epoch + 1) % self.config["val_interval"] == 0:
                val = self.validate()
                # Select on a blend of overlap and centreline coverage: a model that
                # scores well on Dice but drops peripheral branches is not what this
                # ensemble is for.
                metric = 0.5 * val["dice"] + 0.5 * val["centerline_recall"]
                logger.info("epoch %d val: dice %.4f cl-recall %.4f metric %.4f",
                            epoch, val["dice"], val["centerline_recall"], metric)

                if metric > self.best_metric:
                    self.best_metric = metric
                    path = self.export()
                    logger.info("new best (%.4f); exported to %s", metric, path)

                history.append({"epoch": epoch, **stats, **val, "metric": metric})
                with open(os.path.join(self.run_dir, "history.json"), "w") as handle:
                    json.dump(history, handle, indent=2)

            self.save_train_state(epoch)

        logger.info("finished %s, best metric %.4f", self.model_name, self.best_metric)


# -------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--iters-per-epoch", type=int, default=None)
    parser.add_argument("--overfit", type=int, default=None,
                        help="sanity check: train on N cases with no augmentation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    config = get_config(args.model)
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.iters_per_epoch is not None:
        config["iters_per_epoch"] = args.iters_per_epoch
    if args.overfit:
        config["overfit"] = args.overfit
        config["aug_flip_prob"] = 0.0
        config["aug_rotate_prob"] = 0.0
        config["use_lung_attention"] = False
        config["hard_mining_enabled"] = False

    trainer = Trainer(config, args.device)

    # Also log to a file under the run dir, so a run that dies (OOM-kill, GPU eviction,
    # closed terminal) still leaves a trace even when the launcher's own capture fails.
    file_handler = logging.FileHandler(os.path.join(trainer.run_dir, "train.log"))
    file_handler.setFormatter(logging.Formatter(
        "(%(asctime)s)(%(levelname)s) %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    if args.overfit:
        # Restrict to the first N training cases; if the model cannot memorise those,
        # nothing downstream is meaningful.
        trainer.train_set.keys = trainer.train_set.keys[: args.overfit]
        logger.info("overfit mode on %s", trainer.train_set.keys)

    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
