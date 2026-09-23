# -*- coding: utf-8 -*-
"""Trainer for the AirMorph anatomical-labeling transformer (`networks/transformer.py::our_net`).

The transformer classifies every node of one airway graph at three anatomical levels. Each
step processes one whole graph (variable node count; the model adds the batch dim itself),
with gradient accumulation over a few graphs to smooth the per-case noise.

Paper (p25): Adam, lr 5e-4, 600 epochs. The loss is unspecified -> cross-entropy
deep-supervision over the six heads (training/losses/labeling_loss.py). The 4th forward
argument `p` is the stochastic DropHead probability (on for training, 0 for eval).

The checkpoint is written as `{"state_dict": ...}` to `checkpoints/airway_cls.ckpt`, the
exact layout `models/labeling_model.py::load_checkpoint` loads, so a trained model drops
straight into the released inference pipeline.

Usage:
    python -m training.train_labeling
    python -m training.train_labeling --overfit 1 --max-epochs 300   # sanity check
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from training.configs.train_config_labeling import get_config              # noqa: E402
from training.data.labeling_dataset import LabelingGraphDataset            # noqa: E402
from training.losses.labeling_loss import (                               # noqa: E402
    LabelingDeepSupervisionLoss,
    build_class_weights,
)
from models.evaluation_matrics import calculate_CS                        # noqa: E402
from networks.transformer import our_net                                  # noqa: E402

logger = logging.getLogger("airmorph.train_labeling")


def build_model(config: Dict, device: str) -> torch.nn.Module:
    net = our_net(
        input_dim=config["input_dim"],
        num_classes1=config["num_classes1"],
        num_classes2=config["num_classes2"],
        num_classes3=config["num_classes3"],
        dim=config["dim"],
        heads=config["heads"],
        mlp_dim=config["mlp_dim"],
        dim_head=config["dim_head"],
        dropout=config["dropout"],
        trans_depth=config["trans_depth"],
        outlier_depth=config["outlier_depth"],
    )
    return net.to(device)


class LabelingTrainer:

    def __init__(self, config: Dict, device: str):
        self.config = config
        self.device = device

        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

        self.run_dir = config["work_dir"]
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(config["checkpoint_dir"], exist_ok=True)

        self.net = build_model(config, device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=config["lr"],
            weight_decay=config["weight_decay"])

        self.train_set = LabelingGraphDataset(config, split="train")
        self.val_set = LabelingGraphDataset(config, split="val")

        class_weights = None
        if config.get("class_balanced"):
            class_weights = build_class_weights(
                self.train_set,
                (config["num_classes1"], config["num_classes2"], config["num_classes3"]),
                ignore_index=config["ignore_index"])
            logger.info("using inverse-frequency class weights")

        self.criterion = LabelingDeepSupervisionLoss(
            aux_weight=config["aux_weight"],
            level_weights=config["level_weights"],
            ignore_index=config["ignore_index"],
            class_weights=class_weights)

        self.p_train = float(config["drophead_p"])
        self.grad_accum = max(1, int(config["grad_accum"]))
        self.ignore_index = config["ignore_index"]
        self.best_metric = -1.0

    # ------------------------------------------------------------------ helpers

    def _to_device(self, item: Dict) -> Dict:
        out = {}
        for key in ("x", "spd", "mask_top", "y_lob", "y_seg", "y_sub"):
            out[key] = item[key].to(self.device, non_blocking=True)
        return out

    def _forward(self, batch: Dict, p: float) -> Tuple:
        return self.net(batch["x"], batch["mask_top"], batch["spd"], p)

    # ------------------------------------------------------------------ training

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.net.train()
        order = np.random.permutation(len(self.train_set))

        running = 0.0
        count = 0
        self.optimizer.zero_grad(set_to_none=True)

        for i, index in enumerate(order):
            batch = self._to_device(self.train_set[int(index)])
            outputs = self._forward(batch, self.p_train)
            targets = {"y_lob": batch["y_lob"], "y_seg": batch["y_seg"],
                       "y_sub": batch["y_sub"]}
            loss, logs = self.criterion(outputs, targets)

            (loss / self.grad_accum).backward()
            running += logs["loss"]
            count += 1

            if (i + 1) % self.grad_accum == 0 or (i + 1) == len(order):
                if self.config["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.config["grad_clip"])
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        return {"loss": running / max(1, count)}

    # ---------------------------------------------------------------- validation

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.net.eval()
        names = ("lob", "seg", "sub")
        correct = {n: 0 for n in names}
        total = {n: 0 for n in names}
        cs_seg_sum = cs_sub_sum = cs_count = 0.0

        for index in range(len(self.val_set)):
            item = self.val_set[index]
            batch = self._to_device(item)
            outputs = self._forward(batch, 0.0)            # DropHead off at eval
            _, _, _, x1_2, x2_2, x3_2 = outputs[:6]
            preds = {
                "lob": x1_2.argmax(dim=1),
                "seg": x2_2.argmax(dim=1),
                "sub": x3_2.argmax(dim=1),
            }
            ys = {"lob": batch["y_lob"], "seg": batch["y_seg"], "sub": batch["y_sub"]}
            for n in names:
                mask = ys[n] != self.ignore_index
                total[n] += int(mask.sum().item())
                correct[n] += int(((preds[n] == ys[n]) & mask).sum().item())

            # TreeCons (consistency), the paper's topology metric, via the provided helper.
            pred_seg = preds["seg"].cpu().numpy()
            pred_sub = preds["sub"].cpu().numpy()
            mask_top = item["mask_top"].numpy()
            try:
                cs_seg, cs_sub = calculate_CS(
                    pred_seg, pred_sub, mask_top, self.config["label_old2new_path"])
                if not (np.isnan(cs_seg) or np.isnan(cs_sub)):
                    cs_seg_sum += cs_seg
                    cs_sub_sum += cs_sub
                    cs_count += 1
            except Exception:
                pass

        metrics = {f"acc_{n}": correct[n] / max(1, total[n]) for n in names}
        if cs_count:
            metrics["treecons_seg"] = cs_seg_sum / cs_count
            metrics["treecons_sub"] = cs_sub_sum / cs_count
        return metrics

    # ---------------------------------------------------------------- checkpoint

    def export(self) -> str:
        path = os.path.join(self.config["checkpoint_dir"], self.config["checkpoint_name"])
        state_dict = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        torch.save({"state_dict": state_dict}, path)
        return path

    # ---------------------------------------------------------------------- loop

    def run(self) -> None:
        logger.info("labeling training on %s: train=%d val=%d",
                    self.device, len(self.train_set), len(self.val_set))
        if len(self.train_set) == 0:
            raise RuntimeError(
                "no cached graphs. Run: python -m training.data.prepare_labeling")

        history: List[Dict] = []
        for epoch in range(self.config["max_epochs"]):
            start = time.time()
            stats = self.train_one_epoch(epoch)

            if (epoch + 1) % self.config["val_interval"] == 0 or \
                    (epoch + 1) == self.config["max_epochs"]:
                val = self.validate()
                # Select on subsegmental accuracy: the finest, hardest, clinically key level.
                metric = val.get("acc_sub", 0.0)
                logger.info(
                    "epoch %d loss %.4f | acc lob/seg/sub %.3f/%.3f/%.3f "
                    "treecons seg/sub %.3f/%.3f (%.1fs)",
                    epoch, stats["loss"], val.get("acc_lob", 0), val.get("acc_seg", 0),
                    val.get("acc_sub", 0), val.get("treecons_seg", 0),
                    val.get("treecons_sub", 0), time.time() - start)

                if metric > self.best_metric:
                    self.best_metric = metric
                    path = self.export()
                    logger.info("new best (acc_sub %.4f); exported to %s", metric, path)

                history.append({"epoch": epoch, **stats, **val, "metric": metric})
                with open(os.path.join(self.run_dir, "history.json"), "w") as handle:
                    json.dump(history, handle, indent=2)
            else:
                logger.info("epoch %d loss %.4f (%.1fs)",
                            epoch, stats["loss"], time.time() - start)

        logger.info("finished labeling training, best acc_sub %.4f", self.best_metric)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--overfit", type=int, default=None,
                        help="sanity check: train on N graphs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    config = get_config()
    device = args.device or config["device"]
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.overfit:
        config["drophead_p"] = 0.0     # remove stochasticity for a clean memorisation test

    trainer = LabelingTrainer(config, device)

    # Persistent file log under the run dir, robust to launcher capture failures.
    file_handler = logging.FileHandler(os.path.join(trainer.run_dir, "train.log"))
    file_handler.setFormatter(logging.Formatter(
        "(%(asctime)s)(%(levelname)s) %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    if args.overfit:
        trainer.train_set.keys = trainer.train_set.keys[: args.overfit]
        trainer.val_set.keys = trainer.train_set.keys[: args.overfit]
        logger.info("overfit mode on %s", trainer.train_set.keys)

    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
