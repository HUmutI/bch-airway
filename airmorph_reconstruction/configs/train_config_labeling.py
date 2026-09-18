# -*- coding: utf-8 -*-
"""Training configuration for the AirMorph anatomical-labeling transformer (`our_net`).

The architecture hyperparameters mirror `configs/labeling_config.py` exactly, because a
checkpoint trained here must load into the provided `models/labeling_model.py` without any
code change (`our_net` is state-dict-sensitive to dim / heads / depth). The training knobs
(lr, epochs, DropHead p, aux weight) come from the paper's Methods where stated and are
documented deviations where the paper is silent.

Paper (p25) training details for the labeling stage:
  - Adam optimizer, learning rate 5e-4, 600 epochs.
  - Lobar + segmental: two Transformer layers each; subsegmental: an additional two layers
    with SSC (soft-subtree-consistency) modules. Hidden dimension 128.
    (The text also says "32 heads", which conflicts with the shipped config's heads=4 /
    dim_head=32 giving inner_dim=128; we follow the shipped config so weights stay
    loadable, and treat "32 heads" as a wording slip for dim_head=32. Documented.)
  - No explicit labeling loss is given -> cross-entropy deep-supervision (see
    training/losses/labeling_loss.py).
"""

import os

from training.configs.train_config_base import config as base_config

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


config = {
    # ---------------------------------------------------------------- device
    "device": "cuda:0",
    "seed": base_config["seed"],

    # ------------------------------------------------- architecture (== labeling_config.py)
    "input_dim": 11,          # the 11 graph-node features (paper Table 3)
    "num_classes1": 6,        # lobar
    "num_classes2": 20,       # segmental
    "num_classes3": 82,       # subsegmental
    "dim": 128,               # (paper: hidden dimension 128)
    "heads": 4,               # shipped config (paper text "32 heads" -> dim_head, see above)
    "mlp_dim": 256,
    "dim_head": 32,
    "dropout": 0.0,           # (shipped config)
    "trans_depth": 2,         # (paper: two Transformer layers per level)
    "outlier_depth": 2,       # (paper: +two layers with SSC at subsegmental level)

    # ------------------------------------------------------------- optimiser
    "optimizer": "adam",                       # (paper)
    "lr": 5e-4,                                # (paper: learning rate 5e-4)
    "weight_decay": 0.0,                       # (deviation: unspecified)
    "max_epochs": 600,                         # (paper: 600 epochs)
    "grad_clip": 5.0,                          # (deviation: stability guard)
    # One graph (case) per forward pass (variable N nodes; the model batches internally).
    # Accumulate gradients over this many graphs before an optimizer step to smooth the
    # per-case noise a batch size of 1 graph produces. (deviation: unspecified)
    "grad_accum": 4,
    "amp": False,             # tiny model; fp32 is simpler and the attention has a python
                              # head-drop loop that bf16 would not speed up.

    # ------------------------------------------------------- DropHead regulariser
    # `p` = the 4th argument to our_net.forward, driving the stochastic whole-head drop in
    # Attention_spd (networks/transformer.py:62-77). The paper does not name it; the
    # released code clearly intends a train-time head-drop. On (0.1) for training, off (0)
    # for eval/inference. (deviation: value unspecified)
    "drophead_p": 0.1,

    # --------------------------------------------------------------- loss weights
    # Cross-entropy deep-supervision over the six classification heads. Stage-2 is what
    # inference reads, so it gets full weight; stage-1 is an auxiliary head. (deviation:
    # loss unspecified by the paper)
    "aux_weight": 0.5,
    # Level weights (lobar / segmental / subsegmental) applied to both stages.
    "level_weights": (1.0, 1.0, 1.0),
    # Optional inverse-frequency class weighting per level. Off by default (faithful:
    # the paper mentions none). (deviation)
    "class_balanced": False,
    # Target value used to ignore a node in the CE (e.g. unlabeled / out-of-taxonomy).
    "ignore_index": -1,

    # Subsegmental supervision. The public ATM26 Task-2 labels are SEGMENTAL only (values
    # 1..20 -> class2anno seg id v-1); there is no subsegmental (a/b/c) ground truth. So:
    #   "base_segment" -> the subseg head is supervised toward each segment's base subseg
    #                     class (segment-granularity; keeps the head trained, no a/b/c).
    #   "ignore"       -> the subseg head is left unsupervised (stays near init).
    # Lobar (6) and segmental (20) heads always get real ground truth. (deviation forced by
    # the data: subsegmental labels were never released.)
    "subseg_policy": "base_segment",

    # ------------------------------------------------------------- validation
    "val_interval": 10,                        # epochs between validation passes

    # ----------------------------------------------------------------- paths
    "label_cache_dir": base_config["label_cache_dir"],
    "label_split_path": base_config["label_split_path"],
    "split_path": base_config["split_path"],
    "seg_images": base_config["seg_images"],
    "seg_labels": base_config["seg_labels"],
    "label_branch_dir": base_config["label_branch_dir"],
    "dataset_name": base_config["dataset_name"],
    "work_dir": os.path.join(base_config["work_dir"], "labeling"),
    "checkpoint_dir": base_config["checkpoint_dir"],
    # The exact filename + format models/labeling_model.py::load_checkpoint expects.
    "checkpoint_name": "airway_cls.ckpt",
    "checkpoint_format": "state_dict",         # torch.load(path)["state_dict"]

    # Provided taxonomy resources (reused, not copied).
    "class2anno_path": os.path.join(project_root, "configs/class2anno.json"),
    "label_old2new_path": os.path.join(project_root, "models/label_old2new.json"),
}


def get_config() -> dict:
    """Return a deep copy of the labeling training config."""

    from copy import deepcopy
    merged = deepcopy(config)
    merged["model_name"] = "labeling"
    return merged
