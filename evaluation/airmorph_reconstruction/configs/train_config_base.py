# -*- coding: utf-8 -*-
"""Shared training configuration for the AirMorph binary airway segmentation ensemble.

Values marked (paper) come straight from the AirMorph methods section. Values marked
(deviation) are not specified by the paper; see training/README.md for the full list.

Style deliberately mirrors the provided configs/ modules: a plain module-level `config`
dict resolved against the project root.
"""

import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


config = {
    # ------------------------------------------------------------------ paths
    # Raw challenge data (ATM26). Point these at the extracted download locations.
    # Files are .mha (SimpleITK-readable). Images carry a `_0000` suffix; labels do not.
    #
    #   ATM26 Task-1 (299 cases): imagesTr/ATM_XXX_0000.mha + labelsTr/ATM_XXX.mha
    #                             (binary airway masks)  -> segmentation training.
    #   ATM26 Task-2 NewTrainLabels (118 cases): labelsTr/ATM_XXX.mha
    #                             (branch anatomical-label volumes) -> labeling training;
    #                             their CT/binary masks are reused from Task-1 above.
    "dataset_name": "ATM26",
    "seg_images": os.path.join(project_root, "data/ATM26/imagesTr"),
    "seg_labels": os.path.join(project_root, "data/ATM26/labelsTr"),
    "label_branch_dir": os.path.join(project_root, "data/ATM26/branch_labelsTr"),

    # Preprocessing cache (the big one: ~0.5-0.8 GB per case).
    "cache_dir": os.path.join(project_root, "data/cache"),
    # Per-case graph cache for the labeling stage (small: a few MB per case).
    # NOTE: this mount runs close to 100% full; the first cache build died with
    # OSError [Errno 28] No space left on device. Check `df -h` before rebuilding.
    "label_cache_dir": os.path.join(project_root, "data/label_cache"),
    # Split manifests, written once and reused everywhere.
    "split_path": os.path.join(project_root, "data/splits.json"),
    "label_split_path": os.path.join(project_root, "data/labeling_splits.json"),
    # Training run outputs (logs, intermediate checkpoints).
    "work_dir": os.path.join(project_root, "training/runs"),
    # Final checkpoints land here, in the exact filenames the inference code loads.
    "checkpoint_dir": os.path.join(project_root, "checkpoints"),

    # ------------------------------------------------------- data / preprocess
    # HU window applied by networks.airway_network.lumTrans. Training must match
    # inference exactly, so this is informational only - lumTrans is called directly.
    "hu_window": (-1200.0, 600.0),                      # (paper / repo)
    # Margin around the lung bounding box, matching configs/lungmask_config.py.
    "lung_bbox_margin": 5,                              # (repo)
    # No in-plane resampling: the inference path does none, so training does none.
    "resample_spacing": None,                           # (deviation: unspecified)

    # ------------------------------------------------------------- patching
    "patch_size": (128, 224, 304),                      # (paper)
    # Fraction of patches centred on a centreline point (the paper's "dense cropping
    # around airway-centered regions"); the remainder are drawn from the lung region
    # so the model still sees negative context.
    "airway_centred_ratio": 0.7,                        # (deviation: unspecified)
    # Exponent on branch generation when weighting centreline points. >0 biases
    # sampling toward distal/peripheral branches, per the paper's over-sampling note.
    "peripheral_weight_exponent": 1.0,                  # (deviation: unspecified)
    # Random offset (voxels) applied to the sampled patch centre.
    "centre_jitter": 16,                                # (deviation: unspecified)

    # ---------------------------------------------------------- augmentation
    "aug_flip_prob": 0.5,                               # (paper: random flipping)
    "aug_rotate_prob": 0.5,                             # (deviation: prob unspecified)
    "aug_rotate_degrees": 10.0,                         # (paper: [-10, +10] degrees)

    # ------------------------------------------------------------- optimiser
    "optimizer": "adam",                                # (paper)
    "lr": 2e-3,                                         # (paper: initial LR 0.002)
    "weight_decay": 0.0,                                # (deviation: unspecified)
    "batch_size": 1,                                    # (paper: memory constraints)
    # Paper states neither epoch count nor schedule.
    "max_epochs": 300,                                  # (deviation: unspecified)
    "iters_per_epoch": 250,                             # (deviation: unspecified)
    "lr_schedule": "cosine",                            # (deviation: unspecified)
    "lr_min": 1e-5,                                     # (deviation: unspecified)
    # lr 2e-3 at batch size 1 is aggressive; warmup is close to mandatory.
    "warmup_iters": 500,                                # (deviation: unspecified)
    "poly_power": 0.9,                                  # used when lr_schedule="poly"
    "grad_clip": 12.0,                                  # (deviation: stability guard)
    "amp": True,                                        # (deviation: A100 throughput)
    # The paper names no initialisation, so PyTorch's default is used - which is also
    # what measurement favours. Enabling Kaiming made every model worse and blew up
    # WingsNet's gradients ~1000x; see training/nets/__init__.py::init_weights_kaiming.
    "kaiming_init": False,

    # ------------------------------------------------- two-stage curriculum
    # Paper: "(1) lung region segmentation is used as a hard attention mechanism to
    # suppress irrelevant areas and reduce false positives; (2) early-stage model
    # predictions are used to identify and resample hard patches in later epochs."
    "stage1_epochs": 150,                               # (deviation: split unspecified)
    "use_lung_attention": True,                         # (paper, stage 1)
    "hard_mining_enabled": True,                        # (paper, stage 2)
    # Refresh the hard-patch pool every N epochs during stage 2.
    "hard_mining_interval": 10,                         # (deviation: unspecified)
    # Fraction of stage-2 patches drawn from the mined hard pool.
    "hard_mining_ratio": 0.5,                           # (deviation: unspecified)
    # Fraction of training cases re-scored at each refresh (rotating subset).
    "hard_mining_subset": 0.25,                         # (deviation: unspecified)
    "hard_mining_top_k": 256,                           # (deviation: unspecified)
    # Breakage (false negatives) is what the topology losses target, so weight FN above FP.
    "hard_mining_fn_weight": 3.0,                       # (deviation: unspecified)
    "hard_mining_fp_weight": 1.0,                       # (deviation: unspecified)

    # ------------------------------------------------------------ validation
    "val_interval": 10,                                 # (deviation)
    "val_patches": 40,                                  # patches per validation pass
    # Sliding-window settings mirror configs/airway_config.py so validation matches
    # the deployed inference behaviour.
    "sw_batch_size": 1,
    "overlap": 0.75,
    "sw_mode": "gaussian",
    "sigma_scale": 0.25,

    # ----------------------------------------------------------------- splits
    "split_seed": 20250720,                             # (deviation: unspecified)
    "split_fractions": (0.8, 0.1, 0.1),                 # (deviation: unspecified)

    # ------------------------------------------------------------------ misc
    "num_workers": 4,
    "seed": 20250720,
}
