# -*- coding: utf-8 -*-
"""Per-model training configuration for the 5-model AirMorph segmentation ensemble.

The mapping below is not guessed: the repo expects exactly 5 airway checkpoints, and
the paper describes exactly 5 models each trained with a distinct topology-preserving
loss. Reading models/airway_model.py, models/airway_model_wingsnet.py and
models/airway_model_break.py pins which checkpoint is which network, which output
channel inference reads, and which on-disk format torch.load expects.

networks/airway_network_break.py is byte-identical to networks/airway_network_wingsnet.py,
which confirms models 4 and 5 differ only by training loss - "break" names the
breakage-sensitive loss, not a separate repair architecture.
"""

from copy import deepcopy

from training.configs.train_config_base import config as base_config


# Architecture kwargs mirroring configs/airway_config.py, so that a checkpoint trained
# here loads into the provided inference wrappers without any code change.
_UNET_ARCH = {
    "in_channels": 1,
    "out_channels": 2,
    "finalsigmoid": 1,
    "fmaps_degree": 16,
    "fmaps_layer_number": 4,
    "GroupNormNumber": 4,
}


MODEL_CONFIGS = {
    "model1": {
        "network": "unet3d",
        "layer_order": "cip",          # configs/airway_config.py: layer_order
        "arch": deepcopy(_UNET_ARCH),
        "loss": "dice_focal",
        "checkpoint_name": "airway_model1.pth",
        "checkpoint_format": "raw",    # models/airway_model.py loads the dict directly
        "output_channel": 1,           # inference reads pred[:, 1, ...]
        "returns_logits": True,        # UNet3D.forward: logits while self.training
    },
    "model2": {
        "network": "unet3d",
        "layer_order": "cpi",          # configs/airway_config.py: layer_order_for_model2
        "arch": deepcopy(_UNET_ARCH),
        "loss": "connectivity_aware",
        "checkpoint_name": "airway_model2.pth",
        "checkpoint_format": "raw",
        "output_channel": 1,
        "returns_logits": True,
    },
    "model3": {
        "network": "unet3d",
        "layer_order": "cip",
        "arch": deepcopy(_UNET_ARCH),
        "loss": "connectivity_aware_lsd",
        "checkpoint_name": "airway_model3.pth",
        "checkpoint_format": "raw",
        "output_channel": 1,
        "returns_logits": True,
    },
    "wingsnet": {
        "network": "wingsnet",
        "arch": {"in_channel": 1, "n_classes": 1},
        "loss": "general_union",
        "checkpoint_name": "wingsnet.ckpt",
        "checkpoint_format": "state_dict",   # torch.load(...)["state_dict"]
        "output_channel": 0,                 # inference reads pred[:, 0, ...]
        "returns_logits": False,             # forward already applies sigmoid
        "deep_supervision": True,
    },
    "break": {
        "network": "wingsnet",
        "arch": {"in_channel": 1, "n_classes": 1},
        "loss": "breakage_sensitive",
        "checkpoint_name": "break1.ckpt",
        "checkpoint_format": "state_dict",
        "output_channel": 0,
        "returns_logits": False,
        "deep_supervision": True,
    },
}

MODEL_NAMES = tuple(MODEL_CONFIGS.keys())


def get_config(model_name: str) -> dict:
    """Merge the shared base config with one model's overrides."""

    if model_name not in MODEL_CONFIGS:
        raise KeyError(
            f"unknown model '{model_name}'; expected one of {list(MODEL_CONFIGS)}")

    merged = deepcopy(base_config)
    merged.update(deepcopy(MODEL_CONFIGS[model_name]))
    merged["model_name"] = model_name
    return merged
