# -*- coding: utf-8 -*-
"""Loss factory for the five AirMorph segmentation models."""

import torch.nn as nn

from training.losses.breakage_sensitive import BreakageSensitiveLoss
from training.losses.connectivity_aware import ConnectivityAwareLoss
from training.losses.deep_supervision import DeepSupervisionLoss
from training.losses.dice_focal import DiceFocalLoss
from training.losses.distance_transform import (
    LocalSensitiveDistanceLoss,
    soft_distance_transform,
    soft_erode,
)
from training.losses.general_union import GeneralUnionLoss
from training.losses.weights import apply_loss_mask, build_voxel_weights

__all__ = [
    "BreakageSensitiveLoss",
    "ConnectivityAwareLoss",
    "DeepSupervisionLoss",
    "DiceFocalLoss",
    "GeneralUnionLoss",
    "LocalSensitiveDistanceLoss",
    "apply_loss_mask",
    "build_voxel_weights",
    "build_loss",
    "soft_distance_transform",
    "soft_erode",
]


def build_loss(name: str, **kwargs) -> nn.Module:
    """Instantiate the loss a given model was trained with in the paper."""

    if name == "dice_focal":
        return DiceFocalLoss(**kwargs)
    if name == "connectivity_aware":
        return ConnectivityAwareLoss(use_lsd=False, **kwargs)
    if name == "connectivity_aware_lsd":
        return ConnectivityAwareLoss(use_lsd=True, **kwargs)
    if name == "general_union":
        return GeneralUnionLoss(**kwargs)
    if name == "breakage_sensitive":
        return BreakageSensitiveLoss(**kwargs)

    raise KeyError(
        f"unknown loss '{name}'; expected one of dice_focal, connectivity_aware, "
        f"connectivity_aware_lsd, general_union, breakage_sensitive")
