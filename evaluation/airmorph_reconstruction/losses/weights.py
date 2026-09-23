# -*- coding: utf-8 -*-
"""Per-voxel weight maps shared by the connectivity-aware and general-union losses.

The paper specifies *what* these weights depend on but not their functional form:

  CAL: "alpha_x defines the weight of each airway voxel based on the Euclidean
        distance to the centerline."
  GUL: reuses a per-voxel weight w_x / alpha_x in numerator and denominator.

We therefore derive both from the cached Euclidean distance-to-centreline map with an
exponential decay, which gives maximum weight on the centreline itself and falls off
toward the lumen wall. Thin peripheral branches - where the wall is close to the
centreline - stay highly weighted throughout, which is the behaviour both losses are
designed to produce. The exact decay is a documented deviation (see training/README.md).
"""

from typing import Optional

import torch


def build_voxel_weights(
    target: torch.Tensor,
    dist_to_centerline: Optional[torch.Tensor],
    tau: float = 3.0,
    fg_gain: float = 4.0,
    bg_weight: float = 1.0,
) -> torch.Tensor:
    """Build the per-voxel weight map alpha_x / w_x.

    Args:
        target: binary ground truth, shape [B, 1, D, H, W].
        dist_to_centerline: Euclidean distance (in voxels) from each voxel to the
            nearest airway centreline point, same shape as `target`. If None, a flat
            weight map is returned.
        tau: decay constant of the exponential falloff, in voxels.
        fg_gain: peak additional weight applied on the centreline.
        bg_weight: baseline weight, applied everywhere including background.

    Returns:
        Weight map with the same shape as `target`, minimum value `bg_weight`.
    """

    if dist_to_centerline is None:
        return torch.full_like(target, bg_weight)

    decay = torch.exp(-dist_to_centerline.clamp(min=0.0) / tau)
    # The distance-derived boost applies to airway voxels; background keeps the floor.
    return bg_weight + fg_gain * decay * target


def apply_loss_mask(
    prob: torch.Tensor,
    target: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
):
    """Restrict a loss to a region of interest.

    Used for the paper's stage-1 curriculum, where "lung region segmentation is used as
    a hard attention mechanism to suppress irrelevant areas and reduce false positives".
    Zeroing both prediction and target outside the lung removes those voxels from every
    sum-based term without biasing the ratio.
    """

    if loss_mask is None:
        return prob, target
    return prob * loss_mask, target * loss_mask
