# -*- coding: utf-8 -*-
"""Differentiable distance transform used by the CAL + local-sensitive-distance loss.

The paper defers this entirely: "Detailed calculation of the differentiable distance
transform Dist can be referred to [22]." Reference [22] is not reproduced here, so we
implement the standard differentiable surrogate: iterated soft erosion.

A voxel's distance to the background boundary equals the number of erosion steps it
survives. Summing successive soft erosions therefore approximates that distance while
staying differentiable, since soft erosion is a min-pool implemented as
`-max_pool3d(-x)`. This is the same construction used by soft-skeleton / clDice work.

Documented deviation: an approximation of an unspecified operator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """3D soft erosion: a min filter over the 3x3x3 neighbourhood."""

    return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)


def soft_distance_transform(x: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """Approximate the distance from each voxel to the nearest background voxel.

    Args:
        x: soft occupancy map in [0, 1], shape [B, 1, D, H, W].
        iterations: maximum distance representable, in voxels. Airway lumens are thin,
            so a small number of steps covers the structure of interest.

    Returns:
        Non-negative map of the same shape; roughly `x` plus one unit per erosion the
        voxel survives.
    """

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    eroded = x
    distance = x
    for _ in range(iterations):
        eroded = soft_erode(eroded)
        distance = distance + eroded
    return distance


class LocalSensitiveDistanceLoss(nn.Module):
    """The `|| Dist(g_x) - Dist(p_x) ||_2` term of AirMorph model 3."""

    def __init__(self, iterations: int = 5, normalize: bool = True):
        super().__init__()
        self.iterations = iterations
        self.normalize = normalize

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist_pred = soft_distance_transform(prob, self.iterations)
        dist_true = soft_distance_transform(target, self.iterations)

        diff = dist_true - dist_pred
        # L2 norm as printed. Normalising by voxel count keeps the term's scale
        # comparable to L_CAL across different patch sizes.
        norm = torch.sqrt((diff ** 2).sum() + 1e-8)
        if self.normalize:
            norm = norm / (diff.numel() ** 0.5)
        return norm
