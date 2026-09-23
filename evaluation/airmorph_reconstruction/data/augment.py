# -*- coding: utf-8 -*-
"""Augmentation for AirMorph segmentation training.

The paper lists exactly two: "random horizontal flipping" and "random rotation in
[-10 degrees, +10 degrees]". Nothing else is applied - no intensity jitter, no scaling,
no elastic deformation - because the paper specifies none and the stated priority is
fidelity.

Every array in a sample is transformed with the same random parameters, since the label,
lung mask, centreline and distance map must stay voxel-aligned with the image. The image
is interpolated bilinearly; all other channels use nearest-neighbour so binary maps stay
binary. The distance map is a special case: rotating it nearest-neighbour keeps its
values approximately correct, which is sufficient for a loss weight.
"""

from typing import Dict

import numpy as np
import torch
from monai.transforms import Flip, Rotate


class AirwayAugmentor:
    """Paper-specified flip + small-angle rotation, applied consistently to a sample."""

    #: Keys interpolated with nearest neighbour (labels and derived maps).
    NEAREST_KEYS = ("label", "lung", "centerline", "dist_to_centerline")

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        rotate_degrees: float = 10.0,
        # "horizontal flipping" in an axial CT is the left-right axis, which is the last
        # axis of the [D, H, W] arrays this pipeline uses.
        flip_axis: int = 2,
        seed: int = 0,
    ):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_radians = np.deg2rad(rotate_degrees)
        self.flip_axis = flip_axis
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(sample)

        if self.rng.random() < self.flip_prob:
            flip = Flip(spatial_axis=self.flip_axis)
            for key in ("image",) + self.NEAREST_KEYS:
                if key in out:
                    out[key] = flip(out[key])

        if self.rng.random() < self.rotate_prob:
            # Rotation in the axial plane only, matching "rotation in [-10, +10] degrees"
            # for a chest CT; out-of-plane rotation is not implied by the paper.
            angle = float(self.rng.uniform(-self.rotate_radians, self.rotate_radians))
            angles = (angle, 0.0, 0.0)

            if "image" in out:
                out["image"] = Rotate(
                    angle=angles, keep_size=True, mode="bilinear",
                    padding_mode="zeros")(out["image"])
            for key in self.NEAREST_KEYS:
                if key in out:
                    out[key] = Rotate(
                        angle=angles, keep_size=True, mode="nearest",
                        padding_mode="zeros")(out[key])

        return out
