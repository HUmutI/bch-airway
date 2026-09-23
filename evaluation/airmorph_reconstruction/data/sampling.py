# -*- coding: utf-8 -*-
"""Patch-centre sampling for AirMorph segmentation training.

The paper describes the sampling strategy qualitatively:

    "dense cropping was applied around airway-centered regions"
    "Additional supervision is introduced through label-informed sampling to improve
     intra-class discrimination. In particular, peripheral airway branches are emphasized
     via over-sampling strategies guided by the spatial distribution of centerline points."

Neither the airway/background patch ratio nor the peripheral weighting function is given,
so both are implemented as configurable and recorded as deviations.

Two effects combine to emphasise the periphery:

1. Drawing patch centres uniformly from *centreline points* already over-samples distal
   airways, because the great majority of centreline points lie in small branches.
2. On top of that, each centreline point is weighted by `radius^(-exponent)`. The local
   airway radius is the cached distance from that skeleton point to the airway wall, so
   thin peripheral branches receive proportionally more patches than the trachea.

A configurable fraction of patches is still drawn from the wider lung region, so the model
keeps seeing negative context and does not drift toward over-segmentation.
"""

from typing import Optional, Sequence, Tuple

import numpy as np


class PatchSampler:
    """Choose patch origins for one cached case."""

    def __init__(
        self,
        volume_shape: Sequence[int],
        patch_size: Sequence[int],
        centerline_points: np.ndarray,
        centerline_radius: np.ndarray,
        airway_centred_ratio: float = 0.7,
        peripheral_weight_exponent: float = 1.0,
        centre_jitter: int = 16,
        min_radius_mm: float = 0.3,
    ):
        self.volume_shape = tuple(int(s) for s in volume_shape)
        self.patch_size = tuple(int(s) for s in patch_size)
        self.airway_centred_ratio = airway_centred_ratio
        self.centre_jitter = centre_jitter

        self.points = centerline_points
        if self.points is not None and self.points.shape[0] > 0:
            radius = np.asarray(centerline_radius, dtype=np.float64)
            radius = np.maximum(radius, min_radius_mm)
            weights = radius ** (-float(peripheral_weight_exponent))
            total = weights.sum()
            self.weights = weights / total if total > 0 else None
        else:
            self.points = None
            self.weights = None

        # Hard-patch pool for the paper's stage-2 curriculum; filled by the trainer.
        self.hard_centres: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ internals

    def _clamp_origin(self, centre: Sequence[int]) -> Tuple[int, ...]:
        """Convert a desired centre into a valid in-bounds patch origin."""

        origin = []
        for axis, (c, size, extent) in enumerate(
                zip(centre, self.patch_size, self.volume_shape)):
            start = int(round(c)) - size // 2
            start = max(0, min(start, extent - size))
            if extent < size:
                # Volume smaller than the patch on this axis; the dataset pads afterwards.
                start = 0
            origin.append(start)
        return tuple(origin)

    def _jitter(self, point: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.centre_jitter <= 0:
            return point
        offset = rng.integers(-self.centre_jitter, self.centre_jitter + 1, size=3)
        return point + offset

    # --------------------------------------------------------------------- public

    def set_hard_centres(self, centres: Optional[np.ndarray]) -> None:
        """Install the stage-2 hard-patch pool (array of [N, 3] voxel centres)."""

        self.hard_centres = centres if centres is not None and len(centres) else None

    def sample(
        self,
        rng: np.random.Generator,
        hard_ratio: float = 0.0,
    ) -> Tuple[int, ...]:
        """Return a patch origin (start index per axis)."""

        if (self.hard_centres is not None and hard_ratio > 0
                and rng.random() < hard_ratio):
            index = rng.integers(0, len(self.hard_centres))
            centre = self._jitter(np.asarray(self.hard_centres[index]), rng)
            return self._clamp_origin(centre)

        if self.points is not None and rng.random() < self.airway_centred_ratio:
            index = rng.choice(len(self.points), p=self.weights)
            centre = self._jitter(self.points[index].astype(np.int64), rng)
            return self._clamp_origin(centre)

        # Fall back to a uniform position over the (already lung-cropped) volume.
        centre = np.array([
            rng.integers(0, max(1, extent)) for extent in self.volume_shape])
        return self._clamp_origin(centre)
