# -*- coding: utf-8 -*-
"""Patch dataset over the preprocessed AirMorph cache.

Cases are memory-mapped, so a full 620-case cache never has to fit in RAM: each sample
reads only the 128x224x304 window it needs. One epoch is a fixed number of randomly
drawn patches (`iters_per_epoch`) rather than a pass over cases, which is the usual
arrangement for patch-based 3D segmentation and keeps epoch length independent of how
many cases are cached.
"""

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from training.data.sampling import PatchSampler

#: Arrays loaded per case; value is the dtype the training loop wants.
_ARRAYS = {
    "image": np.float32,
    "label": np.float32,
    "lung": np.float32,
    "centerline": np.float32,
    "dist_to_centerline": np.float32,
}


class CachedCase:
    """Lazily memory-mapped view of one preprocessed case."""

    def __init__(self, cache_dir: str, key: str, patch_size: Sequence[int], config: Dict):
        self.key = key
        self.path = os.path.join(cache_dir, key)
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"cache entry missing: {self.path}")

        with open(os.path.join(self.path, "meta.json"), "r") as handle:
            self.meta = json.load(handle)

        self.shape = tuple(self.meta["cropped_shape"])
        self.spacing = self.meta["spacing"]
        self._arrays: Dict[str, np.ndarray] = {}

        self.sampler = PatchSampler(
            volume_shape=self.shape,
            patch_size=patch_size,
            centerline_points=np.load(
                os.path.join(self.path, "centerline_points.npy")),
            centerline_radius=np.load(
                os.path.join(self.path, "centerline_radius.npy")),
            airway_centred_ratio=config["airway_centred_ratio"],
            peripheral_weight_exponent=config["peripheral_weight_exponent"],
            centre_jitter=config["centre_jitter"],
        )

    def array(self, name: str) -> np.ndarray:
        """Memory-mapped array, opened on first use (per worker process)."""

        if name not in self._arrays:
            self._arrays[name] = np.load(
                os.path.join(self.path, f"{name}.npy"), mmap_mode="r")
        return self._arrays[name]


def _extract_patch(
    volume: np.ndarray,
    origin: Sequence[int],
    patch_size: Sequence[int],
    dtype,
) -> np.ndarray:
    """Crop a patch, zero-padding when the volume is smaller than the patch."""

    slices = tuple(
        slice(o, min(o + size, extent))
        for o, size, extent in zip(origin, patch_size, volume.shape))
    patch = np.asarray(volume[slices], dtype=dtype)

    if patch.shape != tuple(patch_size):
        padded = np.zeros(tuple(patch_size), dtype=dtype)
        padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        patch = padded

    return patch


class AirwayPatchDataset(Dataset):
    """Random patches drawn from the cached ATM26 cases."""

    def __init__(
        self,
        config: Dict,
        split: str = "train",
        augment: bool = True,
        length: Optional[int] = None,
    ):
        self.config = config
        self.split = split
        self.patch_size = tuple(config["patch_size"])
        self.length = length if length is not None else config["iters_per_epoch"]

        with open(config["split_path"], "r") as handle:
            splits = json.load(handle)
        if split not in splits:
            raise KeyError(f"split '{split}' not in {config['split_path']}")

        self.keys: List[str] = splits[split]
        if not self.keys:
            raise ValueError(f"split '{split}' is empty")

        self.cases: Dict[str, CachedCase] = {}
        for key in self.keys:
            self.cases[key] = CachedCase(
                config["cache_dir"], key, self.patch_size, config)

        self.augmentor = None
        if augment:
            from training.data.augment import AirwayAugmentor
            self.augmentor = AirwayAugmentor(
                flip_prob=config["aug_flip_prob"],
                rotate_prob=config["aug_rotate_prob"],
                rotate_degrees=config["aug_rotate_degrees"],
                seed=config["seed"],
            )

        # Stage-2 curriculum state, controlled by the trainer.
        self.hard_ratio = 0.0

    def __len__(self) -> int:
        return self.length

    def set_hard_ratio(self, ratio: float) -> None:
        self.hard_ratio = ratio

    def set_hard_centres(self, key: str, centres: Optional[np.ndarray]) -> None:
        self.cases[key].sampler.set_hard_centres(centres)

    def _rng(self, index: int) -> np.random.Generator:
        """Per-sample generator that differs across workers, epochs and ranks."""

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = (self.config["seed"], worker_id, index, torch.initial_seed() % (2 ** 31))
        return np.random.default_rng(seed)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rng = self._rng(index)

        key = self.keys[rng.integers(0, len(self.keys))]
        case = self.cases[key]
        origin = case.sampler.sample(rng, hard_ratio=self.hard_ratio)

        sample: Dict[str, torch.Tensor] = {}
        for name, dtype in _ARRAYS.items():
            patch = _extract_patch(case.array(name), origin, self.patch_size, dtype)
            # [1, D, H, W] - MONAI's transforms expect a leading channel dimension.
            sample[name] = torch.from_numpy(patch[None])

        if self.augmentor is not None:
            sample = self.augmentor(sample)

        # Binary maps can pick up interpolation noise; re-threshold defensively.
        for name in ("label", "lung", "centerline"):
            sample[name] = (sample[name] > 0.5).float()

        sample["key"] = key
        sample["origin"] = torch.as_tensor(origin, dtype=torch.long)
        return sample
