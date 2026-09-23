# -*- coding: utf-8 -*-
"""One-off preprocessing / caching for AirMorph segmentation training.

For every ATM26 case this writes a memory-mappable cache directory holding
everything the training losses and the patch sampler need, so that no skeletonisation or
lung masking happens in the training loop.

Preprocessing deliberately mirrors the inference path exactly
(`segmentator/airway_segmentator.py` -> `models/airway_model.py`):

    lung mask (lungmask R231) -> lung bounding box with margin 5
    -> crop CT to that box -> lumTrans (HU window [-1200, 600]) -> normalize_CT

Any divergence here would mean the trained weights see a different input distribution at
test time, so the same provided functions are called rather than reimplemented.

Per case the cache contains:

    image.npy               float16 [D, H, W]  windowed + normalised CT, lung-box cropped
    label.npy               uint8   [D, H, W]  binary airway ground truth, same crop
    lung.npy                uint8   [D, H, W]  lung mask, same crop (stage-1 attention)
    centerline.npy          uint8   [D, H, W]  TEASAR skeleton of the ground truth
    dist_to_centerline.npy  float16 [D, H, W]  mm distance to the nearest skeleton voxel
    centerline_points.npy   int32   [N, 3]     skeleton voxel coordinates
    centerline_radius.npy   float32 [N]        mm distance from each skeleton point to
                                               the airway wall - the peripheral-branch
                                               proxy used by the sampler
    meta.json                                  spacing/origin/direction, bbox, shapes

Usage:
    python -m training.data.prepare_dataset                       # all ATM26 cases
    python -m training.data.prepare_dataset --cases ATM_001 ATM_002   # smoke test
    python -m training.data.prepare_dataset --limit 2                 # first 2 cases
"""

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from networks.airway_network import lumTrans, normalize_CT          # noqa: E402
from training.configs.train_config_base import config as base_config  # noqa: E402
from util.airwayatlas_utils import IMRAirwayAtlas_SkeletonExtraction  # noqa: E402
from util.utils import crop_image_via_box, load_itk_image            # noqa: E402

CACHE_FILES = (
    "image.npy",
    "label.npy",
    "lung.npy",
    "centerline.npy",
    "dist_to_centerline.npy",
    "centerline_points.npy",
    "centerline_radius.npy",
    "meta.json",
)


# --------------------------------------------------------------------------- discovery

_IMAGE_SUFFIXES = (".mha", ".nii.gz", ".nii")


def _strip_case_id(filename: str) -> str:
    """Reduce an image/label filename to a dataset-unique case id.

    Handles both the `.mha` (ATM26) and `.nii[.gz]` layouts, and the nnU-Net style
    `_0000` modality suffix that images carry but their labels do not, so images and
    labels pair up by id (e.g. `ATM_001_0000.mha` and `ATM_001.mha` -> `ATM_001`).
    """

    stem = filename
    for suffix in _IMAGE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.endswith("_0000"):
        stem = stem[: -len("_0000")]
    return stem


def discover_cases(images_dir: str, labels_dir: str, dataset: str) -> List[Dict[str, str]]:
    """Pair image and label files by case id."""

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")

    def index(directory: str) -> Dict[str, str]:
        found = {}
        for name in sorted(os.listdir(directory)):
            if not name.endswith(_IMAGE_SUFFIXES):
                continue
            found[_strip_case_id(name)] = os.path.join(directory, name)
        return found

    images = index(images_dir)
    labels = index(labels_dir)

    cases = []
    for case_id in sorted(images):
        if case_id not in labels:
            print(f"  [skip] {case_id}: no matching label")
            continue
        cases.append({
            "case_id": case_id,
            "dataset": dataset,
            "image": images[case_id],
            "label": labels[case_id],
        })

    missing = sorted(set(labels) - set(images))
    if missing:
        print(f"  [warn] {len(missing)} labels without an image, e.g. {missing[:3]}")

    return cases


# ------------------------------------------------------------------------ preprocessing

def _distance_maps(
    label: np.ndarray,
    skeleton: np.ndarray,
    spacing: Tuple[float, float, float],
):
    """Distance to the centreline, and airway radius sampled at each skeleton point."""

    from scipy import ndimage

    sampling = tuple(float(s) for s in spacing)

    # Distance from every voxel to the nearest skeleton voxel, in mm.
    if skeleton.any():
        dist_to_centerline = ndimage.distance_transform_edt(
            skeleton == 0, sampling=sampling).astype(np.float16)
    else:
        dist_to_centerline = np.zeros(label.shape, dtype=np.float16)

    # Distance from every airway voxel to the airway wall, in mm. Sampled on the
    # skeleton this is the local radius, which is small on peripheral branches.
    radius_field = ndimage.distance_transform_edt(label > 0, sampling=sampling)

    points = np.argwhere(skeleton > 0).astype(np.int32)
    if points.size:
        radii = radius_field[points[:, 0], points[:, 1], points[:, 2]].astype(np.float32)
    else:
        radii = np.zeros((0,), dtype=np.float32)

    return dist_to_centerline, points, radii


def process_case(
    case: Dict[str, str],
    cache_dir: str,
    lung_extractor,
    overwrite: bool = False,
) -> Optional[Dict]:
    """Preprocess one case and write its cache directory."""

    case_id = case["case_id"]
    out_dir = os.path.join(cache_dir, case["dataset"], case_id)

    if not overwrite and all(
            os.path.exists(os.path.join(out_dir, name)) for name in CACHE_FILES):
        print(f"  [cached] {case_id}")
        with open(os.path.join(out_dir, "meta.json"), "r") as handle:
            return json.load(handle)

    os.makedirs(out_dir, exist_ok=True)

    image, origin, spacing, direction = load_itk_image(case["image"])
    label, _, label_spacing, _ = load_itk_image(case["label"])

    if image.shape != label.shape:
        raise ValueError(
            f"{case_id}: image shape {image.shape} != label shape {label.shape}")

    # --- lung mask and bounding box, exactly as inference computes them
    lung = lung_extractor.predict(image)
    bbox = lung_extractor.lung_boundingbox

    image_crop = crop_image_via_box(image, bbox)
    label_crop = (crop_image_via_box(label, bbox) > 0).astype(np.uint8)
    lung_crop = (crop_image_via_box(lung, bbox) > 0).astype(np.uint8)

    # --- intensity pipeline, identical to models/airway_model.py::predict
    image_crop = lumTrans(image_crop)
    image_crop = normalize_CT(image_crop).astype(np.float16)

    # --- centreline of the ground truth
    skeleton = IMRAirwayAtlas_SkeletonExtraction(label_crop, np.array(spacing))
    if skeleton is None:
        raise RuntimeError(
            f"{case_id}: skeletonisation failed (IMRAirwayAtlas_SkeletonExtraction "
            f"swallowed the exception and returned None)")
    skeleton = skeleton.astype(np.uint8)

    dist_to_centerline, points, radii = _distance_maps(label_crop, skeleton, spacing)

    meta = {
        "case_id": case_id,
        "dataset": case["dataset"],
        "image_path": case["image"],
        "label_path": case["label"],
        "origin": list(map(float, origin)),
        "spacing": list(map(float, spacing)),
        "direction": list(map(float, direction)),
        "original_shape": list(map(int, image.shape)),
        "bbox": bbox.astype(int).tolist(),
        "cropped_shape": list(map(int, image_crop.shape)),
        "airway_voxels": int(label_crop.sum()),
        "centerline_points": int(points.shape[0]),
        "label_spacing": list(map(float, label_spacing)),
    }

    np.save(os.path.join(out_dir, "image.npy"), image_crop)
    np.save(os.path.join(out_dir, "label.npy"), label_crop)
    np.save(os.path.join(out_dir, "lung.npy"), lung_crop)
    np.save(os.path.join(out_dir, "centerline.npy"), skeleton)
    np.save(os.path.join(out_dir, "dist_to_centerline.npy"), dist_to_centerline)
    np.save(os.path.join(out_dir, "centerline_points.npy"), points)
    np.save(os.path.join(out_dir, "centerline_radius.npy"), radii)
    with open(os.path.join(out_dir, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)

    print(f"  [done] {case_id}  shape={image_crop.shape} "
          f"airway={meta['airway_voxels']} skel={meta['centerline_points']}")
    return meta


# ------------------------------------------------------------------------------ splits

def build_splits(metas: List[Dict], config: Dict) -> Dict[str, List[str]]:
    """Seeded train/val/test split, stratified by dataset.

    The paper never states its split sizes, so this is a documented deviation. `metas` may
    be either processed-case metadata or raw discovered-case records; only "dataset" and
    "case_id" are read, so the split can be built before preprocessing. With a single
    dataset (ATM26) there is one stratum; the stratification generalises if more are added.
    The labeling stage inherits this exact partition (see data/labeling_splits.py).
    """

    rng = np.random.default_rng(config["split_seed"])
    train_frac, val_frac, _ = config["split_fractions"]

    splits = {"train": [], "val": [], "test": []}
    by_dataset: Dict[str, List[str]] = {}
    for meta in metas:
        by_dataset.setdefault(meta["dataset"], []).append(meta["case_id"])

    for dataset, case_ids in sorted(by_dataset.items()):
        case_ids = sorted(case_ids)
        rng.shuffle(case_ids)
        n = len(case_ids)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        splits["train"] += [f"{dataset}/{c}" for c in case_ids[:n_train]]
        splits["val"] += [f"{dataset}/{c}" for c in case_ids[n_train:n_train + n_val]]
        splits["test"] += [f"{dataset}/{c}" for c in case_ids[n_train + n_val:]]

    return splits


# -------------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=None,
                        help="restrict to these case ids (smoke testing)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N cases")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-splits", action="store_true",
                        help="do not rewrite the split manifest")
    args = parser.parse_args()

    config = base_config
    cache_dir = config["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    dataset = config["dataset_name"]
    images_dir, labels_dir = config["seg_images"], config["seg_labels"]
    print(f"Scanning {dataset}: {images_dir}")
    all_cases = discover_cases(images_dir, labels_dir, dataset)
    # The split is built over every discovered case (below), independent of preprocessing
    # success, so the labeling stage can inherit a stable partition. Subsetting for a run
    # happens after, so --limit / --cases never change the split.
    split_cases = list(all_cases)
    if args.cases:
        all_cases = [c for c in all_cases if c["case_id"] in set(args.cases)]
    if args.limit:
        all_cases = all_cases[: args.limit]
    print(f"  {len(all_cases)} cases to process ({len(split_cases)} for the split)")

    if not all_cases:
        print("No cases found. Check the dataset paths in "
              "training/configs/train_config_base.py")
        return 1

    # Imported lazily so that --help works without torch/lungmask installed.
    from models.lungmask_model import LungMaskExtractionModel
    lung_extractor = LungMaskExtractionModel()

    metas: List[Dict] = []
    failures: List[Tuple[str, str]] = []
    for index, case in enumerate(all_cases, start=1):
        print(f"[{index}/{len(all_cases)}] {case['dataset']}/{case['case_id']}")
        try:
            meta = process_case(case, cache_dir, lung_extractor,
                                overwrite=args.overwrite)
            if meta is not None:
                metas.append(meta)
        except Exception as exc:  # keep going; a few cases failing is normal
            failures.append((case["case_id"], f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()

    print(f"\nProcessed {len(metas)} cases, {len(failures)} failures")
    for case_id, message in failures:
        print(f"  FAILED {case_id}: {message}")

    if not args.skip_splits:
        # Build over *all* discovered cases so the partition is stable regardless of how
        # many were processed this run, and so the labeling stage inherits it faithfully.
        splits = build_splits(split_cases, config)
        os.makedirs(os.path.dirname(config["split_path"]), exist_ok=True)
        with open(config["split_path"], "w") as handle:
            json.dump(splits, handle, indent=2)
        print(f"\nSplits written to {config['split_path']}: "
              f"train={len(splits['train'])} val={len(splits['val'])} "
              f"test={len(splits['test'])}")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
