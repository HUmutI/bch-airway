# -*- coding: utf-8 -*-
"""One-off preprocessing / caching for the AirMorph anatomical-labeling stage.

For each of the 118 ATM26 Task-2 cases this builds the graph representation the labeling
transformer consumes and the per-node ground-truth targets, then caches them as a small
`.npz` so training needs neither pycuda nor a skeletonisation pass.

The graph is built from the **ground-truth binary airway** (ATM26 Task-1 mask) exactly as
the deployed pipeline builds it from the predicted airway (`classifier/airway_classifier.py`),
so the trained model sees the same feature/topology distribution at train and inference:

    binary airway (GT)  -> MPC-Skel skeleton (kimimaro TEASAR)
    -> tree_parse (branch parcellation, parent/child maps, generation, trachea)
    -> feature_extraction_cuda (11 node features, edges)     [pycuda + GPU]
    -> multitask_dataset (x, spd, mask_top)                  [torch_geometric Data]
    -> per-node GT via majority raw code over each parsed branch  (label_transfer)

Per case the cache holds:
    x        float32 [N, 11]   node features (the 11 of Table 3)
    spd      int64   [N, N]    Floyd shortest-path distance (clamped 29)
    mask_top int64   [N, N]    ancestor->descendant topology mask
    y_lob    int64   [N]       lobar target      (ignore_index for unresolved)
    y_seg    int64   [N]       segmental target
    y_sub    int64   [N]       subsegmental target
    node_idx int32   [N]       node -> branch id (for painting predictions back)
    meta.json                  case id, N, label-transfer stats, spacing/shape

Usage:
    python -m training.data.prepare_labeling                 # all 118 cases + splits
    python -m training.data.prepare_labeling --limit 2       # smoke test
    python -m training.data.prepare_labeling --inspect ATM_001   # print label diagnostics
"""

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List, Optional

import numpy as np

# The graph build (data_process/tree_parse_skelbased.py, feature_cuda.py) JIT-compiles CUDA
# kernels via pycuda->nvcc. CUDA 12.3's nvcc rejects host gcc > 12, but this box ships gcc
# 13 with no g++-12. `-allow-unsupported-compiler` lets nvcc use g++-13: it only affects the
# host stub (kernel launch / memcpy); device code is compiled by nvcc's own frontend and is
# unaffected by the host gcc version, so the trivial integer/atomic kernels here are safe.
# Set before pycuda is imported (it reads NVCC_PREPEND_FLAGS at compile time). Append so a
# user-provided flag (e.g. `-ccbin g++-12`) is preserved and wins.
os.environ["NVCC_PREPEND_FLAGS"] = (
    os.environ.get("NVCC_PREPEND_FLAGS", "") + " -allow-unsupported-compiler").strip()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from training.configs.train_config_labeling import get_config           # noqa: E402
from training.data.label_transfer import (                              # noqa: E402
    build_segment_maps,
    transfer_node_labels,
)
from training.data.labeling_splits import write_labeling_splits         # noqa: E402
from training.data.prepare_dataset import _strip_case_id, _IMAGE_SUFFIXES  # noqa: E402
from util.utils import load_itk_image                                   # noqa: E402

CACHE_FILES = ("graph.npz", "meta.json")


# --------------------------------------------------------------------------- discovery

def _index(directory: str) -> Dict[str, str]:
    found = {}
    if not os.path.isdir(directory):
        return found
    for name in sorted(os.listdir(directory)):
        if name.endswith(_IMAGE_SUFFIXES):
            found[_strip_case_id(name)] = os.path.join(directory, name)
    return found


def discover_labeling_cases(config: Dict) -> List[Dict[str, str]]:
    """Cases that have an image, a binary airway mask, AND a branch-label volume."""

    images = _index(config["seg_images"])
    binaries = _index(config["seg_labels"])
    branches = _index(config["label_branch_dir"])

    cases = []
    for cid in sorted(branches):
        if cid not in images:
            print(f"  [skip] {cid}: no image")
            continue
        if cid not in binaries:
            print(f"  [skip] {cid}: no binary airway mask")
            continue
        cases.append({
            "case_id": cid,
            "image": images[cid],
            "binary": binaries[cid],
            "branch": branches[cid],
        })
    return cases


# ------------------------------------------------------------------------ graph build

def build_graph(case: Dict[str, str], lung_extractor, segment_maps, config: Dict) -> Dict:
    """Build the graph tensors + per-node targets for one case."""

    # Lazily imported: they pull in pycuda / torch_geometric, needed only here.
    from data_process.tree_parse_skelbased import tree_parse
    from data_process.feature_cuda import feature_extraction_cuda
    from data_process.dataset import multitask_dataset
    from util.airwayatlas_utils import IMRAirwayAtlas_SkeletonExtraction

    image, _, spacing, _ = load_itk_image(case["image"])
    bin_airway, _, _, _ = load_itk_image(case["binary"])
    branch_label, _, _, _ = load_itk_image(case["branch"])

    bin_airway = (bin_airway > 0).astype(np.uint8)
    if bin_airway.shape != branch_label.shape:
        raise ValueError(
            f"binary {bin_airway.shape} != branch label {branch_label.shape}")
    if image.shape != bin_airway.shape:
        raise ValueError(f"image {image.shape} != binary {bin_airway.shape}")

    # Lung lobe mask (values 1=right, 2=left), full frame, same as the deployed pipeline.
    lobe = lung_extractor.predict(image)

    # Skeleton + tree parse on the GROUND-TRUTH binary airway.
    skeleton = IMRAirwayAtlas_SkeletonExtraction(bin_airway, np.array(spacing))
    if skeleton is None:
        raise RuntimeError("skeletonisation returned None")
    skeleton = skeleton.astype(np.uint8)

    parent_map, children_map, generation, trachea_ind, tree_parsing_skel, airway_parse = \
        tree_parse(bin_airway.copy(), skeleton.copy())

    # Replicate classifier/airway_classifier.py's exact orientation convention so the
    # feature distribution matches inference (the flips negate positional features; the
    # lobe .T only affects the unused left/right column).
    skel_for_feature = tree_parsing_skel.copy()[::-1, ::-1, ::-1]
    lobe_for_feature = lobe.copy().T[::-1, ::-1, ::-1]

    data_final, edge, edge_feature, node_idx, _ = feature_extraction_cuda(
        skel_for_feature, list(np.array(spacing)[::-1]), lobe_for_feature,
        parent_map, children_map, generation, trachea_ind)

    # Graph tensors (x[:, 0:11], spd, mask_top) exactly as inference builds them.
    data = multitask_dataset(case["case_id"], data_final, edge, edge_feature)
    x = data.x.numpy().astype(np.float32)
    spd = data.spd.numpy().astype(np.int64)
    mask_top = data.mask_top.numpy().astype(np.int64)

    # Per-node ground-truth targets from the segmental branch-label volume.
    # tree_parse transposes [D,H,W]->[W,H,D] internally (tree_parse_skelbased.py:220), so
    # airway_parse is in [x,y,z]; the branch label loaded as [z,y,x] must be transposed the
    # same way to share the voxel grid. Node identity itself is orientation-independent.
    branch_label_xyz = np.ascontiguousarray(branch_label.T)
    trachea_node = int(np.argmax(data_final[:, 13]))  # == get_trachea_index(x): max volume
    y_lob, y_seg, y_sub, stats = transfer_node_labels(
        airway_parse, node_idx, branch_label_xyz, segment_maps,
        ignore_index=config["ignore_index"])

    if not (x.shape[0] == spd.shape[0] == mask_top.shape[0]
            == y_lob.shape[0] == node_idx.shape[0]):
        raise ValueError("node count mismatch across tensors/targets")

    return {
        "x": x,
        "spd": spd,
        "mask_top": mask_top,
        "y_lob": y_lob,
        "y_seg": y_seg,
        "y_sub": y_sub,
        "node_idx": node_idx.astype(np.int32),
        "meta": {
            "case_id": case["case_id"],
            "num_nodes": int(x.shape[0]),
            "spacing": list(map(float, spacing)),
            "shape": list(map(int, bin_airway.shape)),
            "trachea_node": trachea_node,
            "branch_label_values": int(np.unique(branch_label).size),
            "label_stats": stats,
        },
    }


def process_case(case, cache_dir, lung_extractor, segment_maps, config, overwrite=False):
    case_id = case["case_id"]
    out_dir = os.path.join(cache_dir, config["dataset_name"], case_id)

    if not overwrite and all(
            os.path.exists(os.path.join(out_dir, name)) for name in CACHE_FILES):
        print(f"  [cached] {case_id}")
        with open(os.path.join(out_dir, "meta.json"), "r") as handle:
            return json.load(handle)

    os.makedirs(out_dir, exist_ok=True)
    graph = build_graph(case, lung_extractor, segment_maps, config)

    np.savez_compressed(
        os.path.join(out_dir, "graph.npz"),
        x=graph["x"], spd=graph["spd"], mask_top=graph["mask_top"],
        y_lob=graph["y_lob"], y_seg=graph["y_seg"], y_sub=graph["y_sub"],
        node_idx=graph["node_idx"])
    with open(os.path.join(out_dir, "meta.json"), "w") as handle:
        json.dump(graph["meta"], handle, indent=2)

    stats = graph["meta"]["label_stats"]
    print(f"  [done] {case_id}  N={graph['meta']['num_nodes']} "
          f"resolved={stats['resolved']} background={stats['background']} "
          f"unknown={stats['unknown']} empty={stats['empty']}")
    return graph["meta"]


# ------------------------------------------------------------------------- inspection

def inspect_case(case: Dict[str, str], lung_extractor, segment_maps, config: Dict) -> None:
    """Print label diagnostics for one case to validate the segmental-scheme assumptions."""

    branch_label, _, _, _ = load_itk_image(case["branch"])
    values = np.unique(branch_label)
    print(f"\n=== {case['case_id']} branch-label volume ===")
    print(f"  unique values ({values.size}): {values.astype(int).tolist()[:60]}")
    print(f"  not in segment table (excl 0): "
          f"{sorted(set(int(v) for v in values) - {0} - set(segment_maps))}")

    graph = build_graph(case, lung_extractor, segment_maps, config)
    x = graph["x"]
    gens = x[:, 0]  # generation feature (raw, col 0)
    y_seg = graph["y_seg"]
    unresolved = np.where(graph["y_seg"] == config["ignore_index"])[0]
    print(f"  N nodes={x.shape[0]}  resolved={graph['meta']['label_stats']['resolved']} "
          f"background={graph['meta']['label_stats']['background']}")
    print(f"  unresolved nodes={unresolved.size} "
          f"(generation range: "
          f"{gens[unresolved].min() if unresolved.size else '-'}.."
          f"{gens[unresolved].max() if unresolved.size else '-'})")
    tn = graph["meta"]["trachea_node"]
    print(f"  trachea node gen={gens[tn]} y_seg={y_seg[tn]} (expect 18=trachea)")


# -------------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--inspect", nargs="*", default=None,
                        help="print label diagnostics for these case ids and exit")
    args = parser.parse_args()

    config = get_config()
    cache_dir = config["label_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    all_cases = discover_labeling_cases(config)
    print(f"Discovered {len(all_cases)} labeling cases")
    if not all_cases:
        print("No labeling cases found. Check paths in train_config_base.py "
              "(seg_images / seg_labels / label_branch_dir).")
        return 1

    segment_maps = build_segment_maps(config["subseg_policy"], config["ignore_index"])
    print(f"subseg_policy={config['subseg_policy']!r} "
          f"(subseg head {'supervised at segment granularity' if config['subseg_policy']=='base_segment' else 'left unsupervised'})")

    from models.lungmask_model import LungMaskExtractionModel
    lung_extractor = LungMaskExtractionModel()

    if args.inspect is not None:
        wanted = set(args.inspect) if args.inspect else {all_cases[0]["case_id"]}
        for case in all_cases:
            if case["case_id"] in wanted:
                inspect_case(case, lung_extractor, segment_maps, config)
        return 0

    cases = all_cases
    if args.cases:
        cases = [c for c in cases if c["case_id"] in set(args.cases)]
    if args.limit:
        cases = cases[: args.limit]

    metas: List[Dict] = []
    failures = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['case_id']}")
        try:
            meta = process_case(case, cache_dir, lung_extractor, segment_maps, config,
                                 overwrite=args.overwrite)
            if meta is not None:
                metas.append(meta)
        except Exception as exc:
            failures.append((case["case_id"], f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()

    print(f"\nProcessed {len(metas)} cases, {len(failures)} failures")
    for case_id, message in failures:
        print(f"  FAILED {case_id}: {message}")

    if not args.skip_splits:
        # Inherit the seg partition over ALL discovered labeling ids (independent of how
        # many were processed this run), so the split is stable.
        write_labeling_splits(config, [c["case_id"] for c in all_cases])

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
