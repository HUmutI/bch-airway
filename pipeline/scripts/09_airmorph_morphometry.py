#!/usr/bin/env python3
"""Label fused normal-airway masks and export physical branch morphometry.

The classification phase requires the AirMorph environment, CUDA and PyCUDA.
The export phase is CPU-only and can be rerun from saved classifier products.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


BASE = Path(os.environ.get("OR3_BASE", "/fileserver/external/quin/ege"))
AIRWAY_ROOT = BASE / "or_all3_run/code/AirMorph"
NORMAL_LIST = BASE / "airway-dataset/normal-cases.txt"
SEG_ROOT = BASE / "airway-seg"
DEFAULT_OUTPUT = BASE / "or_all3_run/normal_airway_branch_morphometry.xlsx"
CSV_OUTPUT = BASE / "or_all3_run/normal_airway_branch_morphometry.csv"
STATUS_OUTPUT = BASE / "or_all3_run/normal_airway_morphometry_status.csv"

FEATURE_COLUMNS = [
    "case_id", "subject", "age_group", "branch_instance_id", "parent_instance_id",
    "generation", "branch_name", "lobar_name", "segmental_name", "subsegmental_name",
    "centerline_length_mm", "chord_length_mm", "tortuosity_length_ratio",
    "radius_min_mm", "radius_mean_mm", "radius_max_mm", "radius_sd_mm",
    "diameter_min_mm", "diameter_mean_mm", "diameter_max_mm",
    "branch_angle_deg", "branch_angle_method", "branch_angle_status",
    "volume_mm3", "centerline_voxels", "branch_voxels", "centerline_components",
    "diameter_p10_mm", "diameter_median_mm", "diameter_p90_mm",
    "caliber_reduction_proxy", "caliber_expansion_proxy",
    "legacy_min_mean_caliber_index", "legacy_max_mean_caliber_index",
    "complexity_fractal_dimension",
    "complexity_status", "anatomy_review_status", "geometry_qc_flags",
    "source_mask", "classification_status",
]


def normal_cases() -> list[Path]:
    cases = []
    for line in NORMAL_LIST.read_text().splitlines():
        source = Path(line.strip())
        if not line.strip():
            continue
        rel = source.relative_to(BASE / "airway-dataset")
        case_id = source.name.removesuffix("_source.nii.gz")
        cases.append(SEG_ROOT / rel.parent / case_id)
    return cases


def case_fields(case_dir: Path) -> tuple[str, str, str]:
    return case_dir.name, case_dir.parent.name, case_dir.parents[1].name


def classifier_products(case_dir: Path) -> list[Path]:
    case_id = case_dir.name
    return [
        case_dir / f"{case_id}_skel_parsing.nii.gz",
        case_dir / f"{case_id}_parse.nii.gz",
        case_dir / f"{case_id}_airway_graph.npy",
        case_dir / f"{case_id}_airway_feature_cls.npy",
        case_dir / f"{case_id}_airway_graph_cls.npy",
        case_dir / f"{case_id}_anno.json",
    ]


def classify_case(case_dir: Path, force: bool = False) -> str:
    outputs = classifier_products(case_dir)
    if not force and all(p.exists() for p in outputs):
        return "skipped-complete"

    sys.path.insert(0, str(AIRWAY_ROOT))
    from classifier.airway_classifier import AirwayAtlasMultiAnatomyAirwayClassifier
    from util.airwayatlas_utils import IMRAirwayAtlas_SkeletonExtraction

    mask_path = case_dir / "union3_lcc.nii.gz"
    lobe_path = case_dir / "lunglobe.nii.gz"
    if not mask_path.exists() or not lobe_path.exists():
        raise FileNotFoundError(f"missing fused mask or lung lobe mask in {case_dir}")

    mask_img = sitk.ReadImage(str(mask_path))
    mask = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
    lobe = sitk.GetArrayFromImage(sitk.ReadImage(str(lobe_path))).astype(np.uint8)
    spacing_xyz = np.asarray(mask_img.GetSpacing(), dtype=float)
    skeleton_path = case_dir / "union3_lcc_skeleton.nii.gz"
    if skeleton_path.exists() and not force:
        skeleton = sitk.GetArrayFromImage(sitk.ReadImage(str(skeleton_path))).astype(np.uint8)
    else:
        skeleton = IMRAirwayAtlas_SkeletonExtraction(mask, spacing_xyz)
        if skeleton is None or not np.any(skeleton):
            raise RuntimeError("AirMorph skeleton extraction returned an empty result")
        # Keep this fused-mask skeleton explicit for provenance and later QC.
        skel_img = sitk.GetImageFromArray(skeleton.astype(np.uint8))
        skel_img.CopyInformation(mask_img)
        sitk.WriteImage(skel_img, str(skeleton_path), True)

    data = {
        "patient": case_dir.name,
        "file_path": str(case_dir),
        "bin_airway": mask,
        "lobe": lobe,
        "skeleton": skeleton,
        "itk_meta": {
            "origin": mask_img.GetOrigin(),
            "spacing": mask_img.GetSpacing(),
            "direction": mask_img.GetDirection(),
        },
    }
    AirwayAtlasMultiAnatomyAirwayClassifier()(data)
    return "classified"


NEIGHBOR_OFFSETS = [
    (dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    if (dz, dy, dx) > (0, 0, 0)
]


def centerline_geometry(coords: np.ndarray, spacing_zyx: np.ndarray):
    """Return physical MST geometry and a connectivity QC value."""
    n = len(coords)
    physical = coords.astype(float) * spacing_zyx
    if n < 2:
        point = physical[0] if n else np.full(3, np.nan)
        return 0.0, 0.0, point, point, max(n, 1)

    index = {tuple(c): i for i, c in enumerate(coords)}
    rows, cols, weights = [], [], []
    for i, c in enumerate(coords):
        for off in NEIGHBOR_OFFSETS:
            j = index.get(tuple(c + off))
            if j is not None:
                w = float(np.linalg.norm(np.asarray(off) * spacing_zyx))
                rows.extend((i, j)); cols.extend((j, i)); weights.extend((w, w))
    graph = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
    n_components = int(connected_components(graph, directed=False, return_labels=False))
    tree = minimum_spanning_tree(graph)
    length = float(tree.sum())
    degree = np.asarray((tree + tree.T).astype(bool).sum(axis=1)).ravel()
    endpoints = np.flatnonzero(degree <= 1)
    if len(endpoints) < 2:
        endpoints = np.arange(n)
    pts = physical[endpoints]
    # Branches are small enough that this endpoint distance matrix is modest.
    distances = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    a, b = np.unravel_index(np.argmax(distances), distances.shape)
    return length, float(distances[a, b]), pts[a], pts[b], n_components


def fractal_dimension(coords: np.ndarray) -> float:
    if len(coords) < 8:
        return float("nan")
    shifted = coords - coords.min(axis=0)
    extent = shifted.max(axis=0) + 1
    sizes, counts = [], []
    for size in (1, 2, 4, 8, 16, 32):
        if size > max(extent):
            break
        counts.append(len(np.unique(shifted // size, axis=0)))
        sizes.append(size)
    if len(counts) < 2 or min(counts) <= 0:
        return float("nan")
    return float(np.polyfit(np.log(1 / np.asarray(sizes)), np.log(counts), 1)[0])


def most_specific_name(names: list[str]) -> str:
    for name in reversed(names):
        if name and name.lower() != "trachea":
            return name
    return "trachea"


def _local_direction(points: np.ndarray, endpoint: np.ndarray, away: bool,
                     window_mm: float = 3.0) -> np.ndarray | None:
    """PCA direction local to an endpoint, oriented toward/away from it."""
    if len(points) < 2:
        return None
    distances = np.linalg.norm(points - endpoint, axis=1)
    local = points[distances <= window_mm]
    if len(local) < 3:
        local = points[np.argsort(distances)[:min(5, len(points))]]
    if len(local) < 2:
        return None
    centered = local - local.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    inward = local.mean(axis=0) - endpoint
    # For a parent we want the vector approaching its endpoint; for a child,
    # the vector leaving its endpoint.
    desired = -inward if away else inward
    if np.dot(direction, desired) < 0:
        direction = -direction
    return direction / np.linalg.norm(direction)


def branch_angle(parent_geom, child_geom) -> tuple[float, str]:
    """Angle between local parent/child directions at their nearest endpoints."""
    if parent_geom is None or child_geom is None:
        return float("nan"), "root-no-parent"
    p0, p1, ppoints = parent_geom
    c0, c1, cpoints = child_geom
    pairs = [(np.linalg.norm(p - c), pi, ci) for pi, p in enumerate((p0, p1))
             for ci, c in enumerate((c0, c1))]
    _, pi, ci = min(pairs)
    pv = _local_direction(ppoints, (p0, p1)[pi], away=False)
    cv = _local_direction(cpoints, (c0, c1)[ci], away=True)
    if pv is None or cv is None:
        return float("nan"), "insufficient-centerline-points"
    return (float(np.degrees(np.arccos(np.clip(np.dot(pv, cv), -1, 1)))), "ok")


def measure_case(case_dir: Path) -> list[dict]:
    case_id, subject, age_group = case_fields(case_dir)
    products = classifier_products(case_dir)
    if not all(p.exists() for p in products):
        raise FileNotFoundError("classifier products incomplete")

    mask_img = sitk.ReadImage(str(case_dir / "union3_lcc.nii.gz"))
    mask = sitk.GetArrayFromImage(mask_img) > 0
    skel_parse = sitk.GetArrayFromImage(sitk.ReadImage(str(products[0]))).astype(np.int32)
    airway_parse = sitk.GetArrayFromImage(sitk.ReadImage(str(products[1]))).astype(np.int32)
    spacing_zyx = np.asarray(mask_img.GetSpacing())[::-1]
    voxel_volume = float(np.prod(mask_img.GetSpacing()))

    # The distance transform is only read at skeleton voxels, all inside the
    # mask, and the nearest background voxel to an airway voxel is never
    # outside the mask's bounding box padded by one voxel (that ring is all
    # background). Computing it on that crop gives identical values in a
    # fraction of the time on a 200 M-voxel scan.
    box = ndimage.find_objects(mask.astype(np.uint8))[0]
    box = tuple(slice(max(0, sl.start - 1), min(n, sl.stop + 1)) for sl, n in zip(box, mask.shape))
    distance = np.zeros(mask.shape, dtype=np.float64)
    distance[box] = ndimage.distance_transform_edt(mask[box], sampling=spacing_zyx)

    # Every branch's skeleton voxels in one pass instead of one full-volume
    # comparison per branch. A stable sort keeps the C-order that
    # np.argwhere(skel_parse == id) would have produced.
    all_coords = np.argwhere(skel_parse > 0)
    all_labels = skel_parse[tuple(all_coords.T)]
    order = np.argsort(all_labels, kind="stable")
    all_coords, all_labels = all_coords[order], all_labels[order]
    bounds = np.searchsorted(all_labels, np.arange(int(skel_parse.max()) + 2))
    coords_of = lambda pid: all_coords[bounds[pid]:bounds[pid + 1]] if 0 <= pid <= int(skel_parse.max()) else all_coords[:0]
    voxels_of = np.bincount(airway_parse.ravel(), minlength=int(airway_parse.max()) + 1)

    with products[5].open() as f:
        annotations = json.load(f)
    parse_ids = [int(k) for k in annotations]
    edge = np.load(products[2]).astype(int)
    features = np.load(products[3])
    if len(parse_ids) != len(features):
        raise ValueError(f"annotation/feature row mismatch: {len(parse_ids)} vs {len(features)}")

    parents = {i: None for i in range(len(parse_ids))}
    for parent, child in edge.T:
        parents[int(child)] = int(parent)

    geom = {}
    for row_idx, parse_id in enumerate(parse_ids):
        coords = coords_of(parse_id)
        length, chord, endpoint0, endpoint1, components = centerline_geometry(
            coords, spacing_zyx)
        physical = coords.astype(float) * spacing_zyx
        geom[row_idx] = (length, chord, endpoint0, endpoint1, coords, physical,
                         components)

    rows = []
    for row_idx, parse_id in enumerate(parse_ids):
        names = annotations[str(parse_id)]
        parent_idx = parents[row_idx]
        parent_parse_id = parse_ids[parent_idx] if parent_idx is not None else None
        length, chord, endpoint0, endpoint1, coords, physical, components = geom[row_idx]
        radius = distance[tuple(coords.T)] if len(coords) else np.array([])
        branch_voxels = int(voxels_of[parse_id]) if parse_id < len(voxels_of) else 0
        rmin = float(radius.min()) if len(radius) else float("nan")
        rmean = float(radius.mean()) if len(radius) else float("nan")
        rmax = float(radius.max()) if len(radius) else float("nan")
        rsd = float(radius.std()) if len(radius) else float("nan")
        angle, angle_status = branch_angle(
            (geom[parent_idx][2], geom[parent_idx][3], geom[parent_idx][5])
            if parent_idx is not None else None,
            (endpoint0, endpoint1, physical),
        )
        is_raw_trachea = all(str(n).lower() == "trachea" for n in names)
        is_root = parent_idx is None
        if is_raw_trachea and not is_root:
            display_name = "proximal_or_unclassified"
            anatomy_status = "radiologist-review-required"
        else:
            display_name = most_specific_name(names)
            anatomy_status = "radiologist-review-required-pediatric-domain"
        if len(radius):
            p10, p50, p90 = np.percentile(radius, [10, 50, 90])
        else:
            p10 = p50 = p90 = float("nan")
        qc_flags = []
        if components != 1:
            qc_flags.append(f"disconnected-centerline({components})")
        if len(coords) < 8:
            qc_flags.append("short-branch-complexity-not-estimable")
        if is_raw_trachea and not is_root:
            qc_flags.append("nonroot-trachea-model-output")
        rows.append({
            "case_id": case_id, "subject": subject, "age_group": age_group,
            "branch_instance_id": parse_id, "parent_instance_id": parent_parse_id,
            "generation": int(round(float(features[row_idx, 0]))),
            "branch_name": display_name, "lobar_name": names[0],
            "segmental_name": names[1], "subsegmental_name": names[2],
            "centerline_length_mm": length, "chord_length_mm": chord,
            "tortuosity_length_ratio": length / chord if chord > 0 else float("nan"),
            "radius_min_mm": rmin, "radius_mean_mm": rmean, "radius_max_mm": rmax,
            "radius_sd_mm": rsd, "diameter_min_mm": 2*rmin,
            "diameter_mean_mm": 2*rmean, "diameter_max_mm": 2*rmax,
            "branch_angle_deg": angle,
            "branch_angle_method": "local-3mm-endpoint-PCA",
            "branch_angle_status": angle_status,
            "volume_mm3": branch_voxels * voxel_volume,
            "centerline_voxels": len(coords), "branch_voxels": branch_voxels,
            "centerline_components": components,
            "diameter_p10_mm": 2*p10, "diameter_median_mm": 2*p50,
            "diameter_p90_mm": 2*p90,
            # Descriptive within-branch proxies, not clinical diagnoses.
            "caliber_reduction_proxy": 1-p10/p50 if p50 > 0 else float("nan"),
            "caliber_expansion_proxy": p90/p50-1 if p50 > 0 else float("nan"),
            "legacy_min_mean_caliber_index": 1-rmin/rmean if rmean > 0 else float("nan"),
            "legacy_max_mean_caliber_index": rmax/rmean-1 if rmean > 0 else float("nan"),
            "complexity_fractal_dimension": fractal_dimension(coords),
            "complexity_status": "ok" if len(coords) >= 8 else "not-estimable-<8-voxels",
            "anatomy_review_status": anatomy_status,
            "geometry_qc_flags": ";".join(qc_flags),
            "source_mask": str(case_dir / "union3_lcc.nii.gz"),
            "classification_status": "AirMorph-predicted; not pediatric-validated",
        })
    return rows


def write_csv(path: Path, rows: list[dict], columns=FEATURE_COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    os.replace(partial, path)


def write_excel(path: Path, rows: list[dict], statuses: list[dict]):
    try:
        import pandas as pd
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Excel export requires openpyxl in the AirMorph environment") from exc
    dictionary = [
        ("branch_instance_id", "AirMorph parsed physical-branch label; unique within a case"),
        ("branch_name", "Most-specific AirMorph anatomical prediction"),
        ("centerline_length_mm", "26-neighbour physical minimum-spanning-tree length"),
        ("chord_length_mm", "Euclidean distance between branch endpoints"),
        ("tortuosity_length_ratio", "centerline length / endpoint chord; >=1 ideally"),
        ("radius_*_mm", "Euclidean distance-transform radius sampled on the centerline"),
        ("diameter_*_mm", "Two times the corresponding radius"),
        ("branch_angle_deg", "Parent-to-child direction change at their nearest endpoints"),
        ("volume_mm3", "Parsed branch voxel count times voxel volume"),
        ("legacy_min_mean_caliber_index", "Legacy 1 - minimum/mean EDT radius; descriptive only, not clinical stenosis"),
        ("legacy_max_mean_caliber_index", "Legacy maximum/mean EDT radius - 1; descriptive only, not clinical ectasia/bronchiectasis"),
        ("complexity_fractal_dimension", "Box-counting dimension of branch centerline"),
        ("branch_name", "Review display label; non-root all-trachea predictions are shown as proximal_or_unclassified"),
        ("lobar_name/segmental_name/subsegmental_name", "Unmodified raw AirMorph model outputs"),
        ("branch_angle_deg", "Angle between local PCA directions within 3 mm of the parent-child junction"),
        ("diameter_p10/median/p90_mm", "Twice EDT radius percentiles; segmentation-derived caliber, not wall-measured clinical diameter"),
        ("caliber_reduction_proxy", "1 - radius p10 / radius median; descriptive proxy, not diagnosed stenosis"),
        ("caliber_expansion_proxy", "radius p90 / radius median - 1; descriptive proxy, not diagnosed bronchiectasis"),
        ("complexity_status", "Explains missing fractal dimension; short branches are not assigned unstable estimates"),
        ("anatomy_review_status", "AirMorph anatomy requires expert review because pediatric external validity is not established here"),
    ]
    partial = path.with_suffix(".partial.xlsx")
    with pd.ExcelWriter(partial, engine="openpyxl") as writer:
        pd.DataFrame(rows, columns=FEATURE_COLUMNS).to_excel(
            writer, sheet_name="Branches", index=False)
        pd.DataFrame(statuses).to_excel(writer, sheet_name="Cases", index=False)
        pd.DataFrame(dictionary, columns=["column", "definition"]).to_excel(
            writer, sheet_name="Data Dictionary", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("classify", "export", "all"), default="all")
    parser.add_argument("--case", help="one case ID for pilot/debugging")
    parser.add_argument("--force-classify", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cases = normal_cases()
    if args.case:
        cases = [p for p in cases if p.name == args.case]
        if not cases:
            parser.error(f"normal case not found: {args.case}")

    statuses, all_rows = [], []
    for number, case_dir in enumerate(cases, 1):
        status = {"case_id": case_dir.name, "case_dir": str(case_dir),
                  "classification": "not-requested", "branches": 0,
                  "proximal_or_unclassified_branches": 0,
                  "complexity_not_estimable_branches": 0,
                  "geometry_flagged_branches": 0,
                  "pediatric_anatomy_validation": "pending-radiologist-review",
                  "error": ""}
        try:
            logging.info("[%d/%d] %s", number, len(cases), case_dir.name)
            if args.phase in ("classify", "all"):
                status["classification"] = classify_case(case_dir, args.force_classify)
            if args.phase in ("export", "all"):
                rows = measure_case(case_dir)
                all_rows.extend(rows); status["branches"] = len(rows)
                status["proximal_or_unclassified_branches"] = sum(
                    r["branch_name"] == "proximal_or_unclassified" for r in rows)
                status["complexity_not_estimable_branches"] = sum(
                    r["complexity_status"] != "ok" for r in rows)
                status["geometry_flagged_branches"] = sum(
                    bool(r["geometry_qc_flags"]) for r in rows)
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
            logging.error("%s\n%s", status["error"], traceback.format_exc())
        statuses.append(status)
        write_csv(STATUS_OUTPUT, statuses, list(statuses[0]))
        if all_rows:
            write_csv(CSV_OUTPUT, all_rows)

    if args.phase in ("export", "all") and all_rows:
        write_excel(args.output, all_rows, statuses)
        logging.info("wrote %d branch rows to %s and %s", len(all_rows), CSV_OUTPUT, args.output)
    failures = sum(bool(s["error"]) for s in statuses)
    logging.info("complete: %d cases, %d failures", len(statuses), failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
