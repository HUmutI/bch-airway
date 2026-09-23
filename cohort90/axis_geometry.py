#!/usr/bin/env python3
"""Perpendicular cross-sectional airway geometry.

This module replaces two measurement conventions used in the earlier packages:

1. Tracheal length was previously the whole segmented root centerline, which can
   include cervical trachea above the thoracic inlet. Here the tracheal
   centerline is additionally clipped at a lung-apex proxy for the thoracic
   inlet, giving an intrathoracic carina-to-inlet length comparable with
   Kuo et al., AJR 2018.

2. Diameter was previously twice the Euclidean distance-transform radius
   sampled on the centerline. Because that radius is the largest sphere that
   fits inside the lumen, it approximates the shortest diameter of a
   non-circular airway such as the trachea. Here a plane perpendicular to the
   local centerline tangent is resampled from the fused airway mask, and the
   shortest and longest calliper widths of the lumen contour are measured. The
   reported diameter is their mean, following the source paper's convention.

Both conventions are computed for every case so the two can be compared
directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial import ConvexHull
from skimage.measure import find_contours

# Plane-resampling controls. The half extent bounds the resampled patch and the
# step is its in-plane pixel size; both are millimetres.
PLANE_STEP_MM = 0.25
TRACHEA_HALF_EXTENT_MM = 28.0
BRONCHUS_HALF_EXTENT_MM = 18.0

# Centerline sampling. The trachea is sampled finely for the distal/middle/
# proximal thirds, and the subset falling on 10-mm multiples reproduces the
# source paper's tracheal sampling interval. Main bronchi use 5 mm throughout,
# also following the source paper.
TRACHEA_FINE_STEP_MM = 2.0
TRACHEA_PROTOCOL_STEP_MM = 10.0
BRONCHUS_FINE_STEP_MM = 1.0
BRONCHUS_PROTOCOL_STEP_MM = 5.0

# Centerline margins that keep cross-sections away from bifurcation flare.
TRACHEA_CARINA_MARGIN_MM = 5.0
TRACHEA_SUPERIOR_MARGIN_MM = 2.0
# At the carina the trachea and both main bronchi form one confluent lumen, so a
# cross-section taken too close to it measures the confluence and can exceed the
# tracheal diameter. The proximal margin is therefore a fraction of the
# bronchial length as well as an absolute floor.
BRONCHUS_PROXIMAL_MARGIN_MM = 3.0
BRONCHUS_PROXIMAL_MARGIN_FRACTION = 0.30
BRONCHUS_DISTAL_MARGIN_MM = 2.0
BRONCHUS_DISTAL_MARGIN_FRACTION = 0.10

# Tangent estimation window, millimetres either side of the sample point.
TANGENT_WINDOW_MM = 3.0

NEIGHBOR_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) > (0, 0, 0)
]


class GeometryError(RuntimeError):
    """Raised when a case cannot be measured with the requested convention."""


@dataclass
class CaseVolumes:
    """Arrays and geometry shared by every measurement on one scan."""

    case_id: str
    parse: np.ndarray
    mask: np.ndarray
    spacing_zyx: np.ndarray
    index_to_physical: np.ndarray
    physical_to_index: np.ndarray
    origin: np.ndarray
    lung_apex_z_mm: float
    total_lung_volume_mL: float
    lung_label_volumes_mL: dict[int, float] = field(default_factory=dict)


def _affine(image: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return index->physical matrix, origin, and its inverse, in (x, y, z)."""
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.diag(np.asarray(image.GetSpacing(), dtype=float))
    forward = direction @ spacing
    return forward, np.asarray(image.GetOrigin(), dtype=float), np.linalg.inv(forward)


def voxels_to_physical(volumes: CaseVolumes, coords_zyx: np.ndarray) -> np.ndarray:
    """Convert (z, y, x) voxel coordinates to physical LPS points."""
    index_xyz = np.atleast_2d(coords_zyx)[:, ::-1].astype(float)
    return index_xyz @ volumes.index_to_physical.T + volumes.origin


def load_case(case_id: str, case_dir: Path) -> CaseVolumes:
    """Read the skeleton parsing, fused airway mask, and lung mask for a case."""
    parse_path = case_dir / f"{case_id}_skel_parsing.nii.gz"
    mask_path = case_dir / "union3_lcc.nii.gz"
    lung_path = case_dir / "lunglobe.nii.gz"
    for path in (parse_path, mask_path, lung_path):
        if not path.exists():
            raise GeometryError(f"{case_id}: missing {path.name}")

    parse_image = sitk.ReadImage(str(parse_path))
    parse = sitk.GetArrayFromImage(parse_image)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
    if mask.shape != parse.shape:
        raise GeometryError(f"{case_id}: airway mask and skeleton parsing geometry differ")

    lung_image = sitk.ReadImage(str(lung_path))
    lung = sitk.GetArrayFromImage(lung_image)
    if lung.shape != parse.shape:
        raise GeometryError(f"{case_id}: lung mask and skeleton parsing geometry differ")

    forward, origin, inverse = _affine(parse_image)
    volumes = CaseVolumes(
        case_id=case_id,
        parse=parse,
        mask=mask,
        spacing_zyx=np.asarray(parse_image.GetSpacing(), dtype=float)[::-1],
        index_to_physical=forward,
        physical_to_index=inverse,
        origin=origin,
        lung_apex_z_mm=np.nan,
        total_lung_volume_mL=0.0,
    )

    lung_coords = np.argwhere(lung > 0)
    if not len(lung_coords):
        raise GeometryError(f"{case_id}: lung mask is empty, no thoracic-inlet proxy")
    # The lung apex is the most superior segmented lung voxel. In LPS physical
    # space the z axis increases towards the head.
    volumes.lung_apex_z_mm = float(voxels_to_physical(volumes, lung_coords)[:, 2].max())

    voxel_mm3 = float(np.prod(np.asarray(lung_image.GetSpacing(), dtype=float)))
    volumes.total_lung_volume_mL = float(np.count_nonzero(lung) * voxel_mm3 / 1000.0)
    if volumes.total_lung_volume_mL <= 0:
        raise GeometryError(f"{case_id}: zero segmented lung volume")
    for label in (1, 2):
        volumes.lung_label_volumes_mL[label] = float(
            np.count_nonzero(lung == label) * voxel_mm3 / 1000.0
        )
    return volumes


def branch_centerline(volumes: CaseVolumes, branch_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the longest centerline path of one parsed branch.

    The path is returned as voxel (z, y, x) coordinates together with the
    cumulative physical arc length in millimetres at each point.
    """
    coords = np.argwhere(volumes.parse == branch_id)
    if len(coords) < 2:
        raise GeometryError(
            f"{volumes.case_id}: branch {branch_id} has {len(coords)} centerline voxels"
        )

    lookup = {tuple(coord): i for i, coord in enumerate(coords)}
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for i, coord in enumerate(coords):
        for offset in NEIGHBOR_OFFSETS:
            j = lookup.get(tuple(coord + offset))
            if j is None:
                continue
            weight = float(np.linalg.norm(np.asarray(offset) * volumes.spacing_zyx))
            rows.extend((i, j))
            cols.extend((j, i))
            weights.extend((weight, weight))
    if not rows:
        raise GeometryError(f"{volumes.case_id}: branch {branch_id} centerline is disconnected")

    graph = sparse.csr_matrix((weights, (rows, cols)), shape=(len(coords), len(coords)))
    tree = minimum_spanning_tree(graph)
    undirected = (tree + tree.T).tocsr()
    degree = np.asarray(undirected.astype(bool).sum(axis=1)).ravel()
    endpoints = np.flatnonzero(degree <= 1)
    if len(endpoints) < 2:
        raise GeometryError(f"{volumes.case_id}: branch {branch_id} has fewer than two endpoints")

    distances, predecessors = dijkstra(
        undirected, directed=False, indices=endpoints, return_predecessors=True
    )
    endpoint_distances = distances[:, endpoints]
    source_row, target_col = np.unravel_index(
        np.nanargmax(endpoint_distances), endpoint_distances.shape
    )
    source = int(endpoints[source_row])
    target = int(endpoints[target_col])

    path = [target]
    cursor = target
    while cursor != source:
        cursor = int(predecessors[source_row, cursor])
        if cursor < 0:
            raise GeometryError(f"{volumes.case_id}: branch {branch_id} path is unreachable")
        path.append(cursor)
    path.reverse()

    path_coords = coords[np.asarray(path, dtype=int)]
    steps = np.linalg.norm(np.diff(path_coords, axis=0) * volumes.spacing_zyx, axis=1)
    return path_coords, np.r_[0.0, np.cumsum(steps)]


def orient_path(
    points_mm: np.ndarray, cumulative: np.ndarray, reference_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Order a path so that its first point is the end nearer to `reference_mm`."""
    start = float(np.linalg.norm(points_mm[0] - reference_mm))
    end = float(np.linalg.norm(points_mm[-1] - reference_mm))
    if start <= end:
        return points_mm, cumulative
    return points_mm[::-1], cumulative[-1] - cumulative[::-1]


def _plane_basis(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane normal to `tangent`."""
    normal = tangent / np.linalg.norm(tangent)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, normal))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, seed)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return first, second / np.linalg.norm(second)


def _tangent_at(points_mm: np.ndarray, cumulative: np.ndarray, position_mm: float) -> np.ndarray:
    """Estimate the centerline direction at an arc-length position."""
    low = max(float(cumulative[0]), position_mm - TANGENT_WINDOW_MM)
    high = min(float(cumulative[-1]), position_mm + TANGENT_WINDOW_MM)
    if high - low < 1e-6:
        low, high = float(cumulative[0]), float(cumulative[-1])
    start = np.array([np.interp(low, cumulative, points_mm[:, axis]) for axis in range(3)])
    end = np.array([np.interp(high, cumulative, points_mm[:, axis]) for axis in range(3)])
    tangent = end - start
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        raise GeometryError("degenerate centerline tangent")
    return tangent / norm


def _feret_widths(contour_mm: np.ndarray) -> tuple[float, float]:
    """Return the shortest and longest calliper widths of a closed contour."""
    hull = ConvexHull(contour_mm)
    vertices = contour_mm[hull.vertices]
    differences = vertices[:, None, :] - vertices[None, :, :]
    longest = float(np.sqrt((differences ** 2).sum(axis=2)).max())

    shortest = np.inf
    count = len(vertices)
    for i in range(count):
        edge = vertices[(i + 1) % count] - vertices[i]
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        normal = np.array([-edge[1], edge[0]]) / length
        projections = (vertices - vertices[i]) @ normal
        shortest = min(shortest, float(projections.max() - projections.min()))
    if not np.isfinite(shortest):
        raise GeometryError("degenerate lumen contour")
    return shortest, longest


def cross_section(
    volumes: CaseVolumes, center_mm: np.ndarray, tangent: np.ndarray, half_extent_mm: float
) -> dict | None:
    """Measure one lumen cross-section perpendicular to the centerline.

    Returns ``None`` when the plane cannot be measured, which happens if the
    centerline point falls outside the mask or the lumen touches the edge of the
    resampled patch.
    """
    first, second = _plane_basis(tangent)
    offsets = np.arange(-half_extent_mm, half_extent_mm + PLANE_STEP_MM, PLANE_STEP_MM)
    grid_first, grid_second = np.meshgrid(offsets, offsets, indexing="ij")
    points = (
        center_mm
        + grid_first[..., None] * first
        + grid_second[..., None] * second
    )

    index_xyz = (points - volumes.origin) @ volumes.physical_to_index.T
    index_zyx = index_xyz[..., ::-1]
    # Trilinear interpolation of the binary mask keeps the partial-volume
    # boundary. Nearest-neighbour sampling would reproduce the source voxel
    # staircase and bias the longest calliper outwards by up to one voxel.
    occupancy = ndimage.map_coordinates(
        volumes.mask.astype(np.float32),
        index_zyx.reshape(-1, 3).T,
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(grid_first.shape)

    plane = occupancy >= 0.5
    center_index = len(offsets) // 2
    if not plane[center_index, center_index]:
        return None

    labels, _ = ndimage.label(plane)
    lumen = labels == labels[center_index, center_index]
    if (
        lumen[0, :].any()
        or lumen[-1, :].any()
        or lumen[:, 0].any()
        or lumen[:, -1].any()
    ):
        # The lumen reaches the patch border, so its true width is unknown.
        return None

    # Contour the interpolated field, blanking every other lumen that crosses
    # this plane so neighbouring airways cannot join the measured contour.
    field = np.where(lumen, occupancy, 0.0)
    contours = find_contours(field, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len) * PLANE_STEP_MM
    if len(contour) < 4:
        return None

    shortest, longest = _feret_widths(contour)
    area_mm2 = float(np.clip(field, 0.0, 1.0).sum()) * PLANE_STEP_MM ** 2
    return {
        "short_axis_mm": shortest,
        "long_axis_mm": longest,
        "mean_axis_diameter_mm": 0.5 * (shortest + longest),
        "area_mm2": area_mm2,
        "area_equivalent_diameter_mm": 2.0 * float(np.sqrt(area_mm2 / np.pi)),
        "eccentricity_ratio": shortest / longest if longest > 0 else np.nan,
    }


def sample_segment(
    volumes: CaseVolumes,
    points_mm: np.ndarray,
    cumulative: np.ndarray,
    positions_mm: np.ndarray,
    half_extent_mm: float,
) -> list[dict]:
    """Measure cross-sections at the requested arc-length positions."""
    records = []
    for position in positions_mm:
        center = np.array(
            [np.interp(position, cumulative, points_mm[:, axis]) for axis in range(3)]
        )
        try:
            tangent = _tangent_at(points_mm, cumulative, float(position))
        except GeometryError:
            continue
        measured = cross_section(volumes, center, tangent, half_extent_mm)
        if measured is None:
            continue
        measured["position_mm"] = float(position)
        records.append(measured)
    return records


def _summarise(records: list[dict], prefix: str) -> dict:
    """Reduce a list of cross-sections to summary statistics."""
    result: dict[str, float] = {f"{prefix}_cross_sections": len(records)}
    if not records:
        for key in (
            "short_axis_mm",
            "long_axis_mm",
            "mean_axis_diameter_mm",
            "area_equivalent_diameter_mm",
            "eccentricity_ratio",
        ):
            result[f"{prefix}_{key}"] = np.nan
        return result
    for key in (
        "short_axis_mm",
        "long_axis_mm",
        "mean_axis_diameter_mm",
        "area_equivalent_diameter_mm",
        "eccentricity_ratio",
    ):
        values = np.array([record[key] for record in records], dtype=float)
        result[f"{prefix}_{key}"] = float(np.mean(values))
    values = np.array([record["mean_axis_diameter_mm"] for record in records], dtype=float)
    result[f"{prefix}_mean_axis_diameter_median_mm"] = float(np.median(values))
    result[f"{prefix}_mean_axis_diameter_sd_mm"] = (
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    )
    return result


def measure_case(
    case_id: str,
    case_dir: Path,
    root_branch_id: int,
    lmb_branch_id: int,
    rmb_branch_id: int,
) -> dict:
    """Measure one scan's central airway with the cross-sectional convention."""
    volumes = load_case(case_id, case_dir)

    root_coords, root_cumulative = branch_centerline(volumes, root_branch_id)
    root_points = voxels_to_physical(volumes, root_coords)
    # Orient the tracheal path carina -> superior. Physical z increases towards
    # the head in LPS, so the carina end is the one with the smaller z.
    if root_points[0, 2] > root_points[-1, 2]:
        root_points = root_points[::-1]
        root_cumulative = root_cumulative[-1] - root_cumulative[::-1]
    carina_mm = root_points[0]
    full_length_mm = float(root_cumulative[-1])

    # Thoracic-inlet proxy: clip the tracheal centerline at the most superior
    # segmented lung voxel. Anything above that level is treated as cervical
    # trachea and excluded from the intrathoracic length.
    below_apex = root_points[:, 2] <= volumes.lung_apex_z_mm
    if below_apex.all():
        inlet_length_mm = full_length_mm
        clip_applied = False
    elif not below_apex.any():
        raise GeometryError(
            f"{case_id}: whole tracheal centerline lies above the lung apex"
        )
    else:
        last_inside = int(np.flatnonzero(below_apex)[-1])
        inlet_length_mm = float(root_cumulative[last_inside])
        clip_applied = True
    if inlet_length_mm <= TRACHEA_CARINA_MARGIN_MM + TRACHEA_SUPERIOR_MARGIN_MM:
        raise GeometryError(
            f"{case_id}: intrathoracic tracheal length {inlet_length_mm:.1f} mm is too short to sample"
        )

    low = TRACHEA_CARINA_MARGIN_MM
    high = inlet_length_mm - TRACHEA_SUPERIOR_MARGIN_MM
    fine_positions = np.arange(low, high + 1e-9, TRACHEA_FINE_STEP_MM)
    fine = sample_segment(
        volumes, root_points, root_cumulative, fine_positions, TRACHEA_HALF_EXTENT_MM
    )
    protocol_positions = np.arange(low, high + 1e-9, TRACHEA_PROTOCOL_STEP_MM)
    protocol = sample_segment(
        volumes, root_points, root_cumulative, protocol_positions, TRACHEA_HALF_EXTENT_MM
    )

    result: dict[str, object] = {
        "case_id": case_id,
        "full_segmented_tracheal_length_mm": full_length_mm,
        "intrathoracic_tracheal_length_mm": inlet_length_mm,
        "cervical_tracheal_length_mm": full_length_mm - inlet_length_mm,
        "thoracic_inlet_clip_applied": clip_applied,
        "lung_apex_physical_z_mm": volumes.lung_apex_z_mm,
        "total_segmented_lung_volume_mL": volumes.total_lung_volume_mL,
        "lung_label_1_volume_mL": volumes.lung_label_volumes_mL.get(1, np.nan),
        "lung_label_2_volume_mL": volumes.lung_label_volumes_mL.get(2, np.nan),
        "root_centerline_voxels": int(len(root_coords)),
    }
    # The fine sampling is the reported value because it estimates the same
    # quantity with less sampling noise, especially in infants whose
    # intrathoracic trachea admits only two or three 10-mm stations. The
    # 10-mm subset is retained so the source paper's protocol can be compared.
    result.update(_summarise(fine, "trachea"))
    result.update(_summarise(protocol, "trachea_protocol10mm"))

    # Distal, middle, and proximal thirds of the intrathoracic trachea. Position
    # zero is the carina, matching the earlier packages' section convention.
    for name, (start, end) in {
        "distal": (0.0, inlet_length_mm / 3.0),
        "middle": (inlet_length_mm / 3.0, 2.0 * inlet_length_mm / 3.0),
        "proximal": (2.0 * inlet_length_mm / 3.0, inlet_length_mm + 1e-9),
    }.items():
        section = [
            record for record in fine if start <= record["position_mm"] < end
        ]
        result.update(_summarise(section, f"trachea_{name}"))

    for side, branch_id in (("lmb", lmb_branch_id), ("rmb", rmb_branch_id)):
        coords, cumulative = branch_centerline(volumes, branch_id)
        points = voxels_to_physical(volumes, coords)
        points, cumulative = orient_path(points, cumulative, carina_mm)
        length_mm = float(cumulative[-1])
        result[f"{side}_centerline_length_mm"] = length_mm
        result[f"{side}_first_sample_mm_from_carina"] = float(max(
            BRONCHUS_PROXIMAL_MARGIN_MM, BRONCHUS_PROXIMAL_MARGIN_FRACTION * length_mm))
        low = max(BRONCHUS_PROXIMAL_MARGIN_MM,
                  BRONCHUS_PROXIMAL_MARGIN_FRACTION * length_mm)
        high = length_mm - max(BRONCHUS_DISTAL_MARGIN_MM,
                               BRONCHUS_DISTAL_MARGIN_FRACTION * length_mm)
        if high <= low:
            # Very short main bronchi are measured once, as far from the carina
            # as the distal margin allows.
            midpoint = min(max(low, 0.6 * length_mm), length_mm)
            fine_positions = protocol_positions = np.array([midpoint])
        else:
            fine_positions = np.arange(low, high + 1e-9, BRONCHUS_FINE_STEP_MM)
            protocol_positions = np.arange(low, high + 1e-9, BRONCHUS_PROTOCOL_STEP_MM)
        result.update(
            _summarise(
                sample_segment(
                    volumes, points, cumulative, fine_positions, BRONCHUS_HALF_EXTENT_MM
                ),
                side,
            )
        )
        result.update(
            _summarise(
                sample_segment(
                    volumes, points, cumulative, protocol_positions, BRONCHUS_HALF_EXTENT_MM
                ),
                f"{side}_protocol5mm",
            )
        )

    return result
