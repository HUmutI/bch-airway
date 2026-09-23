#!/usr/bin/env python3
"""Build conservative, analysis-ready tables from the full morphometry export.

The raw export is preserved.  This derivative removes the invalid fractal-dimension
estimator, adds resolution/generation eligibility flags, detects cross-head anatomy
contradictions, corrects root display labels, and marks repeated subjects.
"""
from __future__ import annotations

import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


RUN = Path(os.environ.get("OR3_RUN", "/fileserver/external/quin/ege/or_all3_run"))
SOURCE = RUN / "normal_airway_branch_morphometry.csv"
STATUS = RUN / "normal_airway_morphometry_status.csv"
OUTDIR = RUN / "radiologist_qc/tables"
OUTCSV = OUTDIR / "analysis_ready_airway_branch_morphometry.csv"
OUTXLSX = OUTDIR / "analysis_ready_airway_branch_morphometry.xlsx"


def segment_code(value) -> str | None:
    if pd.isna(value):
        return None
    match = re.match(r"^([LR]B(?:1\+2|\d+))", str(value).strip())
    return match.group(1) if match else None


def side(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.startswith(("Left", "LB")):
        return "L"
    if text.startswith(("Right", "RB")):
        return "R"
    return None


def geometry_and_branch_directions(status: pd.DataFrame, branches: pd.DataFrame):
    """Read true header spacing and estimate each branch axis in grid coordinates."""
    case_geometry, directions = {}, {}
    for row in status.itertuples():
        case_dir = Path(row.case_dir)
        mask_path = case_dir / "union3_lcc.nii.gz"
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(mask_path)); reader.ReadImageInformation()
        spacing_xyz = np.asarray(reader.GetSpacing(), dtype=float)
        case_geometry[row.case_id] = {
            "spacing_x_mm": spacing_xyz[0], "spacing_y_mm": spacing_xyz[1],
            "spacing_z_mm": spacing_xyz[2],
            "voxel_spacing_floor_mm": spacing_xyz.min(),
            "spacing_anisotropy_ratio": spacing_xyz.max() / spacing_xyz.min(),
        }

        case_id = row.case_id
        parse_path = case_dir / f"{case_id}_skel_parsing.nii.gz"
        labels = sitk.GetArrayFromImage(sitk.ReadImage(str(parse_path))).astype(np.int32)
        objects = ndimage.find_objects(labels)
        spacing_zyx = spacing_xyz[::-1]
        for branch_id in branches.loc[branches.case_id == case_id,
                                      "branch_instance_id"].astype(int):
            slc = objects[branch_id - 1] if 0 < branch_id <= len(objects) else None
            if slc is None:
                directions[(case_id, branch_id)] = (np.nan,) * 4
                continue
            crop = labels[slc]
            coords = np.argwhere(crop == branch_id)
            starts = np.asarray([axis.start for axis in slc])
            physical = (coords + starts) * spacing_zyx
            if len(physical) < 2:
                directions[(case_id, branch_id)] = (np.nan,) * 4
                continue
            centered = physical - physical.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            unit_zyx = vh[0] / np.linalg.norm(vh[0])
            unit_xyz = unit_zyx[::-1]
            # Conservative grid resolution projected into the plane normal to
            # the branch axis. A through-plane (z-directed) branch is therefore
            # governed by x/y resolution, while an in-plane branch includes z.
            projected = spacing_xyz * np.sqrt(np.maximum(0, 1 - unit_xyz**2))
            cross_section_resolution = float(projected.max())
            directions[(case_id, branch_id)] = (
                unit_xyz[0], unit_xyz[1], unit_xyz[2], cross_section_resolution)
        del labels
    return case_geometry, directions


def anatomy_flags(row) -> str:
    flags = []
    root = pd.isna(row.parent_instance_id)
    if root and str(row.branch_name).lower() != "trachea":
        flags.append("root-model-label-not-trachea")

    values = [row.lobar_name, row.segmental_name, row.subsegmental_name]
    sides = {s for s in map(side, values) if s}
    if len(sides) > 1:
        flags.append("left-right-head-contradiction")

    seg = segment_code(row.segmental_name)
    sub = segment_code(row.subsegmental_name)
    if seg and sub and seg != sub:
        flags.append(f"segment-subsegment-head-contradiction({seg}!={sub})")
    return ";".join(flags)


def consensus_name(row) -> str:
    if pd.isna(row.parent_instance_id):
        return "trachea"
    if row.anatomy_consistency_flags:
        return "needs_radiologist_review"
    return row.branch_name


def main() -> None:
    branches = pd.read_csv(SOURCE)
    cases = pd.read_csv(STATUS)
    geometry, directions = geometry_and_branch_directions(cases, branches)
    for column in ("spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
                   "voxel_spacing_floor_mm", "spacing_anisotropy_ratio"):
        branches[column] = branches.case_id.map(
            {case_id: values[column] for case_id, values in geometry.items()})
    branches["edt_diameter_floor_mm"] = 2 * branches.voxel_spacing_floor_mm
    branches["radius_touches_resolution_floor"] = (
        branches.radius_min_mm <= branches.voxel_spacing_floor_mm * 1.01)
    direction_values = [directions[(case_id, int(branch_id))]
                        for case_id, branch_id in zip(
                            branches.case_id, branches.branch_instance_id)]
    branches[["branch_axis_x", "branch_axis_y", "branch_axis_z",
              "directional_cross_section_resolution_mm"]] = direction_values
    branches["directional_diameter_resolution_mm"] = (
        2 * branches.directional_cross_section_resolution_mm)
    branches["caliber_directionally_resolution_limited"] = (
        branches.diameter_p10_mm <= branches.directional_diameter_resolution_mm)
    branches["anatomy_consistency_flags"] = branches.apply(anatomy_flags, axis=1)
    branches["anatomy_consensus_name"] = branches.apply(consensus_name, axis=1)

    branches["exclude_generation_gt8"] = branches.generation > 8
    branches["exclude_generation_ge14"] = branches.generation >= 14
    branches["exclude_caliber_resolution"] = (
        branches.radius_touches_resolution_floor |
        branches.caliber_directionally_resolution_limited)
    branches["exclude_tortuosity_short"] = (
        (branches.centerline_voxels < 8) | (branches.centerline_length_mm < 2))
    branches["anatomy_review_required"] = branches.anatomy_consistency_flags.astype(bool)
    branches["high_anisotropy_sensitivity_flag"] = branches.spacing_anisotropy_ratio > 3
    branches["segmentation_qc_flags"] = np.where(
        branches.parent_instance_id.isna() & branches.radius_touches_resolution_floor,
        "root-trachea-radius-touches-grid-floor", "")

    branches["caliber_analysis_status"] = np.select(
        [branches.exclude_generation_gt8, branches.exclude_caliber_resolution],
        ["exclude-generation>8", "exclude-directional-resolution-limit"],
        default="eligible",
    )
    branches["tortuosity_analysis_status"] = np.select(
        [branches.exclude_generation_gt8, branches.exclude_tortuosity_short],
        ["exclude-generation>8", "exclude-short-branch"],
        default="eligible",
    )
    branches["overall_analysis_status"] = np.select(
        [branches.exclude_generation_ge14, branches.anatomy_review_required],
        ["exclude-generation>=14", "anatomy-review-required"],
        default="eligible-with-metric-specific-flags",
    )

    # This estimator is mathematically invalid in the raw table (connected curve
    # estimates below 1); do not propagate it into analysis-ready data.
    drop = ["complexity_fractal_dimension", "complexity_status", "source_mask"]
    branches = branches.drop(columns=[c for c in drop if c in branches])

    identity = branches[["case_id", "subject", "age_group"]].drop_duplicates()
    subject_counts = identity.groupby("subject").agg(
        subject_scan_count=("case_id", "nunique"),
        subject_age_group_count=("age_group", "nunique"),
    )
    cases = cases.merge(identity, on="case_id", how="left").merge(
        subject_counts, on="subject", how="left")
    cases["independence_warning"] = np.where(
        cases.subject_scan_count > 1,
        "repeated-subject-use-mixed-model-or-one-scan-per-subject", "")
    cases["anatomy_contradictory_branches"] = cases.case_id.map(
        branches.groupby("case_id").anatomy_consistency_flags.apply(
            lambda x: int(x.astype(bool).sum())))
    cases["resolution_limited_branches"] = cases.case_id.map(
        branches.groupby("case_id").exclude_caliber_resolution.sum()).astype(int)
    for column in ("spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
                   "spacing_anisotropy_ratio"):
        cases[column] = cases.case_id.map(
            {case_id: values[column] for case_id, values in geometry.items()})
    cases["high_anisotropy_sensitivity_flag"] = cases.spacing_anisotropy_ratio > 3
    cases["root_trachea_resolution_defect"] = cases.case_id.isin(
        branches.loc[branches.segmentation_qc_flags.astype(bool), "case_id"])
    cases = cases.drop(columns=["case_dir", "complexity_not_estimable_branches"],
                       errors="ignore")

    dictionary = [
        ("anatomy_consensus_name", "Root forced to trachea; contradictory model heads become needs_radiologist_review; raw heads are preserved."),
        ("anatomy_consistency_flags", "Root, left/right, and segment-vs-subsegment cross-head checks."),
        ("voxel_spacing_floor_mm", "Smallest physical voxel spacing in the source segmentation grid."),
        ("spacing_x/y/z_mm", "Physical voxel spacing read directly from the NIfTI header with SimpleITK ImageFileReader."),
        ("spacing_anisotropy_ratio", "Maximum header spacing divided by minimum header spacing."),
        ("edt_diameter_floor_mm", "Twice the smallest voxel spacing; practical EDT diameter floor."),
        ("radius_touches_resolution_floor", "Branch minimum EDT radius is within 1% of the grid floor."),
        ("branch_axis_x/y/z", "PCA branch direction in physical grid coordinates."),
        ("directional_cross_section_resolution_mm", "Conservative maximum voxel-axis projection into the plane normal to the branch."),
        ("caliber_directionally_resolution_limited", "Diameter p10 does not exceed twice the direction-aware cross-sectional resolution."),
        ("exclude_* and anatomy_review_required", "Independent boolean flags; use these for counts because status strings have precedence."),
        ("caliber_analysis_status", "Caliber excluded past generation 8 or at the direction-aware resolution limit."),
        ("tortuosity_analysis_status", "Tortuosity excluded past generation 8 or for branches <8 voxels/<2 mm."),
        ("overall_analysis_status", "Generation >=14 excluded; anatomy contradictions require review."),
        ("independence_warning", "Repeated subjects are not statistically independent."),
    ]
    notes = [
        ("Fractal complexity", "Removed from analysis-ready data: 92.35% of non-null raw estimates were below the mathematical lower bound of 1 for a connected curve."),
        ("Caliber", "Primary comparisons restricted to generations <=8 and branches not touching the EDT resolution floor."),
        ("Deep branches", "Generation >=14 excluded from primary analysis because of resolution plateau/leak risk."),
        ("Volumes", "Segmentation-derived volume is resolution/partial-volume sensitive; compare within generation, not across generations."),
        ("Anatomy", "Raw lobar, segmental and subsegmental outputs remain available; consensus fields never silently repair contradictions."),
        ("Status precedence", "Use independent boolean flags for counting; convenience status strings report only the first applicable reason."),
        ("Root defects", "A root trachea touching the minimum-radius grid floor is flagged as a segmentation defect for direct review."),
        ("Repeated subjects", "Use a mixed-effects model or select one scan per subject for independent-sample analyses."),
    ]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    branches.to_csv(OUTCSV, index=False)
    with pd.ExcelWriter(OUTXLSX, engine="openpyxl") as writer:
        branches.to_excel(writer, sheet_name="Analysis Ready Branches", index=False)
        cases.to_excel(writer, sheet_name="Cases", index=False)
        pd.DataFrame(dictionary, columns=["column", "definition"]).to_excel(
            writer, sheet_name="Data Dictionary", index=False)
        pd.DataFrame(notes, columns=["topic", "recommendation"]).to_excel(
            writer, sheet_name="Analysis Rules", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    print(f"wrote {len(branches)} rows to {OUTCSV} and {OUTXLSX}")


if __name__ == "__main__":
    main()
