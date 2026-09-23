#!/usr/bin/env python3
"""Fail-closed validation of the morphometry and radiologist QC deliverables."""
from __future__ import annotations

import os
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(os.environ.get("OR3_RUN", "/fileserver/external/quin/ege/or_all3_run"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--expected-cases", type=int, default=83)
    args = ap.parse_args()
    run = args.run
    branches_path = run / "normal_airway_branch_morphometry.csv"
    cases_path = run / "normal_airway_morphometry_status.csv"
    workbook = run / "normal_airway_branch_morphometry.xlsx"
    review = run / "radiologist_qc/radiologist_review.xlsx"
    analysis_csv = run / "radiologist_qc/tables/analysis_ready_airway_branch_morphometry.csv"
    analysis_xlsx = run / "radiologist_qc/tables/analysis_ready_airway_branch_morphometry.xlsx"
    critical, warnings = [], []

    for path in (branches_path, cases_path, workbook, review, analysis_csv, analysis_xlsx):
        if not path.exists() or path.stat().st_size == 0:
            critical.append(f"missing-or-empty:{path}")
    if critical:
        data = {"passed": False, "critical": critical, "warnings": warnings}
        (run / "delivery_validation.json").write_text(json.dumps(data, indent=2))
        print(json.dumps(data, indent=2))
        return 1

    b = pd.read_csv(branches_path)
    c = pd.read_csv(cases_path)
    analysis = pd.read_csv(analysis_csv)
    case_ids = sorted(c.case_id.unique())
    if len(case_ids) != args.expected_cases:
        critical.append(f"case-count:{len(case_ids)}-expected:{args.expected_cases}")
    if c.error.fillna("").astype(bool).any():
        critical.append(f"case-errors:{int(c.error.fillna('').astype(bool).sum())}")
    if b.case_id.nunique() != len(case_ids):
        critical.append(f"branch-table-case-count:{b.case_id.nunique()}-status-count:{len(case_ids)}")
    if len(analysis) != len(b) or analysis.case_id.nunique() != len(case_ids):
        critical.append("analysis-ready-table-does-not-reconcile-with-raw-export")
    if any("fractal" in name.lower() or "complexity" in name.lower()
           for name in analysis.columns):
        critical.append("invalid-complexity-metric-present-in-analysis-ready-table")
    duplicate_count = int(b.duplicated(["case_id", "branch_instance_id"]).sum())
    if duplicate_count:
        critical.append(f"duplicate-branch-ids:{duplicate_count}")
    root_counts = b.groupby("case_id").parent_instance_id.apply(lambda x: x.isna().sum())
    bad_roots = root_counts[root_counts != 1]
    if len(bad_roots):
        critical.append(f"cases-with-root-count-not-one:{bad_roots.to_dict()}")
    nonroot_display_trachea = int(((b.branch_name == "trachea") & b.parent_instance_id.notna()).sum())
    if nonroot_display_trachea:
        critical.append(f"nonroot-display-trachea:{nonroot_display_trachea}")
    disconnected = int((b.centerline_components != 1).sum())
    if disconnected:
        critical.append(f"disconnected-centerlines:{disconnected}")
    if ((b.tortuosity_length_ratio < 1 - 1e-6) | ~np.isfinite(b.tortuosity_length_ratio)).any():
        critical.append("invalid-tortuosity-values")
    if ((b.branch_angle_deg.dropna() < 0) | (b.branch_angle_deg.dropna() > 180)).any():
        critical.append("invalid-angle-values")
    if (b[["radius_min_mm", "radius_mean_mm", "radius_max_mm"]] <= 0).any().any():
        critical.append("nonpositive-radius-values")

    unclassified = int((b.branch_name == "proximal_or_unclassified").sum())
    complexity_na = int((b.complexity_status != "ok").sum())
    warnings.append(f"proximal-or-unclassified-branches:{unclassified};expert-review-required")
    warnings.append(f"complexity-not-estimable-short-branches:{complexity_na};not-imputed")
    warnings.append("pediatric-anatomy-validity:pending-radiologist-review")
    warnings.append("caliber-proxies:descriptive-segmentation-derived-not-clinical-diagnoses")

    missing_videos = []
    manifest = []
    for case_id in case_ids:
        video = run / "radiologist_qc/videos" / f"{case_id}_qc.mp4"
        if not video.exists() or video.stat().st_size < 10_000:
            missing_videos.append(case_id)
        if video.exists():
            manifest.append({"case_id": case_id, "video": str(video),
                             "video_sha256": sha256(video)})
    if missing_videos:
        critical.append(f"missing-videos:{len(missing_videos)}:{','.join(missing_videos)}")

    report = {
        "passed": not critical,
        "expected_cases": args.expected_cases,
        "status_cases": len(case_ids),
        "branch_rows": len(b),
        "critical": critical,
        "warnings": warnings,
        "files": {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in (branches_path, cases_path, workbook, review,
                            analysis_csv, analysis_xlsx)},
        "case_manifest": manifest,
    }
    path = run / "delivery_validation.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "case_manifest"}, indent=2))
    print(f"full report: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
