#!/usr/bin/env python3
"""Measure the selected 90-scan cohort with the cross-sectional convention.

Reads the cohort chosen by 00_select_cohort.py and its central-airway topology,
then measures every scan again using perpendicular cross-sections and a
lung-apex thoracic-inlet proxy. Results are cached so the reporting step can be
re-run without repeating the imaging work.
"""
from __future__ import annotations

import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import axis_geometry as ag

RUN = Path(__file__).resolve().parents[1]
BASE = Path("/fileserver/external/quin/ege")
BUCKETS = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-14", "14-16", "16-18"]

CACHE = RUN / "cross_section_measurements.csv"
FAILURES = RUN / "cross_section_failures.csv"
COHORT = RUN / "cohort_90.csv"


SOURCES = [
    (BASE / "or_all3_run/normal_airway_branch_morphometry.csv",
     BASE / "or_all3_run/normal_airway_morphometry_status.csv"),
    (BASE / "E3_PEDIATRIC_AIRWAY_RESULTS/normal_airway_branch_morphometry.csv",
     BASE / "E3_PEDIATRIC_AIRWAY_RESULTS/normal_airway_morphometry_status.csv"),
]


def derive_topology(case_ids: set[str]) -> pd.DataFrame:
    """Identify the trachea and both main bronchi from the live branch tables.

    Topology is recomputed here rather than read from an earlier package,
    because re-running AirMorph on a case renumbers its branches. A case that
    has since been re-labelled would otherwise be measured against stale
    branch identifiers.
    """
    import importlib.util

    source = BASE / "or_all3_run/scripts/16_build_meeting_package.py"
    spec = importlib.util.spec_from_file_location("meeting_package", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    frames = []
    for branch_path, status_path in SOURCES:
        branches = pd.read_csv(
            branch_path, low_memory=False,
            dtype={"branch_instance_id": "Int64", "parent_instance_id": "Int64"},
        )
        branches = branches[branches.case_id.isin(case_ids)]
        if not len(branches):
            continue
        topology = module.extract_central_topology(branches)
        status = pd.read_csv(status_path)[["case_id", "case_dir"]]
        frames.append(topology.merge(status, on="case_id", validate="one_to_one"))

    topology = pd.concat(frames, ignore_index=True)
    # Radiologist/QC review of the tube-corrected case established that branch
    # 200 is the dominant LMB fragment; the classifier had selected stub 199.
    topology.loc[topology.case_id.eq("sub-0534_acc-0193"), "lmb_branch_id"] = 200
    missing = case_ids - set(topology.case_id)
    if missing:
        raise SystemExit(f"no central-airway topology for: {sorted(missing)}")
    return topology


def source_fingerprint(case_dir: str, case_id: str) -> str:
    """Modification times of the inputs a measurement depends on.

    Used to invalidate cached measurements when a case is re-segmented or
    re-labelled, so corrected cases are never served from a stale cache.
    """
    directory = Path(case_dir)
    stamps = []
    for name in (f"{case_id}_skel_parsing.nii.gz", "union3_lcc.nii.gz", "lunglobe.nii.gz"):
        path = directory / name
        stamps.append(f"{name}:{int(path.stat().st_mtime) if path.exists() else 0}")
    return "|".join(stamps)


def load_cohort() -> pd.DataFrame:
    """Load the optimized selection and attach freshly derived topology."""
    selection = pd.read_csv(RUN / "cohort_selection_90.csv")
    topology = derive_topology(set(selection.case_id))
    columns = [
        "case_id", "case_dir", "root_branch_id", "lmb_branch_id", "rmb_branch_id",
        "main_bronchus_assignment_method", "lmb_side_vote_margin", "rmb_side_vote_margin",
        "raw_root_mst_length_mm", "raw_root_edt_mean_diameter_mm",
        "lmb_edt_mean_diameter_mm", "rmb_edt_mean_diameter_mm",
    ]
    cohort = selection.merge(
        topology[columns].drop_duplicates("case_id"), on="case_id", validate="one_to_one"
    )
    cohort["source_fingerprint"] = [
        source_fingerprint(row.case_dir, row.case_id) for row in cohort.itertuples()
    ]
    corrected = cohort.case_id.eq("sub-0534_acc-0193")
    cohort.loc[corrected, "source_fingerprint"] += (
        "|topology:" + cohort.loc[corrected, "root_branch_id"].astype(str) + ","
        + cohort.loc[corrected, "lmb_branch_id"].astype(str) + ","
        + cohort.loc[corrected, "rmb_branch_id"].astype(str)
    )

    if len(cohort) != 90:
        raise SystemExit(f"expected 90 scans, found {len(cohort)}")
    if cohort.subject.nunique() != 90:
        raise SystemExit(f"expected 90 unique subjects, found {cohort.subject.nunique()}")
    counts = cohort.groupby("age_bucket").size().reindex(BUCKETS)
    if not counts.eq(10).all():
        raise SystemExit(f"cohort is not balanced: {counts.to_dict()}")
    if cohort.case_id.duplicated().any():
        raise SystemExit("duplicate case_id in the selection")
    if not cohort.sex.isin(["Male", "Female"]).all():
        raise SystemExit("selection contains a scan without a recorded sex")
    return cohort


def measure(record: dict) -> dict:
    """Worker entry point: measure one scan or return its failure reason."""
    try:
        result = ag.measure_case(
            record["case_id"],
            Path(record["case_dir"]),
            int(record["root_branch_id"]),
            int(record["lmb_branch_id"]),
            int(record["rmb_branch_id"]),
        )
        result["measurement_status"] = "ok"
        result["source_fingerprint"] = record["source_fingerprint"]
        return result
    except Exception as error:  # noqa: BLE001 - reported per case, never silent
        return {
            "case_id": record["case_id"],
            "measurement_status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    cohort = load_cohort()
    cohort.to_csv(COHORT, index=False)
    records = cohort[
        ["case_id", "case_dir", "root_branch_id", "lmb_branch_id", "rmb_branch_id",
         "source_fingerprint"]
    ].to_dict("records")

    results: list[dict] = []
    failures: list[dict] = []
    if CACHE.exists():
        cached = pd.read_csv(CACHE)
        fingerprints = dict(zip(cohort.case_id, cohort.source_fingerprint))
        if "source_fingerprint" in cached.columns:
            current = cached.case_id.map(fingerprints)
            fresh = cached[cached.source_fingerprint.eq(current)]
            stale = sorted(set(cached.case_id) - set(fresh.case_id) & set(fingerprints))
            if stale:
                print(f"re-measuring {len(stale)} case(s) whose inputs changed: "
                      f"{', '.join(stale)}")
        else:
            fresh = cached.iloc[0:0]
            print("cache has no fingerprints; measuring every case")
        reusable = fresh[fresh.case_id.isin(set(cohort.case_id))]
        if len(reusable):
            results.extend(reusable.to_dict("records"))
            done = set(reusable.case_id)
            records = [r for r in records if r["case_id"] not in done]
            print(f"reusing {len(done)} cached measurements, "
                  f"measuring {len(records)} scans")

    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(measure, record): record["case_id"] for record in records}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            case_id = result["case_id"]
            if result["measurement_status"] == "ok":
                results.append(result)
                print(
                    f"[{done:02d}/{len(records)}] {case_id}: "
                    f"intrathoracic trachea {result['intrathoracic_tracheal_length_mm']:.1f} mm, "
                    f"mean-axis diameter {result['trachea_mean_axis_diameter_mm']:.2f} mm, "
                    f"{result['trachea_cross_sections']} cross-sections",
                    flush=True,
                )
            else:
                failures.append(result)
                print(f"[{done:02d}/{len(records)}] {case_id}: FAILED {result['error']}", flush=True)

    if failures:
        pd.DataFrame(failures).to_csv(FAILURES, index=False)
    measurements = pd.DataFrame(results).sort_values("case_id").reset_index(drop=True)
    measurements.to_csv(CACHE, index=False)

    print(f"\ncohort measurements available: {len(measurements)}/90")
    if failures:
        print(f"failures written to {FAILURES}")
        raise SystemExit(f"{len(failures)} scans could not be measured")
    print(f"wrote {CACHE}")


if __name__ == "__main__":
    main()
