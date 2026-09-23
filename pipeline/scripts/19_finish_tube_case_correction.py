#!/usr/bin/env python3
"""Relabel corrected tube case, update master exports, QC, and balanced report."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pandas as pd


BASE = Path(os.environ.get("OR3_BASE", "/fileserver/external/quin/ege"))
RUN = BASE / "or_all3_run"
BAL = BASE / "BALANCED_COMBINED_90_2026-09-15"
CASE_ID = "sub-0534_acc-0193"
CASE = BASE / "airway-seg/age_01-02/sub-0534/sub-0534_acc-0193"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    morph = load("tube_morph", RUN / "scripts/09_airmorph_morphometry.py")
    print("reclassifying corrected mask", flush=True)
    classification = morph.classify_case(CASE, force=True)
    rows = morph.measure_case(CASE)
    if not rows:
        raise RuntimeError("corrected case produced no morphometry rows")
    maximum_generation = max(int(row["generation"]) for row in rows)
    print(f"corrected case: {len(rows)} branches, maximum generation {maximum_generation}", flush=True)

    branch_path = RUN / "normal_airway_branch_morphometry.csv"
    status_path = RUN / "normal_airway_morphometry_status.csv"
    branches = pd.read_csv(branch_path, low_memory=False)
    combined = pd.concat(
        [branches[~branches.case_id.eq(CASE_ID)], pd.DataFrame(rows)], ignore_index=True
    ).sort_values(["case_id", "branch_instance_id"])
    status = pd.read_csv(status_path).fillna("")
    idx = status.index[status.case_id.eq(CASE_ID)]
    status.loc[idx, "classification"] = classification
    status.loc[idx, "branches"] = len(rows)
    status.loc[idx, "proximal_or_unclassified_branches"] = sum(
        row["branch_name"] == "proximal_or_unclassified" for row in rows
    )
    status.loc[idx, "complexity_not_estimable_branches"] = sum(
        row["complexity_status"] != "ok" for row in rows
    )
    status.loc[idx, "geometry_flagged_branches"] = sum(
        bool(row["geometry_qc_flags"]) for row in rows
    )
    status.loc[idx, "error"] = ""
    branch_records = combined.where(pd.notna(combined), None).to_dict("records")
    status_records = status.where(pd.notna(status), None).to_dict("records")
    morph.write_csv(branch_path, branch_records)
    morph.write_csv(status_path, status_records, list(status.columns))
    morph.write_excel(RUN / "normal_airway_branch_morphometry.xlsx", branch_records, status_records)

    meeting = load("tube_meeting", RUN / "scripts/16_build_meeting_package.py")
    topology = meeting.extract_central_topology(pd.DataFrame(rows))
    ages = pd.read_csv(RUN / "normal_cases_age.csv")
    ages["case_id"] = ages.anon_subject + "_" + ages.anon_accession
    one = topology.merge(
        ages.loc[ages.case_id.eq(CASE_ID), ["case_id", "age_years", "age_bucket"]],
        on="case_id", validate="one_to_one",
    ).merge(status[["case_id", "case_dir"]], on="case_id", validate="one_to_one")
    imaging = meeting.measure_imaging_case(next(one.itertuples()))
    cache_path = RUN / "AIRWAY_RADIOLOGIST_PROF_MEETING_2026-09-11/csv/imaging_measurement_cache.csv"
    cache = pd.read_csv(cache_path)
    cache = pd.concat(
        [cache[~cache.case_id.eq(CASE_ID)], pd.DataFrame([imaging])], ignore_index=True
    ).sort_values("case_id")
    cache.to_csv(cache_path, index=False)

    qc = load("tube_qc", RUN / "scripts/10_radiologist_qc.py")
    video, _ = qc.make_video(CASE, RUN / "radiologist_qc", frames=72, fps=12, force=True)
    print(f"updated QC video: {video}", flush=True)

    subprocess.run(
        [str(RUN / "envs/airmorph/bin/python"), str(BAL / "scripts/build_balanced_inputs_and_package.py")],
        check=True, cwd=BAL,
    )
    subprocess.run(
        [str(RUN / "envs/airmorph/bin/python"), str(BAL / "scripts/17_build_balanced_latex_report.py")],
        check=True, cwd=BAL,
    )
    out = BAL / "BALANCED_AIRWAY_MEETING_RESULTS_90_2026-09-15"
    tectonic = BAL / "tools/tectonic-musl/tectonic"
    subprocess.run([str(tectonic), "airway_meeting_report.tex", "--keep-logs"], check=True, cwd=out)
    shutil.copy2(out / "airway_meeting_report.pdf", out / "FINAL_balanced_airway_meeting_report.pdf")
    archive = BAL / "BALANCED_AIRWAY_MEETING_RESULTS_90_2026-09-15.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(out, arcname=out.name)
    (CASE / "tube_correction_complete.txt").write_text(
        f"corrected branches={len(rows)}\ncorrected maximum generation={maximum_generation}\n"
    )
    print(f"complete; final PDF: {out / 'FINAL_balanced_airway_meeting_report.pdf'}", flush=True)


if __name__ == "__main__":
    main()
