#!/usr/bin/env python3
"""Select a balanced 90-scan cohort by integer programming.

Hard constraints
  * exactly 10 scans in each of the nine two-year age groups;
  * at most one scan per patient across the whole cohort, so 90 scans come from
    90 distinct patients and no patient appears in two age groups;
  * every selected scan has a recorded sex.

Soft objectives, in priority order
  * sex balance: five male and five female in each age group;
  * age balance: within each age group, an even split between the lower and the
    upper half of the group's age span, so the group median sits near its centre;
  * continuity: where the first two objectives are indifferent, keep the scan
    that the previous selection already used.

Two age groups contain exactly ten eligible patients and are therefore fully
determined; the solver reports the shortfall rather than relaxing a hard
constraint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp, Bounds

BASE = Path("/fileserver/external/quin/ege")
RUN = Path(__file__).resolve().parents[1]
PREVIOUS = BASE / "_archive_2026-09-26" / "BALANCED_COMBINED_90_2026-09-15"
E3_EXCLUDED = {"E3_0026"}  # implausible 5 mm slice spacing, never analysed
CENTRAL_AIRWAY_QC_EXCLUDED = {
    "E3_0002", "sub-0261_acc-0081", "sub-0261_acc-0029", "sub-0359_acc-0369",
    "sub-0435_acc-0491", "E3_0022",
    "E3_0024", "E3_0030", "E3_0057", "E3_0062", "E3_0069",
}
CENTRAL_AIRWAY_QC_EXCLUDED_SUBJECTS = {"sub-0261"}

BUCKETS = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-14", "14-16", "16-18"]
BUCKET_EDGES = {bucket: (float(bucket.split("-")[0]), float(bucket.split("-")[1]))
                for bucket in BUCKETS}
PER_BUCKET = 10
SEX_TARGET = 5.0
AGE_TARGET = 5.0

WEIGHT_SEX = 10.0
WEIGHT_AGE = 4.0
WEIGHT_CONTINUITY = 0.05

OUT = RUN / "cohort_selection_90.csv"
REPORT = RUN / "cohort_selection_report.csv"


def build_pool() -> pd.DataFrame:
    """Every processed scan that could enter the cohort, with age and sex."""
    sex_table = pd.read_csv(BASE / "anon_subject_sex.csv")
    sex_lookup = {
        (row.anon_subject, row.anon_accession): row.sex
        for row in sex_table.itertuples()
    }

    original = pd.read_excel(
        BASE / "or_all3_run/normal_cases_age_distribution.xlsx", sheet_name="Cases"
    )
    original = pd.DataFrame({
        "case_id": original.anon_subject + "_" + original.anon_accession,
        "subject": original.anon_subject,
        "source_accession": original.anon_accession,
        "age_years": original.age_years.astype(float),
        "age_bucket": original.age_bucket,
        "sex": [
            sex_lookup.get((row.anon_subject, row.anon_accession), "Unknown")
            for row in original.itertuples()
        ],
        "source_cohort": "original_83",
    })

    e3 = pd.read_csv(BASE / "E3_PEDIATRIC_AIRWAY_RESULTS/e3_cases_age_sex.csv")
    e3 = e3[~e3.case_id.isin(E3_EXCLUDED)].copy()
    e3 = pd.DataFrame({
        "case_id": e3.case_id,
        "subject": e3.subject,
        "source_accession": "",
        "age_years": e3.age_years.astype(float),
        "age_bucket": pd.cut(
            e3.age_years, bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18.000001],
            labels=BUCKETS, right=False,
        ).astype(str),
        "sex": e3.sex,
        "source_cohort": "E3",
    })

    pool = pd.concat([original, e3], ignore_index=True)
    pool = pool[~pool.case_id.isin(CENTRAL_AIRWAY_QC_EXCLUDED)].copy()
    pool = pool[~pool.subject.isin(CENTRAL_AIRWAY_QC_EXCLUDED_SUBJECTS)].copy()
    pool["sex"] = pool.sex.fillna("Unknown").replace("", "Unknown")
    pool = pool[pool.age_bucket.isin(BUCKETS)].copy()

    excluded_unknown = int(pool.sex.eq("Unknown").sum())
    pool = pool[pool.sex.isin(["Male", "Female"])].copy()
    print(f"eligible scans with a recorded sex: {len(pool)} "
          f"({excluded_unknown} excluded for unknown sex)")

    # Lower half of the age group, used for the within-group age balance.
    low, high = zip(*[BUCKET_EDGES[b] for b in pool.age_bucket])
    midpoint = (np.asarray(low) + np.asarray(high)) / 2.0
    pool["lower_half"] = (pool.age_years < midpoint).astype(int)

    previous = set(pd.read_csv(PREVIOUS / "balanced_cases.csv").case_id)
    pool["previously_selected"] = pool.case_id.isin(previous).astype(int)
    return pool.reset_index(drop=True)


def check_feasibility(pool: pd.DataFrame) -> None:
    print("\navailable distinct patients per age group:")
    shortfall = []
    for bucket in BUCKETS:
        group = pool[pool.age_bucket == bucket]
        subjects = group.subject.nunique()
        male = group[group.sex == "Male"].subject.nunique()
        female = group[group.sex == "Female"].subject.nunique()
        flag = "" if subjects >= PER_BUCKET else "  <-- TOO FEW"
        if subjects < PER_BUCKET:
            shortfall.append(bucket)
        print(f"  {bucket:>6}: {subjects:3d} patients (male {male:3d}, female {female:3d}){flag}")
    if shortfall:
        raise SystemExit(
            f"cannot fill {', '.join(shortfall)} with {PER_BUCKET} known-sex patients"
        )


def solve(pool: pd.DataFrame) -> pd.DataFrame:
    """Choose the cohort with a mixed-integer program."""
    n = len(pool)
    subjects = sorted(pool.subject.unique())
    subject_index = {subject: i for i, subject in enumerate(subjects)}

    # Variables: one binary per candidate scan, then per bucket two deviation
    # variables for sex and two for age balance.
    n_dev = 4 * len(BUCKETS)
    total = n + n_dev

    def sex_dev(b: int) -> int:
        return n + 2 * b

    def age_dev(b: int) -> int:
        return n + 2 * len(BUCKETS) + 2 * b

    cost = np.zeros(total)
    cost[:n] = -WEIGHT_CONTINUITY * pool.previously_selected.to_numpy()
    for b in range(len(BUCKETS)):
        cost[sex_dev(b)] = cost[sex_dev(b) + 1] = WEIGHT_SEX
        cost[age_dev(b)] = cost[age_dev(b) + 1] = WEIGHT_AGE

    constraints = []

    # Exactly ten scans per age group.
    rows = np.zeros((len(BUCKETS), total))
    for b, bucket in enumerate(BUCKETS):
        rows[b, :n] = (pool.age_bucket == bucket).to_numpy().astype(float)
    constraints.append(LinearConstraint(rows, PER_BUCKET, PER_BUCKET))

    # At most one scan per patient, cohort wide.
    rows = np.zeros((len(subjects), total))
    for position, subject in enumerate(pool.subject):
        rows[subject_index[subject], position] = 1.0
    constraints.append(LinearConstraint(rows, 0, 1))

    # Sex balance: males - 5 = dev_plus - dev_minus, both non-negative.
    rows = np.zeros((len(BUCKETS), total))
    for b, bucket in enumerate(BUCKETS):
        rows[b, :n] = ((pool.age_bucket == bucket) & (pool.sex == "Male")).to_numpy().astype(float)
        rows[b, sex_dev(b)] = -1.0
        rows[b, sex_dev(b) + 1] = 1.0
    constraints.append(LinearConstraint(rows, SEX_TARGET, SEX_TARGET))

    # Age balance: lower-half count - 5 = dev_plus - dev_minus.
    rows = np.zeros((len(BUCKETS), total))
    for b, bucket in enumerate(BUCKETS):
        rows[b, :n] = (
            (pool.age_bucket == bucket) & (pool.lower_half == 1)
        ).to_numpy().astype(float)
        rows[b, age_dev(b)] = -1.0
        rows[b, age_dev(b) + 1] = 1.0
    constraints.append(LinearConstraint(rows, AGE_TARGET, AGE_TARGET))

    integrality = np.zeros(total)
    integrality[:n] = 1
    bounds = Bounds(lb=np.zeros(total), ub=np.r_[np.ones(n), np.full(n_dev, PER_BUCKET)])

    result = milp(c=cost, constraints=constraints, integrality=integrality, bounds=bounds)
    if not result.success:
        raise SystemExit(f"selection did not solve: {result.message}")

    chosen = pool[np.round(result.x[:n]).astype(bool)].copy()
    if len(chosen) != PER_BUCKET * len(BUCKETS):
        raise SystemExit(f"solver returned {len(chosen)} scans")
    if chosen.subject.duplicated().any():
        raise SystemExit("solver returned a repeated patient")
    return chosen.sort_values(["age_bucket", "case_id"]).reset_index(drop=True)


def report(chosen: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    print("\nselected cohort:")
    for bucket in BUCKETS:
        group = chosen[chosen.age_bucket == bucket]
        available = pool[pool.age_bucket == bucket]
        male = int((group.sex == "Male").sum())
        lower = int(group.lower_half.sum())
        forced = available.subject.nunique() == PER_BUCKET
        rows.append({
            "age_bucket": bucket,
            "scans": len(group),
            "male": male,
            "female": len(group) - male,
            "lower_half_of_age_span": lower,
            "upper_half_of_age_span": len(group) - lower,
            "median_age_years": round(float(group.age_years.median()), 2),
            "distinct_patients_available": int(available.subject.nunique()),
            "selection_fully_forced": forced,
            "from_original_cohort": int((group.source_cohort == "original_83").sum()),
            "from_e3_cohort": int((group.source_cohort == "E3").sum()),
        })
        note = "  (only 10 eligible patients, no choice)" if forced else ""
        print(f"  {bucket:>6}: {male}M/{len(group) - male}F, "
              f"age split {lower}/{len(group) - lower}, "
              f"median {group.age_years.median():.1f}y{note}")

    summary = pd.DataFrame(rows)
    male_total = int((chosen.sex == "Male").sum())
    print(f"\ntotal: {len(chosen)} scans, {chosen.subject.nunique()} patients, "
          f"{male_total} male / {len(chosen) - male_total} female, "
          f"0 unknown sex")
    retained = int(chosen.previously_selected.sum())
    print(f"kept {retained} of the previous selection's scans, replaced {len(chosen) - retained}")
    return summary


def main() -> None:
    pool = build_pool()
    check_feasibility(pool)
    chosen = solve(pool)
    summary = report(chosen, pool)

    chosen["patient_identity_basis"] = np.where(
        chosen.source_cohort == "original_83",
        "anon_subject",
        "unique E3 ID (source manifest has no separate patient field)",
    )
    chosen["selection_reason"] = (
        "integer-programming selection: 10 per age group, one scan per patient "
        "cohort wide, recorded sex required, sex and within-group age balance optimized"
    )
    columns = [
        "case_id", "subject", "age_years", "age_bucket", "sex", "source_cohort",
        "source_accession", "lower_half", "previously_selected",
        "patient_identity_basis", "selection_reason",
    ]
    chosen[columns].to_csv(OUT, index=False)
    summary.to_csv(REPORT, index=False)
    print(f"\nwrote {OUT}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
