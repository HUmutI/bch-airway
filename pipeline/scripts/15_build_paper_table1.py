#!/usr/bin/env python3
"""Build a paper-style proximal-airway table from the normal cohort.

The input branch table contains one row per parsed branch.  For every case this
script identifies the root as the trachea, selects the two root children with
the largest descendant trees as the main bronchi, and assigns left/right from
the dominant side of their downstream anatomical labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


RUN = Path(__file__).resolve().parents[1]
DEFAULT_BRANCHES = RUN / "normal_airway_branch_morphometry.csv"
DEFAULT_AGES = RUN / "normal_cases_age.csv"
DEFAULT_SEX = RUN / "normal_subject_sex.csv"
DEFAULT_XLSX = RUN / "paper_table1_airway_dimensions.xlsx"
DEFAULT_CSV = RUN / "paper_table1_airway_dimensions.csv"
DEFAULT_CASE_CSV = RUN / "paper_table1_case_level_source.csv"

AGE_BINS = [(start, start + 2) for start in range(0, 20, 2)]
MEASURES = [
    ("tracheal_length_mm", "Tracheal Length (mm)"),
    ("tracheal_diameter_mm", "Tracheal Diameter (mm)"),
    ("lmb_diameter_mm", "LMB Diameter (mm)"),
    ("rmb_diameter_mm", "RMB Diameter (mm)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, default=DEFAULT_BRANCHES)
    parser.add_argument("--ages", type=Path, default=DEFAULT_AGES)
    parser.add_argument("--sex", type=Path, default=DEFAULT_SEX)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--case-output-csv", type=Path, default=DEFAULT_CASE_CSV)
    return parser.parse_args()


def descendants(start: int, children: dict[int, list[int]]) -> list[int]:
    result, stack = [], [start]
    while stack:
        branch_id = stack.pop()
        result.append(branch_id)
        stack.extend(children.get(branch_id, []))
    return result


def side_votes(row) -> tuple[int, int]:
    left = right = 0
    for column in ("branch_name", "lobar_name", "segmental_name", "subsegmental_name"):
        value = str(getattr(row, column, ""))
        left += int(value.startswith(("Left", "LB")))
        right += int(value.startswith(("Right", "RB")))
    return left, right


def extract_proximal_measurements(branches: pd.DataFrame) -> pd.DataFrame:
    records = []
    for case_id, group in branches.groupby("case_id", sort=True):
        rows = {int(row.branch_instance_id): row for row in group.itertuples()}
        children: dict[int, list[int]] = defaultdict(list)
        roots = []
        for row in group.itertuples():
            branch_id = int(row.branch_instance_id)
            if pd.isna(row.parent_instance_id):
                roots.append(branch_id)
            else:
                children[int(row.parent_instance_id)].append(branch_id)
        if len(roots) != 1:
            raise ValueError(f"{case_id}: expected one root, found {len(roots)}")
        root_id = roots[0]

        candidates = []
        for child_id in children.get(root_id, []):
            subtree = descendants(child_id, children)
            left = right = 0
            for branch_id in subtree:
                l_votes, r_votes = side_votes(rows[branch_id])
                left += l_votes
                right += r_votes
            candidates.append(
                {
                    "branch_id": child_id,
                    "descendant_count": len(subtree),
                    "left_votes": left,
                    "right_votes": right,
                }
            )
        candidates.sort(key=lambda item: item["descendant_count"], reverse=True)
        selected = candidates[:2]
        if len(selected) != 2:
            raise ValueError(f"{case_id}: expected at least two root children")
        for item in selected:
            item["side"] = "L" if item["left_votes"] > item["right_votes"] else "R"
        if {item["side"] for item in selected} != {"L", "R"}:
            raise ValueError(f"{case_id}: main-bronchus side assignment failed: {selected}")

        by_side = {item["side"]: item for item in selected}
        root = rows[root_id]
        lmb = rows[by_side["L"]["branch_id"]]
        rmb = rows[by_side["R"]["branch_id"]]
        records.append(
            {
                "case_id": case_id,
                "subject": root.subject,
                "source_age_group": root.age_group,
                "trachea_branch_id": root_id,
                "lmb_branch_id": int(lmb.branch_instance_id),
                "rmb_branch_id": int(rmb.branch_instance_id),
                "tracheal_length_mm": float(root.centerline_length_mm),
                "tracheal_diameter_mm": float(root.diameter_mean_mm),
                "lmb_diameter_mm": float(lmb.diameter_mean_mm),
                "rmb_diameter_mm": float(rmb.diameter_mean_mm),
                "lmb_descendant_count": by_side["L"]["descendant_count"],
                "rmb_descendant_count": by_side["R"]["descendant_count"],
                "lmb_left_votes": by_side["L"]["left_votes"],
                "lmb_right_votes": by_side["L"]["right_votes"],
                "rmb_left_votes": by_side["R"]["left_votes"],
                "rmb_right_votes": by_side["R"]["right_votes"],
            }
        )
    return pd.DataFrame(records)


def normalize_sex(value) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    value = str(value).strip().lower()
    if value in {"m", "male"}:
        return "Male"
    if value in {"f", "female"}:
        return "Female"
    raise ValueError(f"Unrecognized sex value: {value!r}; use Male/Female or M/F")


def join_metadata(case_data: pd.DataFrame, ages_path: Path, sex_path: Path):
    ages = pd.read_csv(ages_path)
    ages["case_id"] = ages["anon_subject"] + "_" + ages["anon_accession"]
    ages = ages[["case_id", "anon_subject", "age_years"]].rename(
        columns={"anon_subject": "age_subject"}
    )
    if ages.case_id.duplicated().any():
        raise ValueError("Age table contains duplicate case_id values")
    merged = case_data.merge(ages, on="case_id", how="left", validate="one_to_one")
    if merged.age_years.isna().any():
        missing = merged.loc[merged.age_years.isna(), "case_id"].tolist()
        raise ValueError(f"Missing age for cases: {missing}")
    if not (merged.subject == merged.age_subject).all():
        raise ValueError("Subject identifiers disagree between branch and age tables")
    merged = merged.drop(columns="age_subject")

    subjects = pd.DataFrame({"subject": sorted(merged.subject.unique())})
    sex_available = False
    if sex_path.exists():
        sex = pd.read_csv(sex_path, dtype=str)
        required = {"subject", "sex"}
        if not required.issubset(sex.columns):
            raise ValueError(f"Sex file must contain columns {sorted(required)}")
        if sex.subject.duplicated().any():
            raise ValueError("Sex table contains duplicate subjects")
        sex["sex"] = sex["sex"].map(normalize_sex)
        subjects = subjects.merge(sex[["subject", "sex"]], on="subject", how="left")
        sex_available = subjects.sex.notna().any()
    else:
        subjects["sex"] = None
    merged = merged.merge(subjects, on="subject", how="left", validate="many_to_one")
    merged["sex_table_value"] = merged.sex.fillna("All (sex unavailable)")
    return merged, subjects, sex_available


def age_label(low: int, high: int) -> str:
    return f"≥ {low} to < {high}"


def median_iqr_width(series: pd.Series) -> str:
    q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
    return f"{median:.1f} ({q3 - q1:.1f})"


def median_quartiles(series: pd.Series) -> str:
    q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
    return f"{median:.1f} ({q1:.1f}, {q3:.1f})"


def summarize(case_data: pd.DataFrame, sex_available: bool) -> pd.DataFrame:
    rows = []
    if sex_available:
        sex_levels = ["Male", "Female"]
        if case_data.sex.isna().any():
            sex_levels.append("Unknown")
        data = case_data.copy()
        data["sex_table_value"] = data.sex.fillna("Unknown")
    else:
        sex_levels = ["All (sex unavailable)"]
        data = case_data

    groups = AGE_BINS + [(None, None)]
    for low, high in groups:
        age_range = "All" if low is None else age_label(low, high)
        age_subset = data if low is None else data[(data.age_years >= low) & (data.age_years < high)]
        for sex in sex_levels:
            subset = age_subset[age_subset.sex_table_value == sex]
            row = {
                "Age, Range (y)": age_range,
                "Sex": sex,
                "No. of Subjects": int(subset.subject.nunique()),
                "No. of Scans": int(len(subset)),
            }
            if len(subset):
                row["Age (y)"] = median_iqr_width(subset.age_years)
                for source, label in MEASURES:
                    row[label] = median_quartiles(subset[source])
            else:
                row["Age (y)"] = ""
                for _, label in MEASURES:
                    row[label] = ""
            rows.append(row)
    return pd.DataFrame(rows)


def style_workbook(path: Path, table_rows: int) -> None:
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    table = workbook["Table 1"]
    dark = "1F4E78"
    light = "D9EAF7"
    thin_blue = Side(style="thin", color="9EADBA")
    table.merge_cells(start_row=1, start_column=1, end_row=1, end_column=9)
    table["A1"] = (
        "TABLE 1. Dimensions of the Trachea and the Left (LMB) and Right (RMB) "
        "Main Bronchi by Age Group"
    )
    table["A1"].font = Font(bold=True, size=13, color="FFFFFF")
    table["A1"].fill = PatternFill("solid", fgColor=dark)
    table["A1"].alignment = Alignment(vertical="center", wrap_text=True)
    table.row_dimensions[1].height = 34
    header_row = 3
    first_data_row = 4
    last_data_row = header_row + table_rows
    for cell in table[header_row]:
        cell.fill = PatternFill("solid", fgColor=dark)
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=thin_blue)
    table.row_dimensions[header_row].height = 44
    for row in range(first_data_row, last_data_row + 1):
        if table.cell(row, 1).value == "All":
            for cell in table[row]:
                cell.fill = PatternFill("solid", fgColor=light)
                cell.font = Font(bold=True)
        for cell in table[row]:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
    widths = [18, 22, 16, 13, 16, 25, 26, 23, 23]
    for column, width in enumerate(widths, 1):
        table.column_dimensions[get_column_letter(column)].width = width
    table.freeze_panes = "A4"
    table.auto_filter.ref = f"A{header_row}:I{last_data_row}"

    footnotes = [
        "Note—Age is median with interquartile range in parentheses. Dimensions are median with first and third quartiles in parentheses.",
        "Tracheal length is the physical centerline length of the root parsed branch. It is not explicitly clipped at the thoracic inlet.",
        "Tracheal, LMB, and RMB diameters are mean physical EDT-derived centerline diameters; these differ from the source paper's mean of shortest and longest cross-sectional diameters.",
        "The current workspace has no sex metadata. Descriptive medians use all scans; No. of Subjects is the distinct-subject count and repeated scans are not independent.",
    ]
    for offset, note in enumerate(footnotes, start=last_data_row + 2):
        table.merge_cells(start_row=offset, start_column=1, end_row=offset, end_column=9)
        table.cell(offset, 1, note)
        table.cell(offset, 1).font = Font(italic=True, size=9, color="404040")
        table.cell(offset, 1).alignment = Alignment(wrap_text=True, vertical="top")
        table.row_dimensions[offset].height = 30

    for sheet in workbook.worksheets[1:]:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor=dark)
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(wrap_text=True)
        for column_cells in sheet.columns:
            max_length = min(55, max(len(str(cell.value or "")) for cell in column_cells) + 2)
            sheet.column_dimensions[get_column_letter(column_cells[0].column)].width = max(12, max_length)
    workbook.save(path)


def main() -> None:
    args = parse_args()
    branches = pd.read_csv(
        args.branches,
        dtype={"branch_instance_id": "Int64", "parent_instance_id": "Int64"},
    )
    case_data = extract_proximal_measurements(branches)
    case_data, subjects, sex_available = join_metadata(case_data, args.ages, args.sex)
    summary = summarize(case_data, sex_available)

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    case_data.to_csv(args.case_output_csv, index=False)

    template = subjects[["subject", "sex"]].copy()
    notes = pd.DataFrame(
        [
            ("Cohort", f"{len(case_data)} normal scans from {case_data.subject.nunique()} unique subjects."),
            ("Age", "Median (interquartile range); bins are left-closed and right-open."),
            ("Dimensions", "Median (first quartile, third quartile)."),
            ("Trachea", "The single root parsed branch; length is physical centerline length."),
            ("Main bronchi", "The two root children with the largest descendant trees; side is assigned by majority downstream left/right anatomical-label votes."),
            ("Diameter", "Mean physical EDT-derived centerline diameter from the fused airway mask. This is not the paper's mean of shortest and longest cross-sectional diameters."),
            ("Repeated subjects", "The table reports both unique subjects and scans. Descriptive medians use all scans; repeated scans are not statistically independent."),
            ("Sex", "No sex metadata was found in the workspace. Fill normal_subject_sex.csv (subject, sex) and rerun to obtain Male/Female rows."),
            ("Validation status", "AirMorph anatomical predictions are not yet externally validated for this pediatric cohort; the root and graph topology are deterministic, while downstream labels are used only for side assignment."),
        ],
        columns=["Topic", "Definition / note"],
    )
    qc = pd.DataFrame(
        [
            ("Cases in branch table", branches.case_id.nunique()),
            ("Cases summarized", len(case_data)),
            ("Unique subjects", case_data.subject.nunique()),
            ("Cases with one root", len(case_data)),
            ("Cases with opposed L/R main-bronchus assignment", len(case_data)),
            ("Missing ages", int(case_data.age_years.isna().sum())),
            ("Sex metadata available", sex_available),
        ],
        columns=["Check", "Result"],
    )
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Table 1", index=False, startrow=2)
        case_data.to_excel(writer, sheet_name="Case-level Source", index=False)
        template.to_excel(writer, sheet_name="Sex Metadata Template", index=False)
        notes.to_excel(writer, sheet_name="Methods and Notes", index=False)
        qc.to_excel(writer, sheet_name="QC", index=False)
    style_workbook(args.output_xlsx, len(summary))

    # Keep a simple editable metadata file beside the outputs.
    template.to_csv(args.sex, index=False)
    print(f"Wrote {args.output_xlsx}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.case_output_csv}")
    print(f"Wrote sex template {args.sex}")


if __name__ == "__main__":
    main()
