#!/usr/bin/env python3
"""Resolve patient sex for the anonymized cohort without exposing identifiers.

The atlas workbooks are keyed by real accession number, and the anonymization
mapping links each real accession to its anonymized identifier. This script
joins the two in memory and writes out only the anonymized subject, the
anonymized accession, and the sex character.

Nothing derived from a real accession number, medical record number, patient
name, or date of birth is written to the output file, printed to the terminal,
or recorded in any log. Unmatched cases are reported by anonymized identifier
only. The real identifier columns are dropped as soon as the join completes.

Run this on a host that can read both source locations:

    python3 fetch_patient_sex.py \
        --mapping <path-to>/mapping.csv \
        --atlas /fileserver/external/quin/liam/ct/atlas_subjects \
        --out /fileserver/external/quin/ege/anon_subject_sex.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Column headers as they appear in the atlas workbooks, matched case- and
# whitespace-insensitively.
ACCESSION_HEADERS = {"accession number", "accession", "accession no", "accession_number"}
SEX_HEADERS = {"patient sex", "sex", "patient_sex", "gender"}

SEX_VALUES = {
    "m": "Male", "male": "Male",
    "f": "Female", "female": "Female",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True,
                        help="anonymization mapping CSV")
    parser.add_argument("--atlas", type=Path, required=True,
                        help="directory holding the atlas .xlsx workbooks")
    parser.add_argument("--out", type=Path, required=True,
                        help="output CSV: anon_subject, anon_accession, sex")
    return parser.parse_args()


def normalize_key(value) -> str:
    """Reduce an accession to a comparison key.

    Only ever used as a dictionary key inside this process. Digits and letters
    are kept, separators dropped, and leading zeros removed so that values
    stored as text and as numbers compare equal.
    """
    if pd.isna(value):
        return ""
    text = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    stripped = text.lstrip("0")
    return stripped or text


def normalize_sex(value) -> str:
    if pd.isna(value):
        return ""
    return SEX_VALUES.get(str(value).strip().lower(), "")


def find_column(frame: pd.DataFrame, candidates: set[str]) -> str | None:
    for column in frame.columns:
        if str(column).strip().lower() in candidates:
            return str(column)
    return None


def read_workbook(path: Path) -> pd.DataFrame:
    """Return accession key and sex from one workbook, or an empty frame.

    Some workbooks carry a title row above the real header, so each sheet is
    retried with the first few rows treated as the header.
    """
    records: list[pd.DataFrame] = []
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    except Exception as error:  # noqa: BLE001 - one bad workbook must not stop the run
        print(f"  skipped {path.name}: {type(error).__name__}", file=sys.stderr)
        return pd.DataFrame(columns=["key", "sex"])

    for sheet_name, raw in sheets.items():
        if raw.empty:
            continue
        for header_row in range(min(5, len(raw))):
            frame = raw.iloc[header_row + 1:].copy()
            frame.columns = [str(v) for v in raw.iloc[header_row]]
            accession_column = find_column(frame, ACCESSION_HEADERS)
            sex_column = find_column(frame, SEX_HEADERS)
            if not accession_column or not sex_column:
                continue
            subset = pd.DataFrame({
                "key": frame[accession_column].map(normalize_key),
                "sex": frame[sex_column].map(normalize_sex),
            })
            subset = subset[subset.key.ne("") & subset.sex.ne("")]
            if len(subset):
                records.append(subset)
                print(f"  {path.name} [{sheet_name}]: {len(subset)} sex values")
            break
    if not records:
        return pd.DataFrame(columns=["key", "sex"])
    return pd.concat(records, ignore_index=True)


def main() -> None:
    args = parse_args()
    if not args.mapping.exists():
        raise SystemExit(f"mapping file not found: {args.mapping}")
    if not args.atlas.is_dir():
        raise SystemExit(f"atlas directory not found: {args.atlas}")

    mapping = pd.read_csv(args.mapping, dtype=object)
    for column in ("accession", "anon_subject", "anon_accession"):
        if column not in mapping.columns:
            raise SystemExit(f"mapping is missing the {column!r} column")
    mapping = mapping[["accession", "anon_subject", "anon_accession"]].copy()
    mapping["key"] = mapping.accession.map(normalize_key)
    # The real accession is no longer needed once the join key exists.
    mapping = mapping.drop(columns=["accession"])
    print(f"mapping rows: {len(mapping)}")

    print(f"reading workbooks in {args.atlas}")
    frames = [read_workbook(path) for path in sorted(args.atlas.glob("*.xlsx"))
              if not path.name.startswith("~$")]
    atlas = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if atlas.empty:
        raise SystemExit("no accession/sex pairs found in the atlas workbooks")

    # One sex per accession. A key with conflicting values is dropped rather
    # than guessed, and reported as a count only.
    grouped = atlas.groupby("key").sex.nunique()
    conflicts = set(grouped[grouped > 1].index)
    if conflicts:
        print(f"dropped {len(conflicts)} accessions with conflicting sex values")
    atlas = atlas[~atlas.key.isin(conflicts)].drop_duplicates("key")
    print(f"atlas accessions with a usable sex value: {len(atlas)}")

    merged = mapping.merge(atlas, on="key", how="left")
    # Drop the join key so the output cannot be reversed to a real accession.
    result = merged[["anon_subject", "anon_accession", "sex"]].copy()
    result["sex"] = result.sex.fillna("")
    result = result.sort_values(["anon_subject", "anon_accession"]).reset_index(drop=True)

    resolved = int(result.sex.ne("").sum())
    print(f"\nresolved sex for {resolved}/{len(result)} anonymized scans")
    print(result.sex.replace("", "Unknown").value_counts().to_string())

    missing = result[result.sex.eq("")]
    if len(missing):
        print(f"\nno sex found for {len(missing)} anonymized scans:")
        for subject in sorted(missing.anon_subject.unique())[:40]:
            print(f"  {subject}")
        if missing.anon_subject.nunique() > 40:
            print(f"  ... and {missing.anon_subject.nunique() - 40} more subjects")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("This file contains anonymized identifiers and sex only.")


if __name__ == "__main__":
    main()
