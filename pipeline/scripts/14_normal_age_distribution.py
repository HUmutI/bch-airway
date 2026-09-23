#!/usr/bin/env python3
"""Build an age table for the anonymized normal CT cases.

Joins three sources:
  1. a list of anonymized normal case paths (sub-XXXX_acc-YYYY_source.nii.gz)
  2. mapping.csv, which links anon_accession back to the real accession
  3. the per-age-bucket .xlsx files, which carry age against the real accession

Only the anonymized identifiers and the age are written to the output; the real
accession numbers stay in memory and are never printed or saved.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ACC_COL_HINTS = ("accession", "acc", "acc#", "acc_no", "accnum")
AGE_COL_HINTS = ("age",)


def norm_acc(value):
    """Normalize an accession for joining: uppercase, no punctuation, no leading zeros."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if text.endswith(".0"):
        text = text[:-2]
    text = re.sub(r"[^A-Z0-9]", "", text)
    text = text.lstrip("0")
    return text or None


DEFAULT_BUCKETS = "0-3,3-6,6-9,10-12,12-14,14-16,16-18"
# These two buckets claim an age equal to their upper edge; the rest are
# half-open, so an age on a shared edge falls to the upper bucket.
DEFAULT_CLOSED_UPPER = "6-9,16-18"


def parse_buckets(spec):
    """Parse "0-3,3-6,..." into [(label, lower, upper), ...]."""
    buckets = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            lower, upper = (float(x) for x in part.split("-", 1))
        except ValueError:
            sys.exit(f"bad bucket {part!r}: expected the form LOW-HIGH, e.g. 0-3")
        if upper <= lower:
            sys.exit(f"bad bucket {part!r}: the upper edge must exceed the lower")
        buckets.append((part, lower, upper))
    if not buckets:
        sys.exit("no buckets given")
    return buckets


def bucket_of(age, buckets, closed_upper):
    """Label of the first matching bucket, or None.

    A bucket is [low, high) unless its label is in closed_upper, when it is
    [low, high] and claims an age sitting exactly on its upper edge.
    """
    if age is None or pd.isna(age):
        return None
    for label, lower, upper in buckets:
        closed = label in closed_upper
        if age >= lower and (age <= upper if closed else age < upper):
            return label
    return None


def pick_column(columns, hints):
    """Best-effort match of a column name against the hint words."""
    lowered = {c: str(c).strip().lower() for c in columns}
    for col, low in lowered.items():
        if low in hints:
            return col
    for col, low in lowered.items():
        if any(h in low for h in hints):
            return col
    return None


def read_age_tables(xlsx_dir, verbose):
    """Return {normalized accession: age} from every .xlsx in the directory."""
    ages = {}
    conflicts = []
    for path in sorted(Path(xlsx_dir).glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        try:
            sheets = pd.read_excel(path, sheet_name=None, dtype=object)
        except Exception as exc:
            print(f"WARN  could not read {path.name}: {exc}", file=sys.stderr)
            continue
        for sheet_name, frame in sheets.items():
            acc_col = pick_column(frame.columns, ACC_COL_HINTS)
            age_col = pick_column(frame.columns, AGE_COL_HINTS)
            if acc_col is None or age_col is None:
                if verbose:
                    print(f"SKIP  {path.name}[{sheet_name}]: "
                          f"acc={acc_col!r} age={age_col!r} "
                          f"columns={list(frame.columns)}", file=sys.stderr)
                continue
            used = 0
            for acc_raw, age in zip(frame[acc_col], frame[age_col]):
                acc = norm_acc(acc_raw)
                if acc is None or age is None or pd.isna(age):
                    continue
                if acc in ages and str(ages[acc]) != str(age):
                    conflicts.append((acc, ages[acc], age))
                ages[acc] = age
                used += 1
            if verbose:
                print(f"READ  {path.name}[{sheet_name}]: {used} rows "
                      f"(acc={acc_col!r}, age={age_col!r})", file=sys.stderr)
    if conflicts:
        print(f"WARN  {len(conflicts)} accessions had conflicting ages; "
              f"the last value read wins", file=sys.stderr)
    return ages


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normals",
                    default="/fileserver/external/quin/ege/airway-dataset/normal-cases.txt")
    ap.add_argument("--mapping", default="<path-to>/link_mapping.csv")
    ap.add_argument("--xlsx-dir",
                    default="/fileserver/external/quin/liam/ct/atlas_subjects")
    ap.add_argument("--out",
                    default="/fileserver/external/quin/ege/or_all3_run/normal_cases_age.csv")
    ap.add_argument("--report",
                    default="/fileserver/external/quin/ege/or_all3_run/normal_cases_age_distribution.txt",
                    help="plain-text age distribution summary")
    ap.add_argument("--xlsx",
                    default="/fileserver/external/quin/ege/or_all3_run/normal_cases_age_distribution.xlsx",
                    help="spreadsheet of the bucket summary and the per-case rows")
    ap.add_argument("--buckets", default=DEFAULT_BUCKETS,
                    help=f"comma-separated age buckets (default: {DEFAULT_BUCKETS})")
    ap.add_argument("--closed-upper", default=DEFAULT_CLOSED_UPPER,
                    help="buckets that include their upper edge, by label. Every "
                         "other bucket is [low, high), so an age on a shared edge "
                         f"goes to the upper bucket (default: {DEFAULT_CLOSED_UPPER})")
    ap.add_argument("--verbose", action="store_true",
                    help="report the column choice made for every sheet")
    args = ap.parse_args()

    mapping = pd.read_csv(args.mapping, dtype=str)
    missing = {"anon_accession", "accession"} - set(mapping.columns)
    if missing:
        sys.exit(f"mapping.csv is missing column(s): {sorted(missing)}")

    # anon accession -> real accession, normalized on both sides
    anon_to_real = {}
    for anon, real in zip(mapping["anon_accession"], mapping["accession"]):
        key = norm_acc(anon)
        if key:
            anon_to_real[key] = norm_acc(real)

    ages = read_age_tables(args.xlsx_dir, args.verbose)
    print(f"loaded {len(ages)} accession/age pairs from {args.xlsx_dir}",
          file=sys.stderr)

    rows = []
    for line in Path(args.normals).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.search(r"(sub-\d+)_(acc-\d+)", line)
        if not match:
            print(f"WARN  unparsable path: {line}", file=sys.stderr)
            continue
        subject, anon_acc = match.group(1), match.group(2)
        real = anon_to_real.get(norm_acc(anon_acc))
        age = ages.get(real) if real else None
        rows.append({
            "anon_path": line,
            "anon_subject": subject,
            "anon_accession": anon_acc,
            "age_bucket_dir": Path(line).parts[-3] if len(Path(line).parts) >= 3 else "",
            "age": age,
            "join_status": ("ok" if age is not None else
                            "no-mapping" if real is None else "no-age-in-xlsx"),
        })

    table = pd.DataFrame(rows)
    # The xlsx files mix numbers and strings in the age column, so keep the value
    # as written and add a numeric column for sorting and analysis.
    table["age_years"] = pd.to_numeric(table["age"], errors="coerce")
    buckets = parse_buckets(args.buckets)
    closed_upper = {p.strip() for p in args.closed_upper.split(",") if p.strip()}
    unknown = closed_upper - {label for label, _, _ in buckets}
    if unknown:
        sys.exit(f"--closed-upper names bucket(s) not in --buckets: {sorted(unknown)}")
    table["age_bucket"] = [bucket_of(a, buckets, closed_upper)
                           for a in table["age_years"]]
    table.to_csv(args.out, index=False)

    ok = (table["join_status"] == "ok").sum()
    print(f"\nwrote {args.out}: {len(table)} rows, {ok} with an age", file=sys.stderr)
    for status, count in table["join_status"].value_counts().items():
        if status != "ok":
            print(f"  {status}: {count}", file=sys.stderr)

    resolved = table[table["join_status"] == "ok"]
    write_report(table, resolved, buckets, args)


def write_report(table, resolved, buckets, args):
    """Write the 1-year-bucket distribution and summary statistics."""
    lines = []
    add = lines.append
    bucket_rows = []

    add("Age distribution of the anonymized normal CT cases")
    add("=" * 52)
    add("")
    add(f"case list   : {args.normals}")
    add(f"age source  : {args.xlsx_dir}")
    add(f"case table  : {args.out}")
    add(f"cases       : {len(table)} listed, {len(resolved)} with an age")
    add(f"buckets     : {args.buckets}")
    add(f"edges       : [low, high) except {args.closed_upper}, "
        f"which are [low, high]")
    add("")

    unresolved = table[table["join_status"] != "ok"]
    if not unresolved.empty:
        add("Cases with no age:")
        for status, count in unresolved["join_status"].value_counts().items():
            add(f"  {status:<16} {count}")
        add("")

    numeric = resolved["age_years"].dropna()
    unparsed = resolved[resolved["age_years"].isna()]
    if not unparsed.empty:
        add(f"WARNING: {len(unparsed)} age values could not be read as a number "
            f"and are excluded from the statistics below:")
        for value in sorted(set(unparsed["age"].astype(str))):
            add(f"  {value!r}")
        add("")

    if numeric.empty:
        add("No numeric ages available.")
    else:
        add("Summary statistics (years)")
        add("-" * 52)
        add(f"  n          {len(numeric)}")
        add(f"  mean       {numeric.mean():.2f}")
        add(f"  std        {numeric.std(ddof=1):.2f}")
        add(f"  min        {numeric.min():.2f}")
        add(f"  p25        {numeric.quantile(0.25):.2f}")
        add(f"  median     {numeric.median():.2f}")
        add(f"  p75        {numeric.quantile(0.75):.2f}")
        add(f"  max        {numeric.max():.2f}")
        add("")

        add("Distribution by age bucket (years)")
        add("-" * 52)
        add(f"{'bucket':<10}{'scans':>6}{'pct':>8}{'subjects':>10}   histogram")
        total = len(numeric)
        aged = resolved.loc[numeric.index]
        counts = aged["age_bucket"].value_counts()
        for label, _, _ in buckets:
            count = int(counts.get(label, 0))
            # A subject with two scans in one bucket counts once here.
            subjects = aged.loc[aged["age_bucket"] == label, "anon_subject"].nunique()
            pct = 100.0 * count / total
            add(f"{label:<10}{count:>6}{pct:>7.1f}%{subjects:>10}   {'#' * count}")
            bucket_rows.append({"Age bucket (years)": label,
                                "Subjects": int(subjects),
                                "CT scans": count,
                                "Percent of scans": round(pct, 1)})
        all_subjects = aged["anon_subject"].nunique()
        add(f"{'total':<10}{total:>6}{100.0:>7.1f}%{all_subjects:>10}")
        bucket_rows.append({"Age bucket (years)": "total",
                            "Subjects": int(all_subjects),
                            "CT scans": total,
                            "Percent of scans": 100.0})
        if all_subjects != sum(aged.loc[aged["age_bucket"] == label,
                                        "anon_subject"].nunique()
                               for label, _, _ in buckets):
            add("")
            add("Note: the subject counts do not sum to the total because some "
                "subjects have scans in more than one bucket.")

        # Buckets need not be contiguous, so say plainly what fell through.
        outside = numeric[aged["age_bucket"].isna()]
        if not outside.empty:
            add("")
            add(f"NOT IN ANY BUCKET: {len(outside)} case(s), "
                f"ages {sorted(round(float(a), 2) for a in outside)}")
            add("These are excluded from the bucket counts above but included "
                "in the summary statistics.")
        add("")

    add("Counts by dataset age-bucket directory")
    add("-" * 52)
    for bucket, count in resolved["age_bucket_dir"].value_counts().sort_index().items():
        add(f"  {bucket:<14}{count:>4}")
    add("")

    text = "\n".join(lines)
    Path(args.report).write_text(text + "\n")
    print(text)
    print(f"\nwrote {args.report}", file=sys.stderr)

    if bucket_rows:
        write_workbook(bucket_rows, resolved, args)
        print(f"wrote {args.xlsx}", file=sys.stderr)


def write_workbook(bucket_rows, resolved, args):
    """Write the bucket summary, plus the per-case rows behind it."""
    summary = pd.DataFrame(bucket_rows)
    cases = resolved[["anon_subject", "anon_accession", "age", "age_years",
                      "age_bucket", "age_bucket_dir", "anon_path"]].copy()
    cases = cases.sort_values(["age_years", "anon_subject"])
    notes = pd.DataFrame({
        "Field": ["Case list", "Age source", "Buckets", "Edges",
                  "Scans", "Subjects", "Subject counts"],
        "Value": [
            args.normals,
            args.xlsx_dir,
            args.buckets,
            f"[low, high) except {args.closed_upper}, which are [low, high]",
            len(resolved),
            resolved["anon_subject"].nunique(),
            "A subject scanned in two buckets is counted in each, so the "
            "per-bucket subject counts need not sum to the total.",
        ],
    })
    with pd.ExcelWriter(args.xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Distribution", index=False)
        cases.to_excel(writer, sheet_name="Cases", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
        for name, frame in (("Distribution", summary), ("Cases", cases),
                            ("Notes", notes)):
            sheet = writer.sheets[name]
            for i, column in enumerate(frame.columns, start=1):
                width = max(len(str(column)),
                            *(len(str(v)) for v in frame[column])) + 2
                sheet.column_dimensions[
                    sheet.cell(row=1, column=i).column_letter
                ].width = min(width, 60)


if __name__ == "__main__":
    main()
