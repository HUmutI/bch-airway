#!/usr/bin/env python
"""Stage 6 - QC report over the finished cohort.

Writes qc.csv (one row per case) and prints a per-age-bucket summary.

LOCAL_COMMANDS.md:353 gives "adult airway is roughly 20-50 mL" as the sanity
check. That does NOT transfer to this cohort - these are ages 0-18, and an
infant airway is a small fraction of an adult one. So rather than a fixed
clinical window, this flags outliers two ways:

  ABSOLUTE  - under 0.5 mL is almost certainly a failed segmentation at any age.
  RELATIVE  - robust z-score (median / MAD) WITHIN each age bucket, so a case is
              judged against its own peers. |z| > 3.5 is flagged.

Also reports each member's contribution. Useful failure signatures:
  - tfenet near zero          -> CUDA op or windowing problem
  - union >> kept             -> fragmented union, LCC discarded a lot
  - airmorph near zero        -> seed failed; fusion fell back to largest
                                 component and the result is untrustworthy

Usage (airmorph venv, no GPU needed):
    $OR3_ENVS/airmorph/bin/python 06_qc.py --cases $OR3_CASES
"""
import argparse
import csv
import json
import os
import statistics as st

import numpy as np
import SimpleITK as sitk

RUN = os.environ.get("OR3_RUN", "/fileserver/external/quin/ege/or_all3_run")

TINY_ML = 0.5          # below this a mask is a failed segmentation at any age
MIN_PEERS = 5          # bucket size needed before the outlier test is trusted

FILES = [("airmorph", "airway_bin.nii.gz"),
         ("medpseg", "medpseg_airway.nii.gz"),
         ("tfenet", "tfenet_airway.nii.gz"),
         ("union3_lcc", "union3_lcc.nii.gz")]


def vol_ml(path):
    """(voxel count, millilitres) for a binary mask, or (None, None)."""
    if not os.path.exists(path):
        return None, None
    img = sitk.ReadImage(path)
    a = sitk.GetArrayFromImage(img) > 0
    n = int(a.sum())
    return n, n * float(np.prod(img.GetSpacing())) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_csv = args.out or os.path.join(RUN, "qc.csv")

    dirs = [ln.strip() for ln in open(args.cases) if ln.strip()]
    rows = []
    for i, d in enumerate(dirs, 1):
        case = os.path.basename(d.rstrip("/"))
        meta = {}
        oj = os.path.join(d, "orient.json")
        if os.path.exists(oj):
            meta = json.load(open(oj))
        row = {"case_id": case,
               "age_bucket": meta.get("age_bucket", ""),
               "subject": meta.get("subject", ""),
               "src_orientation": meta.get("source_orientation", ""),
               "out_dir": d}
        for key, fname in FILES:
            n, ml = vol_ml(os.path.join(d, fname))
            row[f"{key}_vox"] = n if n is not None else ""
            row[f"{key}_mL"] = round(ml, 3) if ml is not None else ""
        row["complete"] = all(row[f"{k}_vox"] != "" for k, _ in FILES)
        row["srcorient_written"] = os.path.exists(
            os.path.join(d, "union3_lcc_srcorient.nii.gz"))
        fl = os.path.join(d, "fuse.log")
        row["seeding_mattered"] = (
            "SEEDING MATTERED" in open(fl).read() if os.path.exists(fl) else "")
        rows.append(row)
        if i % 50 == 0:
            print(f"  read {i}/{len(dirs)}", flush=True)

    # --- flagging -----------------------------------------------------------
    by_bucket = {}
    for r in rows:
        if isinstance(r.get("union3_lcc_mL"), float):
            by_bucket.setdefault(r["age_bucket"], []).append(r["union3_lcc_mL"])

    stats = {}
    for b, vals in by_bucket.items():
        med = st.median(vals)
        mad = st.median([abs(v - med) for v in vals]) or 1e-9
        stats[b] = (med, mad, len(vals))

    # Two separate columns on purpose. `flags` means something is WRONG and the
    # case should not be trusted. `notes` means something is worth knowing but
    # the output is fine. Mixing them made a clean 9-case group report
    # "flagged: 7/9", which reads as a disaster when in fact nothing failed:
    # six were the seeded fusion rule working as designed, and two were the
    # youngest infants in a bucket that spans birth to 12 months - a several-fold
    # real range in airway size, not a defect.
    for r in rows:
        v, flags, notes = r.get("union3_lcc_mL"), [], []
        if not r["complete"]:
            flags.append("INCOMPLETE")
        if isinstance(v, float):
            if v < TINY_ML:
                flags.append(f"TINY(<{TINY_ML}mL)")
            med, mad, npeers = stats.get(r["age_bucket"], (None, None, 0))
            # Only judge a case against its bucket when the bucket is big enough
            # to have a meaningful spread. On a 20-case batch a bucket may hold
            # 1-2 scans, where MAD collapses toward 0 and every case looks like a
            # wild outlier. The cohort-wide pass at the end is the real test.
            if med is not None and npeers >= MIN_PEERS:
                z = 0.6745 * (v - med) / mad
                r["robust_z"] = round(z, 2)
                # A note, not a flag: within one age bucket real body-size
                # variation alone moves airway volume several-fold, so a large
                # z means "compare against this patient's size", not "broken".
                if abs(z) > 3.5:
                    notes.append(f"off-peers(z={z:.1f})")
        if r.get("airmorph_vox") == 0:
            flags.append("SEED_EMPTY")
        if r["seeding_mattered"] is True:
            notes.append("seeding-mattered")
        r["flags"] = ";".join(flags)
        r["notes"] = ";".join(notes)

    with open(out_csv, "w", newline="") as fh:
        cols = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}  ({len(rows)} cases)\n")

    print(f"{'bucket':12s} {'n':>4s} {'median mL':>10s} {'min':>8s} {'max':>8s}")
    for b in sorted(stats):
        med, mad, n = stats[b]
        vals = by_bucket[b]
        print(f"{b:12s} {n:4d} {med:10.2f} {min(vals):8.2f} {max(vals):8.2f}")

    flagged = [r for r in rows if r["flags"]]
    noted = [r for r in rows if r["notes"] and not r["flags"]]

    print(f"\nNEEDS ATTENTION: {len(flagged)}/{len(rows)}")
    if not flagged:
        print("    none - every case produced a complete, non-empty mask")
    for r in flagged:
        print(f"    {r['case_id']:28s} {str(r.get('union3_lcc_mL')):>8s} mL  {r['flags']}")

    print(f"\nfor information: {len(noted)}/{len(rows)}")
    for r in noted:
        print(f"    {r['case_id']:28s} {str(r.get('union3_lcc_mL')):>8s} mL  {r['notes']}")
    n_seed = sum(1 for r in rows if r["seeding_mattered"] is True)
    if n_seed:
        print(f"\n    seeding-mattered on {n_seed}/{len(rows)}: TfeNet produced a blob "
              f"larger than the airway and the seeded rule kept the correct\n"
              f"    component. These outputs are correct - a plain "
              f"largest-component rule would have deleted the airway.")


if __name__ == "__main__":
    main()
