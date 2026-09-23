#!/usr/bin/env python
"""Stage 1 - build the output tree and the LPS-reoriented working images.

Mirrors the dataset layout into $OR3_OUT:

    airway-dataset/age_00-01/sub-0012/sub-0012_acc-0295_source.nii.gz
    airway-seg/    age_00-01/sub-0012/sub-0012_acc-0295/image.nii.gz

Runs in two passes:
  1. HEADER PASS over all 849 scans - reads geometry only, no pixel data, so it
     takes seconds. Writes manifest.csv, cases_all.txt and cases_pilot.txt.
  2. WRITE PASS over the chosen subset only - decompress, reorient, recompress.
     This is the expensive part (~239 GB for the full cohort), which is why
     --subset pilot stages 13 scans rather than all 849.

WHY REORIENT. Every one of the 849 scans is LAS. AirMorph and TfeNet both
ignore direction cosines and assume LPS (LOCAL_COMMANDS.md:134-136), so feeding
them an LAS volume yields a mask mirrored front-to-back with every left/right
branch name flipped. LAS -> LPS flips the anterior/posterior axis. Stage 5
rotates the finished mask back to LAS so results overlay on the original scans,
and stage 7 verifies that end to end.

Also writes medpseg_in/image.nii.gz as a symlink. MEDPSeg's -i takes a FOLDER
and derives its output ID from the basename of every file it finds
(medpseg/pipeline.py:161), so pointing it at the case directory would make it
re-segment AirMorph's airway_bin.nii.gz and lunglobe.nii.gz. Isolating the
input makes stage order irrelevant.

Run with the airmorph venv (it has SimpleITK):
    $OR3_ENVS/airmorph/bin/python 01_stage.py --subset pilot
    $OR3_ENVS/airmorph/bin/python 01_stage.py --subset all
    $OR3_ENVS/airmorph/bin/python 01_stage.py --subset none   # lists only
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import SimpleITK as sitk

DATASET = os.environ.get("OR3_DATASET", "/fileserver/external/quin/ege/airway-dataset")
OUT = os.environ.get("OR3_OUT", "/fileserver/external/quin/ege/airway-seg")
RUN = os.environ.get("OR3_RUN", "/fileserver/external/quin/ege/or_all3_run")
SUFFIX = "_source.nii.gz"


def discover():
    """Every scan in the dataset, as (age_bucket, subject, case_id, src_path)."""
    found = []
    for bucket in sorted(os.listdir(DATASET)):
        bdir = os.path.join(DATASET, bucket)
        if not (bucket.startswith("age_") and os.path.isdir(bdir)):
            continue
        for subject in sorted(os.listdir(bdir)):
            sdir = os.path.join(bdir, subject)
            if not os.path.isdir(sdir):
                continue
            for fname in sorted(os.listdir(sdir)):
                if fname.endswith(SUFFIX):
                    found.append((bucket, subject, fname[: -len(SUFFIX)],
                                  os.path.join(sdir, fname)))
    return found


def out_dir_for(bucket, subject, case_id):
    return os.path.join(OUT, bucket, subject, case_id)


def read_header(item):
    """Pass 1: geometry only, no pixel data."""
    bucket, subject, case_id, src = item
    row = {"case_id": case_id, "subject": subject, "age_bucket": bucket,
           "src": src, "out_dir": out_dir_for(bucket, subject, case_id)}
    try:
        r = sitk.ImageFileReader()
        r.SetFileName(src)
        r.ReadImageInformation()
        size, spacing = r.GetSize(), r.GetSpacing()
        row.update({
            "src_orientation":
                sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                    r.GetDirection()),
            "nx": size[0], "ny": size[1], "nz": size[2],
            "sx": round(spacing[0], 4), "sy": round(spacing[1], 4),
            "sz": round(spacing[2], 4),
            "min_axis": min(size), "status": "ok", "error": ""})
    except Exception as exc:
        row.update({"src_orientation": "", "nx": 0, "ny": 0, "nz": 0,
                    "sx": 0, "sy": 0, "sz": 0, "min_axis": 0,
                    "status": "HEADER_FAILED", "error": str(exc)[:200]})
    return row


def stage_one(row):
    """Pass 2: reorient one scan to LPS and set up its case directory."""
    d = row["out_dir"]
    os.makedirs(d, exist_ok=True)
    img_path = os.path.join(d, "image.nii.gz")
    t0 = time.time()
    try:
        if not os.path.exists(img_path):          # resumable
            img = sitk.ReadImage(row["src"])
            f = sitk.DICOMOrientImageFilter()
            f.SetDesiredCoordinateOrientation("LPS")
            # Write to a temp name and rename. os.replace is atomic within a
            # filesystem, so a kill mid-write leaves no half-written
            # image.nii.gz for a later resume to mistake for finished work.
            # The temp name must KEEP the .nii.gz suffix - SimpleITK picks its
            # writer from the extension and rejects "....nii.gz.partial".
            tmp = os.path.join(d, ".partial.image.nii.gz")
            sitk.WriteImage(f.Execute(img), tmp, True)
            os.replace(tmp, img_path)

        # Isolated input folder for MEDPSeg (see module docstring).
        mp_in = os.path.join(d, "medpseg_in")
        os.makedirs(mp_in, exist_ok=True)
        link = os.path.join(mp_in, "image.nii.gz")
        if not os.path.islink(link) and not os.path.exists(link):
            os.symlink(os.path.join("..", "image.nii.gz"), link)

        json.dump({"case": row["case_id"], "subject": row["subject"],
                   "age_bucket": row["age_bucket"], "source": row["src"],
                   "source_orientation": row["src_orientation"],
                   "size": [row["nx"], row["ny"], row["nz"]],
                   "spacing": [row["sx"], row["sy"], row["sz"]]},
                  open(os.path.join(d, "orient.json"), "w"), indent=2)
        return row["case_id"], "ok", round(time.time() - t0, 1), ""
    except Exception as exc:
        return row["case_id"], "STAGE_FAILED", round(time.time() - t0, 1), str(exc)[:200]


def pick_pilot(rows):
    """A handful of cases that between them exercise every risk in the cohort."""
    ok = [r for r in rows if r["status"] == "ok"]
    chosen, seen = [], set()

    def take(r, why):
        if r and r["case_id"] not in seen:
            seen.add(r["case_id"])
            chosen.append((r, why))

    # The whole youngest bucket: smallest airways, hardest segmentation, and
    # the furthest from the adult data these networks were trained on.
    for r in [r for r in ok if r["age_bucket"] == "age_00-01"]:
        take(r, "age_00-01 (all)")
    if ok:
        # Largest volume: the TfeNet runtime and peak-memory worst case.
        take(max(ok, key=lambda r: r["nx"] * r["ny"] * r["nz"]), "largest volume")
        # Smallest axis: closest to TfeNet's 128-cube floor.
        take(min(ok, key=lambda r: r["min_axis"]), "smallest axis")
        # Non-square in-plane: 40 scans are not 512x512.
        take(next((r for r in ok if r["nx"] != r["ny"]), None),
             "non-square in-plane")
        # Two adolescents, where volumes should look most adult-like.
        for r in [r for r in ok if r["age_bucket"] == "age_16-18"][:2]:
            take(r, "age_16-18")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["pilot", "all", "none"], default="pilot",
                    help="which cases to actually write LPS images for")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("OR3_STAGE_WORKERS", "10")))
    args = ap.parse_args()

    items = discover()
    print(f"discovered {len(items)} scans under {DATASET}", flush=True)
    if not items:
        sys.exit("no scans found - check OR3_DATASET")

    # --- pass 1: headers ----------------------------------------------------
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(read_header, items, 8)
    rows.sort(key=lambda r: (r["age_bucket"], r["subject"], r["case_id"]))
    print(f"header pass: {time.time()-t0:.1f}s", flush=True)

    os.makedirs(RUN, exist_ok=True)
    with open(os.path.join(RUN, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote manifest.csv ({len(rows)} scans)")

    ok = [r for r in rows if r["status"] == "ok"]
    with open(os.path.join(RUN, "cases_all.txt"), "w") as fh:
        fh.writelines(r["out_dir"] + "\n" for r in ok)
    print(f"wrote cases_all.txt ({len(ok)} cases)")

    pilot = pick_pilot(rows)
    with open(os.path.join(RUN, "cases_pilot.txt"), "w") as fh:
        fh.writelines(r["out_dir"] + "\n" for r, _ in pilot)
    print(f"wrote cases_pilot.txt ({len(pilot)} cases)")
    for r, why in pilot:
        print(f"    {r['case_id']:26s} {r['nx']}x{r['ny']}x{r['nz']:<5d}  {why}")

    bad = [r for r in rows if r["status"] != "ok"]
    small = [r for r in ok if r["min_axis"] < 128]
    print(f"\nunreadable headers: {len(bad)}   "
          f"axis<128 (TfeNet cannot run): {len(small)}")
    for r in bad + small:
        print(f"    {r['case_id']} {r['status']} min_axis={r['min_axis']} {r['error']}")
    print(f"source orientations present: {sorted({r['src_orientation'] for r in ok})}")

    # --- pass 2: write LPS images ------------------------------------------
    if args.subset == "none":
        print("\n--subset none: lists only, no images written")
        return
    targets = ok if args.subset == "all" else [r for r, _ in pilot]
    todo = [r for r in targets
            if not os.path.exists(os.path.join(r["out_dir"], "image.nii.gz"))]
    gb = sum(os.path.getsize(r["src"]) for r in todo) / 1e9
    print(f"\nstaging {len(todo)} of {len(targets)} ({args.subset}); "
          f"{len(targets)-len(todo)} already present. ~{gb:.0f} GB to read.",
          flush=True)

    t0, failed = time.time(), []
    with Pool(args.workers) as pool:
        for i, (case, status, secs, err) in enumerate(
                pool.imap_unordered(stage_one, todo, 1), 1):
            if status != "ok":
                failed.append(case)
                print(f"  FAILED {case}: {err}", flush=True)
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  staged {i}/{len(todo)}  {el/60:.1f} min elapsed, "
                      f"~{el/i*(len(todo)-i)/60:.1f} min left", flush=True)

    print(f"\nstaging complete. failed: {len(failed)}")
    for c in failed:
        print("   ", c)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
