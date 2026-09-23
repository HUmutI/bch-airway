#!/usr/bin/env python
"""Stage 5 - fuse the three masks (this IS or_all3_lcc), then rotate back.

    union = AirMorph | MEDPSeg | TfeNet
    out   = the connected component of union containing the AirMorph tree

Delegates the fusion itself to the shipped fuse_union3.py via subprocess so the
validated implementation runs byte-for-byte unmodified. Then reorients the
result from the LPS working frame back to the scan's source orientation (LAS
for all 849 scans here), so masks overlay on the original dataset files.

Watch for "SEEDING MATTERED" in the log: it means TfeNet emitted a blob larger
than the airway on that case. The output is still correct - that is precisely
the collapse the seeded rule prevents - but flag the case for review
(LOCAL_COMMANDS.md:316-318). Those lines are collected into seeding_mattered.txt.

Connectivity must stay FIXED across the cohort. 6 is the fuse_union3.py default
and the rule the published or_all3_lcc scores were measured with; 26 matches
the BCH_LIAM clinical runs.

Usage (airmorph venv - needs SimpleITK + scipy; no GPU needed):
    $OR3_ENVS/airmorph/bin/python 05_fuse.py --cases $OR3_CASES --connectivity 6
"""
import argparse
import json
import os
import subprocess
import sys
import time

import SimpleITK as sitk

TARBALLS = os.environ.get("OR3_TARBALLS", "/fileserver/external/quin/ege/all_3_lcc")
RUN = os.environ.get("OR3_RUN", "/fileserver/external/quin/ege/or_all3_run")
FUSE = os.path.join(TARBALLS, "fuse_union3.py")

MEMBERS = (("airmorph", "airway_bin.nii.gz"),
           ("medpseg", "medpseg_airway.nii.gz"),
           ("tfenet", "tfenet_airway.nii.gz"))


def tmp_path(path):
    """Temp name for an atomic write that KEEPS the .nii.gz suffix.

    SimpleITK selects its ImageIO from the file extension, so writing to
    "union3_lcc.nii.gz.partial" fails with "Unable to determine ImageIO
    writer". Prefixing instead of suffixing keeps the extension intact, and the
    leading dot keeps the temp file out of ordinary globs.
    """
    d, b = os.path.split(path)
    return os.path.join(d, ".partial." + b)


def fuse_one(d, connectivity, save_union):
    out = os.path.join(d, "union3_lcc.nii.gz")
    # Point fuse_union3.py at a temp path and rename on success. The fusion
    # script is left byte-for-byte unmodified; this just means a killed or
    # failed fuse never leaves a partial union3_lcc.nii.gz behind for resume to
    # skip over.
    tmp = tmp_path(out)
    cmd = [sys.executable, FUSE, "--out", tmp,
           "--connectivity", str(connectivity)]
    for flag, fname in MEMBERS:
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            raise RuntimeError(f"missing member {fname}")
        cmd += [f"--{flag}", p]
    if save_union:
        cmd += ["--save-union", os.path.join(d, "union3_raw.nii.gz")]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    with open(os.path.join(d, "fuse.log"), "w") as fh:
        fh.write(log)
    if proc.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"fuse exit {proc.returncode}\n{log[-2000:]}")
    if not os.path.exists(tmp):
        raise RuntimeError(f"fuse exited 0 but wrote nothing\n{log[-2000:]}")
    os.replace(tmp, out)          # publish only once complete
    return log


def rotate_back(d):
    """LPS working frame -> the scan's original orientation."""
    oj = os.path.join(d, "orient.json")
    if not os.path.exists(oj):
        return None
    code = json.load(open(oj))["source_orientation"]
    if code == "LPS":
        return code
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation(code)
    src = os.path.join(d, "union3_lcc.nii.gz")
    dst = os.path.join(d, "union3_lcc_srcorient.nii.gz")
    tmp = tmp_path(dst)
    sitk.WriteImage(f.Execute(sitk.ReadImage(src)), tmp, True)
    os.replace(tmp, dst)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--connectivity", type=int,
                    default=int(os.environ.get("OR3_CONNECTIVITY", "6")),
                    choices=[6, 26])
    ap.add_argument("--save-union", action="store_true",
                    help="also keep the pre-LCC union (union3_raw.nii.gz)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(FUSE):
        sys.exit(f"fuse_union3.py not found at {FUSE}")

    dirs = [ln.strip() for ln in open(args.cases) if ln.strip()]
    todo = [d for d in dirs
            if args.force or not os.path.exists(os.path.join(d, "union3_lcc.nii.gz"))]
    print(f"fuse (connectivity={args.connectivity}): {len(todo)} to run, "
          f"{len(dirs) - len(todo)} already done", flush=True)

    t0, failed, seeded = time.time(), [], []
    for i, d in enumerate(todo, 1):
        case = os.path.basename(d.rstrip("/"))
        print(f"=== [{i}/{len(todo)}] {case}", flush=True)
        try:
            log = fuse_one(d, args.connectivity, args.save_union)
            if "SEEDING MATTERED" in log:
                seeded.append(case)
                print("    SEEDING MATTERED - review this case", flush=True)
            code = rotate_back(d)
            for line in log.splitlines():
                if "kept " in line:
                    print("   ", line.split("fuse: ")[-1], flush=True)
            print(f"    rotated back to {code}", flush=True)
        except Exception as exc:
            failed.append(case)
            print(f"    FAILED {case}: {exc}", flush=True)

    if seeded:
        with open(os.path.join(RUN, "seeding_mattered.txt"), "w") as fh:
            fh.writelines(c + "\n" for c in seeded)

    print(f"\nfuse complete in {(time.time()-t0)/60:.1f} min. "
          f"failed: {len(failed)}   seeding mattered: {len(seeded)}")
    for c in failed:
        print("    FAILED", c)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
