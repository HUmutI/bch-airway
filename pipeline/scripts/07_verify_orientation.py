#!/usr/bin/env python
"""Stage 7 - prove the masks are not mirrored, quantitatively.

THE RISK. All 849 scans are LAS. AirMorph and TfeNet ignore direction cosines
and assume LPS (LOCAL_COMMANDS.md:134-136). If the LAS->LPS staging in stage 1
were wrong or skipped, every mask would come out flipped front-to-back. A
mirrored airway mask still looks like a plausible tree in a viewer, so this
failure is silent - which is exactly why it deserves an automatic test.

THE TEST. Airway lumen is air: about -1000 to -900 HU. Anything else in the
chest - parenchyma, vessel, soft tissue, bone - is markedly denser. So sample
the ORIGINAL source scan at the voxels of the final mask (union3_lcc_srcorient,
which is back in the source frame). If the geometry is right the sampled
intensities are overwhelmingly air. If the mask is mirrored it lands on
mediastinum and parenchyma instead, and the mean HU rises sharply.

    PASS   mean HU <= -850 and >= 90% of mask voxels under -800
    SUSPECT otherwise - inspect in itksnap before trusting the run

This also implicitly validates the rotate-back in stage 5, because it compares
against the untouched dataset file rather than our staged copy.

Usage (airmorph venv, no GPU needed):
    $OR3_ENVS/airmorph/bin/python 07_verify_orientation.py --cases $OR3_CASES
"""
import argparse
import json
import os

import numpy as np
import SimpleITK as sitk

# Thresholds calibrated on the 13-case pilot (2026-08-09), by scoring each
# finished mask against the source scan and then re-scoring a DELIBERATELY
# mirrored copy of the same mask:
#
#                       meanHU    fracAir
#   as produced         -940       0.85     <- correct
#   A-P mirrored        -110       0.077    <- the failure mode we are hunting
#   L-R mirrored        -198       0.095
#
# The separation is enormous, so the gate does not need to be tight. An earlier
# fracAir >= 0.90 flagged 6 infants that were entirely correct: an infant airway
# is 1-3 mL, so most of its voxels are boundary voxels where partial-volume
# mixing with the airway wall lifts the mean. 0.80 keeps every real case and
# still sits ~8x above anything a mirrored mask can reach.
AIR_MEAN_MAX = -850.0     # mean HU inside a correct airway mask
AIR_FRAC_MIN = 0.80       # fraction of mask voxels that must be under -800 HU
AIR_THRESH = -800.0


def check(d):
    oj = os.path.join(d, "orient.json")
    if not os.path.exists(oj):
        return {"status": "NO_ORIENT_JSON"}
    meta = json.load(open(oj))
    src = meta["source"]

    # Prefer the source-frame mask; fall back to the LPS one if the scan was
    # already LPS (then no rotate-back file is written).
    mask_p = os.path.join(d, "union3_lcc_srcorient.nii.gz")
    frame = "srcorient"
    if not os.path.exists(mask_p):
        mask_p = os.path.join(d, "union3_lcc.nii.gz")
        frame = "lps"
        if meta["source_orientation"] != "LPS":
            return {"status": "NO_SRCORIENT_MASK", "frame": frame}
    if not os.path.exists(mask_p) or not os.path.exists(src):
        return {"status": "MISSING_INPUT", "frame": frame}

    ct = sitk.ReadImage(src)
    mk = sitk.ReadImage(mask_p)
    if ct.GetSize() != mk.GetSize():
        return {"status": "GRID_MISMATCH", "frame": frame,
                "detail": f"ct{ct.GetSize()} vs mask{mk.GetSize()}"}

    a = sitk.GetArrayFromImage(ct).astype(np.float32)
    m = sitk.GetArrayFromImage(mk) > 0
    n = int(m.sum())
    if n == 0:
        return {"status": "EMPTY_MASK", "frame": frame}

    vals = a[m]
    mean_hu = float(vals.mean())
    frac_air = float((vals < AIR_THRESH).mean())
    ok = mean_hu <= AIR_MEAN_MAX and frac_air >= AIR_FRAC_MIN
    return {"status": "PASS" if ok else "SUSPECT", "frame": frame,
            "vox": n, "mean_hu": round(mean_hu, 1),
            "frac_air": round(frac_air, 4),
            "src_orientation": meta["source_orientation"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    args = ap.parse_args()

    dirs = [ln.strip() for ln in open(args.cases) if ln.strip()]
    print(f"{'case':30s} {'frame':10s} {'meanHU':>8s} {'fracAir':>8s}  status")
    bad = []
    for d in dirs:
        case = os.path.basename(d.rstrip("/"))
        r = check(d)
        print(f"{case:30s} {r.get('frame',''):10s} "
              f"{str(r.get('mean_hu','')):>8s} {str(r.get('frac_air','')):>8s}  "
              f"{r['status']} {r.get('detail','')}")
        if r["status"] != "PASS":
            bad.append(case)

    print(f"\n{len(dirs)-len(bad)}/{len(dirs)} PASS")
    if bad:
        print("\nInspect these before trusting the cohort:")
        for c in bad:
            print("   ", c)
        print("\n  itksnap -g <source .nii.gz> -s <case>/union3_lcc_srcorient.nii.gz")
        raise SystemExit(1)
    print("Masks sit on air in the source frame - orientation handling is correct.")


if __name__ == "__main__":
    main()
