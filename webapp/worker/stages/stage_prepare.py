#!/usr/bin/env python
"""Turn one uploaded NIfTI into the case directory the pipeline expects.

Mirrors stage 01 of the cohort run. The models assume LPS and ignore direction
cosines, so anything else would come back mirrored with every left and right
branch name swapped. The source orientation is recorded so the final mask can
be rotated back onto the scan the user uploaded.

Runs in the AirMorph environment (it has SimpleITK).
    python stage_prepare.py --input upload.nii.gz --case-dir job/case
"""
import argparse, json, os, sys
import numpy as np
import SimpleITK as sitk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--case-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.case_dir, exist_ok=True)

    img = sitk.ReadImage(a.input)
    if img.GetDimension() != 3:
        sys.exit(f"expected a 3D volume, got {img.GetDimension()}D")
    if img.GetNumberOfComponentsPerPixel() != 1:
        sys.exit("expected a single-channel volume")
    size, spacing = img.GetSize(), img.GetSpacing()
    src_code = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        img.GetDirection())
    print(f"input {size} spacing {tuple(round(s, 3) for s in spacing)} "
          f"orientation {src_code}", flush=True)

    # The models were trained on HU. A scan stored as unsigned with an offset,
    # or as a float in a strange range, would segment nothing and fail late.
    arr = sitk.GetArrayViewFromImage(img)
    lo, hi = float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5))
    if lo > -200 or hi < 200:
        sys.exit(f"intensities look non-HU ({lo:.0f} to {hi:.0f} at the 0.5 and "
                 f"99.5 percentiles); the models expect Hounsfield units")

    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation("LPS")
    lps = f.Execute(img)
    lps = sitk.Cast(lps, sitk.sitkInt16)
    tmp = os.path.join(a.case_dir, ".partial.image.nii.gz")
    sitk.WriteImage(lps, tmp, True)
    os.replace(tmp, os.path.join(a.case_dir, "image.nii.gz"))

    # MEDPSeg takes a folder and segments every file it finds there, so it gets
    # a folder holding only the image.
    mp_in = os.path.join(a.case_dir, "medpseg_in")
    os.makedirs(mp_in, exist_ok=True)
    link = os.path.join(mp_in, "image.nii.gz")
    if not os.path.lexists(link):
        os.symlink(os.path.join("..", "image.nii.gz"), link)

    json.dump({"source_orientation": src_code,
               "size": list(size), "spacing": list(spacing),
               "lps_size": list(lps.GetSize())},
              open(os.path.join(a.case_dir, "orient.json"), "w"), indent=2)
    print("prepared", flush=True)


if __name__ == "__main__":
    main()
