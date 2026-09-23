#!/usr/bin/env python
"""Fuse the three masks and rotate the result back to the uploaded orientation.

    union = AirMorph | MEDPSeg | TfeNet
    out   = the 6-connected component of the union that holds the AirMorph seed

This is the paper's segmentation. The fusion itself is delegated to the shipped
fuse_union3.py so the validated implementation runs unmodified. "SEEDING
MATTERED" in its output means a member emitted something larger than the
airway and the largest-component rule would have kept that instead; the result
is still correct, which is the point of seeding.

Runs in the AirMorph environment, CPU only.
    python stage_fuse.py --case-dir job/case --fuse-script .../fuse_union3.py
"""
import argparse, json, os, subprocess, sys
import SimpleITK as sitk

MEMBERS = [("airmorph", "airway_bin.nii.gz"),
           ("medpseg", "medpseg_airway.nii.gz"),
           ("tfenet", "tfenet_airway.nii.gz")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--fuse-script", required=True)
    ap.add_argument("--connectivity", type=int, default=6, choices=(6, 26))
    a = ap.parse_args()
    d = os.path.abspath(a.case_dir)

    out = os.path.join(d, "union3_lcc.nii.gz")
    tmp = os.path.join(d, ".partial.union3_lcc.nii.gz")
    cmd = [sys.executable, a.fuse_script, "--out", tmp,
           "--connectivity", str(a.connectivity),
           "--save-union", os.path.join(d, "union3_raw.nii.gz")]
    for flag, f in MEMBERS:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            sys.exit(f"missing member mask {f}")
        cmd += [f"--{flag}", p]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    print(log, flush=True)
    if proc.returncode != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        sys.exit(f"fusion failed (exit {proc.returncode})")
    os.replace(tmp, out)
    if "SEEDING MATTERED" in log:
        print("note: seeding mattered on this scan, a member produced something "
              "larger than the airway and largest-component selection would have "
              "discarded the tree", flush=True)

    # Back to the orientation the user uploaded, so the mask overlays their scan.
    code = json.load(open(os.path.join(d, "orient.json")))["source_orientation"]
    src = sitk.ReadImage(out)
    if code != "LPS":
        f = sitk.DICOMOrientImageFilter()
        f.SetDesiredCoordinateOrientation(code)
        src = f.Execute(src)
    tmp2 = os.path.join(d, ".partial.union3_lcc_srcorient.nii.gz")
    sitk.WriteImage(src, tmp2, True)
    os.replace(tmp2, os.path.join(d, "union3_lcc_srcorient.nii.gz"))
    n = int((sitk.GetArrayFromImage(sitk.ReadImage(out)) > 0).sum())
    print(f"fused airway: {n} voxels", flush=True)


if __name__ == "__main__":
    main()
