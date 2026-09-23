#!/usr/bin/env python
"""AirMorph binary airway segmentation, the seed of the fusion.

Writes lunglobe.nii.gz, airway_bin.nii.gz and airway_skeleton.nii.gz into the
case directory, exactly as stage 02 of the cohort run does. airway_bin is the
five-member vote already reduced to its largest component, which is what the
cohort recipe fed to the fusion, so it is what this feeds too.

Runs in the AirMorph environment on the GPU.
    python stage_airmorph.py --case-dir job/case
"""
import argparse, os, sys, time
import SimpleITK as sitk

CODE = os.environ.get("OR3_CODE", "/fileserver/external/quin/ege/or_all3_run/code")
sys.path.insert(0, os.path.join(CODE, "AirMorph"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    a = ap.parse_args()
    d = os.path.abspath(a.case_dir)

    from segmentator.airway_segmentator import AirwayAtlasBinaryAirwaySegmentator
    t0 = time.time()
    print("loading AirMorph ensemble", flush=True)
    seg = AirwayAtlasBinaryAirwaySegmentator()
    print(f"loaded in {time.time()-t0:.0f}s, segmenting", flush=True)
    seg({"patient": os.path.basename(d), "file_path": d})

    out = os.path.join(d, "airway_bin.nii.gz")
    n = int((sitk.GetArrayFromImage(sitk.ReadImage(out)) > 0).sum())
    if n == 0:
        sys.exit("AirMorph produced an empty airway; the fusion would have no seed")
    print(f"airmorph done in {time.time()-t0:.0f}s, {n} seed voxels", flush=True)


if __name__ == "__main__":
    main()
