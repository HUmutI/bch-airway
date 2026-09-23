#!/usr/bin/env python
"""Stage 2 - AirMorph binary airway segmentation (the fusion seed).

Writes into each case dir: lunglobe.nii.gz, airway_bin.nii.gz,
airway_skeleton.nii.gz (segmentator/airway_segmentator.py:96-102).

airway_bin.nii.gz is a 5-model majority vote (>=3 of airway_model1/2/3 +
wingsnet + break) followed by KeepLargestConnectedComponent
(airway_segmentator.py:66-73). It is therefore ALREADY pruned when the fuse
step consumes it. fuse_union3.py's docstring flags this and points at a
RUN_ON_TARGET.md tweak to also save the pre-LCC ensemble - that file is not in
the shipped tarball, so we follow LOCAL_COMMANDS.md section 4/7 as written and
feed airway_bin.nii.gz. AirMorph is the SEED of the fusion, and a pruned seed
still seeds the right component; the cost is only whatever distal AirMorph-only
fragments the LCC dropped, which MEDPSeg and TfeNet largely re-supply.

Usage (airmorph venv, GPU required):
    $OR3_ENVS/airmorph/bin/python 02_airmorph.py --cases $OR3_CASES
"""
import argparse
import os
import sys
import time
import traceback

import SimpleITK as sitk

CODE = os.environ.get("OR3_CODE", "/fileserver/external/quin/ege/or_all3_run/code")
sys.path.insert(0, os.path.join(CODE, "AirMorph"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="file of case directories")
    ap.add_argument("--force", action="store_true", help="redo finished cases")
    args = ap.parse_args()

    dirs = [ln.strip() for ln in open(args.cases) if ln.strip()]
    todo = [d for d in dirs
            if args.force or not os.path.exists(os.path.join(d, "airway_bin.nii.gz"))]
    print(f"AirMorph: {len(todo)} to run, {len(dirs) - len(todo)} already done",
          flush=True)
    if not todo:
        return

    from segmentator.airway_segmentator import AirwayAtlasBinaryAirwaySegmentator
    seg = AirwayAtlasBinaryAirwaySegmentator()

    t0, failed = time.time(), []
    for i, d in enumerate(todo, 1):
        case = os.path.basename(d.rstrip("/"))
        print(f"=== [{i}/{len(todo)}] {case}", flush=True)
        ts = time.time()
        try:
            seg({"patient": case, "file_path": d})
            # AirMorph writes airway_bin.nii.gz through its own save_itk, so we
            # cannot make that write atomic from here. Read it straight back
            # instead: a process killed mid-write leaves a truncated file that
            # would otherwise look "done" to a resume and only blow up days
            # later in the fuse stage.
            chk = os.path.join(d, "airway_bin.nii.gz")
            n = int((sitk.GetArrayFromImage(sitk.ReadImage(chk)) > 0).sum())
            if n == 0:
                raise RuntimeError("airway_bin.nii.gz is empty - seed would be "
                                   "useless to the fusion")
            print(f"    done in {(time.time()-ts)/60:.1f} min, {n} seed voxels",
                  flush=True)
        except Exception:
            failed.append(case)
            print(f"    FAILED {case}\n{traceback.format_exc()}", flush=True)
        el = time.time() - t0
        print(f"    [{i}/{len(todo)}] {el/60:.1f} min elapsed, "
              f"~{el/i*(len(todo)-i)/3600:.1f} h left", flush=True)

    print(f"\nAirMorph complete. failed: {len(failed)}")
    for c in failed:
        print("   ", c)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
