#!/usr/bin/env python
"""Stage 3 - MEDPSeg airway segmentation.

Two flags matter (LOCAL_COMMANDS.md:209-211):
  --post is OFF by default, which is what the union needs - the union prunes
    once, at the end, in the fuse step. Passing --post here would prune twice.
  --disable_lobe skips the lobes head, which is not part of or_all3_lcc (and
    lober.ckpt was excluded from the tarball anyway).

Input is medpseg_in/, which holds only a symlink to image.nii.gz. MEDPSeg's -i
takes a FOLDER and derives its output ID from every file basename it finds
(medpseg/pipeline.py:161), so aiming it at the case directory would make it
segment AirMorph's airway_bin.nii.gz and lunglobe.nii.gz as if they were CTs.

Output lands at medpseg_out/image_airway.nii.gz (ID "image" + "_airway.nii.gz",
pipeline.py:477) and is copied to medpseg_airway.nii.gz for the fuse step.

Usage (medpseg venv, GPU required):
    $OR3_ENVS/medpseg/bin/python 03_medpseg.py --cases $OR3_CASES
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ENVS = os.environ.get("OR3_ENVS", "/fileserver/external/quin/ege/or_all3_run/envs")
MEDPSEG_BIN = os.path.join(ENVS, "medpseg", "bin", "medpseg")


def run_one(d):
    in_dir = os.path.join(d, "medpseg_in")
    out_dir = os.path.join(d, "medpseg_out")
    final = os.path.join(d, "medpseg_airway.nii.gz")

    if not os.path.exists(os.path.join(in_dir, "image.nii.gz")):
        raise RuntimeError(f"missing {in_dir}/image.nii.gz - run 01_stage.py first")

    proc = subprocess.run([MEDPSEG_BIN, "-i", in_dir, "-o", out_dir,
                           "--disable_lobe"],
                          capture_output=True, text=True)
    combined = proc.stdout + proc.stderr

    # ALWAYS keep the log. MEDPSeg runs its work in a thread launched from
    # another thread (gui.py:168, run.py main() -> MainWindow(...).join()), so
    # an exception in the pipeline is swallowed by threading and the process
    # still exits 0. Logging only on non-zero return threw away the one copy of
    # the traceback and left "succeeded but produced nothing".
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(d, "medpseg.log"), "w") as fh:
        fh.write(combined)

    if proc.returncode != 0:
        raise RuntimeError(f"medpseg exit {proc.returncode}\n{combined[-2000:]}")

    # Exit code 0 is not evidence of success - look for its own abort banner.
    if "Aborting processing" in combined:
        err = next((ln.strip() for ln in combined.splitlines()
                    if ln.startswith("ERROR:")), "see medpseg.log")
        raise RuntimeError(f"medpseg aborted internally (exit 0): {err}")

    produced = os.path.join(out_dir, "image_airway.nii.gz")
    if not os.path.exists(produced):
        # Be forgiving if this install names things differently.
        cands = glob.glob(os.path.join(out_dir, "*airway*.nii.gz"))
        if not cands:
            raise RuntimeError(
                f"no airway output in {out_dir}; found: "
                f"{sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else 'nothing'}")
        produced = cands[0]
    # Atomic: copy then rename, so an interrupted copy cannot leave a truncated
    # medpseg_airway.nii.gz that resume treats as a finished case.
    tmp = final + ".partial"
    shutil.copy2(produced, tmp)
    os.replace(tmp, final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dirs = [ln.strip() for ln in open(args.cases) if ln.strip()]
    todo = [d for d in dirs
            if args.force or not os.path.exists(os.path.join(d, "medpseg_airway.nii.gz"))]
    print(f"MEDPSeg: {len(todo)} to run, {len(dirs) - len(todo)} already done",
          flush=True)

    t0, failed = time.time(), []
    for i, d in enumerate(todo, 1):
        case = os.path.basename(d.rstrip("/"))
        print(f"=== [{i}/{len(todo)}] {case}", flush=True)
        ts = time.time()
        try:
            run_one(d)
            print(f"    done in {(time.time()-ts)/60:.1f} min", flush=True)
        except Exception as exc:
            failed.append(case)
            print(f"    FAILED {case}: {exc}", flush=True)
        el = time.time() - t0
        print(f"    [{i}/{len(todo)}] {el/60:.1f} min elapsed, "
              f"~{el/i*(len(todo)-i)/3600:.1f} h left", flush=True)

    print(f"\nMEDPSeg complete. failed: {len(failed)}")
    for c in failed:
        print("   ", c)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
