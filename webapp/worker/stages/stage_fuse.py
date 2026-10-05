#!/usr/bin/env python
"""Fuse the three masks, attach a confidence to every voxel, rotate back.

    union = AirMorph | MEDPSeg | TfeNet
    out   = the 6-connected component of the union that holds the AirMorph seed

This is the paper's segmentation. The fusion itself is delegated to the shipped
fuse_union3.py so the validated implementation runs unmodified. "SEEDING
MATTERED" in its output means a member emitted something larger than the
airway and the largest-component rule would have kept that instead; the result
is still correct, which is the point of seeding.

A union keeps everything any one model said, so part of the mask rests on a
single model. The confidence map says which part: for every fused voxel,
P(airway | which models marked it, and how many of AirMorph's five networks),
calibrated on the ATM'22 validation split (topofuse.confidence,
confidence_lut.json). It is written as confidence.nii.gz, uint8 percent, and
the viewer colours the mask by it.

Optional --fusion topofuse (config.FUSION, env AIRWAY_FUSION): after the
confidence, improbable leaf branches are pruned and the boundary trimmed
relative to each branch's centreline confidence (topofuse.branch_tree /
branch_prob / prune / refine). The plain union is kept as
union3_lcc_union.nii.gz. The default, "union", leaves the mask exactly as
before.

Runs in the AirMorph environment, CPU only.
    python stage_fuse.py --case-dir job/case --fuse-script .../fuse_union3.py
"""
import argparse, json, os, subprocess, sys
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topofuse  # noqa: E402

MEMBERS = [("airmorph", "airway_bin.nii.gz"),
           ("medpseg", "medpseg_airway.nii.gz"),
           ("tfenet", "tfenet_airway.nii.gz")]

# topofuse tunables: leaf branches below TAU probability are cut (generations
# 1-3 are never cut); a voxel survives when its confidence is at least KAPPA
# times its branch's centreline confidence, or lies within CORE voxels of the
# centreline. The defaults are the study's operating point.
TAU = float(os.environ.get("AIRWAY_TOPOFUSE_TAU", "0"))
KAPPA = float(os.environ.get("AIRWAY_TOPOFUSE_KAPPA", "0.4"))
CORE = float(os.environ.get("AIRWAY_TOPOFUSE_CORE", "1.5"))


def read_bin(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)) > 0


def write_like(arr, ref, path):
    """Write arr on ref's grid, atomically (the API may be listing the dir)."""
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref)
    tmp = os.path.join(os.path.dirname(path), ".partial." + os.path.basename(path))
    sitk.WriteImage(img, tmp, True)
    os.replace(tmp, path)


def write_srcorient(d, name, code):
    """A copy in the orientation the user uploaded, so it overlays their scan."""
    src = sitk.ReadImage(os.path.join(d, name))
    if code != "LPS":
        f = sitk.DICOMOrientImageFilter()
        f.SetDesiredCoordinateOrientation(code)
        src = f.Execute(src)
    out = os.path.join(d, name.replace(".nii.gz", "_srcorient.nii.gz"))
    tmp = os.path.join(d, ".partial." + os.path.basename(out))
    sitk.WriteImage(src, tmp, True)
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--fuse-script", required=True)
    ap.add_argument("--connectivity", type=int, default=6, choices=(6, 26))
    ap.add_argument("--fusion", default="union", choices=("union", "topofuse"))
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

    # ---- confidence per voxel ------------------------------------------------
    img = sitk.ReadImage(out)
    mask = sitk.GetArrayFromImage(img) > 0
    airway = read_bin(os.path.join(d, "airway_bin.nii.gz"))
    medpseg = read_bin(os.path.join(d, "medpseg_airway.nii.gz"))
    tfenet = read_bin(os.path.join(d, "tfenet_airway.nii.gz"))
    votes_p = os.path.join(d, "airmorph_votes.nii.gz")
    if os.path.exists(votes_p):
        votes = sitk.GetArrayFromImage(sitk.ReadImage(votes_p)).astype(np.uint8)
    else:
        # jobs from before the vote count was saved: AirMorph's mask is the
        # >=3 vote, so treat it as unanimous. Coarser, never wrong by much.
        votes = (airway.astype(np.uint8) * 5)
        print("note: airmorph_votes.nii.gz missing, using 5 votes wherever "
              "AirMorph's mask is set", flush=True)
    conf = topofuse.confidence(votes, medpseg, tfenet, mask)
    n_union = int(mask.sum())
    branches_removed = 0

    # ---- optional topology-aware fusion ---------------------------------------
    if a.fusion == "topofuse":
        spacing_zyx = np.array(img.GetSpacing()[::-1])
        bm, dist, tab = topofuse.branch_tree(mask, spacing_zyx, conf, airway, medpseg, tfenet, votes)
        p = topofuse.branch_prob(tab)
        alive = topofuse.prune(tab, p, tau=TAU, protect_gen=3)
        branches_removed = int((~alive[1:]).sum())
        mask2 = mask & alive[bm]
        mask3 = topofuse.refine(mask2, conf, bm, dist, tab, spacing_zyx,
                                kappa=KAPPA, core_min_vox=CORE) > 0
        os.replace(out, os.path.join(d, "union3_lcc_union.nii.gz"))
        write_like(mask3.astype(np.uint8), img, out)
        print(f"topofuse: {branches_removed} of {len(p) - 1} branches removed "
              f"(tau={TAU}), {n_union} -> {int(mask2.sum())} voxels after pruning "
              f"-> {int(mask3.sum())} after boundary refinement "
              f"(kappa={KAPPA}, core={CORE} vox)", flush=True)
        mask = mask3
        conf[~mask] = 0.0

    write_like(np.rint(conf * 100).astype(np.uint8), img, os.path.join(d, "confidence.nii.gz"))
    inside = conf[mask]
    fusion = {"mode": a.fusion, "mask_voxels_union": n_union, "mask_voxels_final": int(mask.sum()),
              "branches_removed": branches_removed,
              "mean_confidence": round(float(inside.mean()), 4) if inside.size else None,
              "fraction_low_confidence": round(float((inside < 0.5).mean()), 4) if inside.size else None}
    json.dump(fusion, open(os.path.join(d, "fusion.json"), "w"), indent=1)
    print(f"confidence: mean {fusion['mean_confidence']}, "
          f"{100 * (fusion['fraction_low_confidence'] or 0):.1f}% of the mask below 0.5", flush=True)

    # Back to the orientation the user uploaded, so the mask overlays their scan.
    code = json.load(open(os.path.join(d, "orient.json")))["source_orientation"]
    write_srcorient(d, "union3_lcc.nii.gz", code)
    write_srcorient(d, "confidence.nii.gz", code)
    print(f"fused airway: {fusion['mask_voxels_final']} voxels", flush=True)


if __name__ == "__main__":
    main()
