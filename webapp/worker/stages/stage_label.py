#!/usr/bin/env python
"""Skeletonise the fused airway, cut it into branches, and name each branch.

The classification half of stage 09 of the cohort run, without the morphometry
export the website does not need. Writes, with CASE the case directory name:
    union3_lcc_skeleton.nii.gz      TEASAR centreline of the fused mask
    CASE_parse.nii.gz               branch instance id per airway voxel
    CASE_skel_parsing.nii.gz        the skeleton cut into branches
    CASE_pred_lob.nii.gz            lobar label per voxel
    CASE_pred_seg.nii.gz            segmental label per voxel
    CASE_pred_sub.nii.gz            subsegmental head, see the paper's caveat
    CASE_anno.json                  branch id -> [lobar, segmental, subsegmental]
plus the graph and feature arrays. A small labels.json is added next to them
so the viewer can name colours without loading AirMorph.

Runs in the AirMorph environment. The classifier needs CUDA and PyCUDA.
    python stage_label.py --case-dir job/case --class2anno .../class2anno.json
"""
import argparse, json, os, sys, time
import numpy as np
import SimpleITK as sitk

CODE = os.environ.get("OR3_CODE", "/fileserver/external/quin/ege/or_all3_run/code")
sys.path.insert(0, os.path.join(CODE, "AirMorph"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--class2anno", required=True)
    a = ap.parse_args()
    d = os.path.abspath(a.case_dir)
    case = os.path.basename(d)
    t0 = time.time()

    from classifier.airway_classifier import AirwayAtlasMultiAnatomyAirwayClassifier
    from util.airwayatlas_utils import IMRAirwayAtlas_SkeletonExtraction

    mask_img = sitk.ReadImage(os.path.join(d, "union3_lcc.nii.gz"))
    mask = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
    lobe = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(d, "lunglobe.nii.gz"))).astype(np.uint8)
    spacing_xyz = np.asarray(mask_img.GetSpacing(), dtype=float)

    print("skeletonising", flush=True)
    skel = IMRAirwayAtlas_SkeletonExtraction(mask, spacing_xyz)
    if skel is None or not np.any(skel):
        sys.exit("skeleton extraction returned nothing")
    si = sitk.GetImageFromArray(skel.astype(np.uint8)); si.CopyInformation(mask_img)
    sitk.WriteImage(si, os.path.join(d, "union3_lcc_skeleton.nii.gz"), True)
    print(f"skeleton in {time.time()-t0:.0f}s, classifying branches", flush=True)

    AirwayAtlasMultiAnatomyAirwayClassifier()({
        "patient": case, "file_path": d,
        "bin_airway": mask, "lobe": lobe, "skeleton": skel,
        "itk_meta": {"origin": mask_img.GetOrigin(),
                     "spacing": mask_img.GetSpacing(),
                     "direction": mask_img.GetDirection()},
    })

    anno = json.load(open(os.path.join(d, f"{case}_anno.json")))
    print(f"labelled {len(anno)} branches in {time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
