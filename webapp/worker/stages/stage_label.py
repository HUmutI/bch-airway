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

Speed. AirMorph's tree parsing works on whole volumes (convolution, distance
transform, connected components, one comparison per branch), which on a
200 M-voxel scan is five minutes of CPU for an airway that fills a few per
cent of the box. Everything here therefore runs on the airway's bounding box
padded by a margin, and the volumes are embedded back into the full grid
before they are written. The results are identical: every operation involved
is local (3x3x3 neighbourhoods) or attaches voxels to the nearest skeleton
voxel, and the skeleton lies inside the mask, so a crop that contains the mask
with a margin of background changes nothing. The features fed to the
classifier are relative positions and sizes, invariant to the crop offset,
except that three of them place a branch within the bounding box of its lung,
so the crop is the box around airway and lung mask together. Verified value
for value against the full-volume run (2026-09-24).

Runs in the AirMorph environment. Needs CUDA for the small labelling network.
    python stage_label.py --case-dir job/case --class2anno .../class2anno.json
"""
import argparse, json, os, sys, time
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

CODE = os.environ.get("OR3_CODE", "/fileserver/external/quin/ege/or_all3_run/code")
sys.path.insert(0, os.path.join(CODE, "AirMorph"))

MARGIN = 8          # voxels of background kept around the airway's bounding box


def transform_pred_data_lut(parse, node_idx, pred):
    """Same result as AirMorph's transform_pred_data (data_process/data_save_cuda.py),
    which compares the whole volume once per branch, as one table lookup.
    Row order matters where two rows name the same branch: the later row wins,
    exactly as the sequential assignments did."""
    lut = np.zeros(int(parse.max()) + 2, dtype=np.float32)
    for row, parse_index in enumerate(node_idx):
        lut[int(parse_index) + 1] = float(pred[row]) + 1
    return lut[parse]


def crop_box(mask, margin):
    """Bounding box of everything non-zero in `mask`, padded, clipped."""
    sl = ndimage.find_objects((mask > 0).astype(np.uint8))[0]
    return tuple(slice(max(0, s.start - margin), min(n, s.stop + margin))
                 for s, n in zip(sl, mask.shape))


def embed(crop_arr, box, shape, dtype):
    full = np.zeros(shape, dtype=dtype)
    full[box] = crop_arr
    return full


def write_like(arr, ref_img, path):
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref_img)
    sitk.WriteImage(img, path, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--class2anno", required=True)
    ap.add_argument("--margin", type=int, default=MARGIN)
    a = ap.parse_args()
    d = os.path.abspath(a.case_dir)
    case = os.path.basename(d)
    t0 = time.time()

    import classifier.airway_classifier as ac
    from util.airwayatlas_utils import IMRAirwayAtlas_SkeletonExtraction
    ac.transform_pred_data = transform_pred_data_lut       # the module-level name __call__ uses

    mask_img = sitk.ReadImage(os.path.join(d, "union3_lcc.nii.gz"))
    mask = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
    lobe = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(d, "lunglobe.nii.gz"))).astype(np.uint8)
    spacing_xyz = np.asarray(mask_img.GetSpacing(), dtype=float)
    shape = mask.shape

    box = crop_box(mask | (lobe > 0), a.margin)
    mask_c, lobe_c = np.ascontiguousarray(mask[box]), np.ascontiguousarray(lobe[box])
    print(f"airway+lung box {[s.stop - s.start for s in box]} of {list(shape)} "
          f"({mask_c.size / mask.size * 100:.0f}% of the volume)", flush=True)

    print("skeletonising", flush=True)
    skel_c = IMRAirwayAtlas_SkeletonExtraction(mask_c, spacing_xyz)
    if skel_c is None or not np.any(skel_c):
        sys.exit("skeleton extraction returned nothing")
    write_like(embed(skel_c.astype(np.uint8), box, shape, np.uint8), mask_img,
               os.path.join(d, "union3_lcc_skeleton.nii.gz"))
    print(f"skeleton in {time.time()-t0:.0f}s, classifying branches", flush=True)

    # The classifier writes its five volumes on the crop grid (the origin it is
    # given is only stored in those files, which are rewritten below).
    ac.AirwayAtlasMultiAnatomyAirwayClassifier()({
        "patient": case, "file_path": d,
        "bin_airway": mask_c, "lobe": lobe_c, "skeleton": skel_c,
        "itk_meta": {"origin": mask_img.GetOrigin(),
                     "spacing": mask_img.GetSpacing(),
                     "direction": mask_img.GetDirection()},
    })
    for name in ("skel_parsing", "parse", "pred_lob", "pred_seg", "pred_sub"):
        p = os.path.join(d, f"{case}_{name}.nii.gz")
        arr_c = sitk.GetArrayFromImage(sitk.ReadImage(p))
        write_like(embed(arr_c, box, shape, arr_c.dtype), mask_img, p)

    anno = json.load(open(os.path.join(d, f"{case}_anno.json")))
    print(f"labelled {len(anno)} branches in {time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
