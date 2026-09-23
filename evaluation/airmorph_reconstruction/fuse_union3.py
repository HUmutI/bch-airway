# -*- coding: utf-8 -*-
"""or_all3_lcc: OR the three airway masks, then keep ONE connected component.

    union = AirMorph | MEDPSeg | TfeNet
    out   = the 6-connected component of `union` that contains the AirMorph tree

This is the whole of or_all3_lcc. The three members are produced by their own pipelines;
this script only combines them. Scored on ATM26 test (30 cases): DSC 93.06, clDice 90.24,
TLD 95.22, BND 92.31, sensitivity 98.23, precision 88.71 - the best variant measured.

WHY "the component containing AirMorph" AND NOT "the largest component".
On 3 of 30 ATM26 test cases (ATM_507, ATM_509, ATM_627) TfeNet emits a spurious blob that
is both larger than the airway tree and disjoint from it. The largest-component rule keeps
the blob and deletes the entire airway - measured intersection with the airway: 0 voxels,
every metric 0.0. Over all 30 cases x 4 union variants the two rules pick the SAME
component on 111 of 120 pairs, so seeding costs nothing and removes a silent total-failure
mode. Full write-up: RESEARCH_LOG.md, "Fusion study correction" (2026-08-07).

LCC IS APPLIED ONCE, HERE, TO THE FINISHED UNION. Do not feed it masks that have already
been pruned per method: TfeNet is noisy but reaches further distally, and its detached
fragments are exactly what the union exists to reconnect. In particular AirMorph's shipped
`airway_bin.nii.gz` is already post-LCC (segmentator/airway_segmentator.py:70-73) - see
RUN_ON_TARGET.md for the two-line change that also saves the pre-LCC ensemble.

ORIENTATION. All three masks must share one voxel grid; the script refuses to run
otherwise. AirMorph and TfeNet both ignore direction cosines and assume LPS, so if your
CT is not LPS their output is mirrored. Check first:
    python -c "import SimpleITK as s;i=s.ReadImage('ct.nii.gz');print(s.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(i.GetDirection()))"

Usage:
    python fuse_union3.py --airmorph a.nii.gz --medpseg m.nii.gz --tfenet t.nii.gz \
        --out union3_lcc.nii.gz
"""
import argparse, logging, os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

log = logging.getLogger("fuse")


def geom(img):
    return (img.GetSize(),
            tuple(round(x, 6) for x in img.GetSpacing()),
            tuple(round(x, 4) for x in img.GetOrigin()),
            tuple(round(x, 4) for x in img.GetDirection()))


def read_bin(path, name):
    if not os.path.exists(path):
        raise SystemExit(f"missing {name}: {path}")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    vals = np.unique(arr)
    if not np.isin(vals, [0, 1]).all():
        log.warning("%s is not strictly 0/1 (values %s) - thresholding > 0",
                    name, vals[:8])
    return (arr > 0).astype(np.uint8), img


def seeded_component(union, seed, connectivity):
    """Keep the component of `union` holding the most `seed` voxels."""
    structure = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    labelled, count = ndimage.label(union, structure=structure)
    if count == 0:
        raise SystemExit("union is empty")
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())

    seed_labels = labelled[seed.astype(bool)]
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        log.warning("seed does not intersect the union at all - falling back to largest")
        chosen = largest
    else:
        chosen = int(np.bincount(seed_labels).argmax())

    if chosen != largest:
        log.warning("SEEDING MATTERED: largest component (%d vox) is NOT the airway; "
                    "kept the seeded one (%d vox). The largest-component rule would "
                    "have deleted the airway on this case.",
                    int(sizes[largest]), int(sizes[chosen]))
    return (labelled == chosen).astype(np.uint8), count, int(sizes[chosen])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--airmorph", required=True, help="AirMorph ensemble mask (the seed)")
    ap.add_argument("--medpseg", required=True)
    ap.add_argument("--tfenet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--connectivity", type=int, default=6, choices=[6, 26],
                    help="6 = the rule the reported or_all3_lcc numbers were measured "
                         "with; 26 = MONAI KeepLargestConnectedComponent(connectivity=3), "
                         "what the BCH_LIAM clinical runs used")
    ap.add_argument("--save-union", default=None,
                    help="also write the pre-LCC union here")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s")

    members, ref, ref_img = {}, None, None
    for name, path in (("airmorph", args.airmorph),
                       ("medpseg", args.medpseg),
                       ("tfenet", args.tfenet)):
        arr, img = read_bin(path, name)
        g = geom(img)
        if ref is None:
            ref, ref_img = g, img
        elif g != ref:
            raise SystemExit(
                f"grid mismatch for {name}:\n  {g}\n  != {ref}\n"
                "All three masks must sit on the same voxel grid. Resample or rerun "
                "the offending member on the same CT file.")
        members[name] = arr
        log.info("%-9s %9d voxels", name, int(arr.sum()))

    union = (members["airmorph"] | members["medpseg"] | members["tfenet"]).astype(np.uint8)
    out, ncomp, kept = seeded_component(union, members["airmorph"], args.connectivity)

    vox_mm3 = float(np.prod(ref_img.GetSpacing()))
    log.info("union %d voxels in %d components (%d-conn)", int(union.sum()), ncomp,
             args.connectivity)
    log.info("kept  %d voxels = %.3f mL (%.1f%% of the union)",
             kept, kept * vox_mm3 / 1000.0, 100.0 * kept / max(int(union.sum()), 1))

    def write(arr, path):
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.CopyInformation(ref_img)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sitk.WriteImage(img, path, True)
        log.info("wrote %s", path)

    if args.save_union:
        write(union, args.save_union)
    write(out, args.out)


if __name__ == "__main__":
    main()
