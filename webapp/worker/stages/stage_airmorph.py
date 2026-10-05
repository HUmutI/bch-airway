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

    import numpy as np
    import torch
    from segmentator import airway_segmentator as seg_mod
    from segmentator.airway_segmentator import AirwayAtlasBinaryAirwaySegmentator
    from util.utils import InnerTransformer, restore_image_via_box

    # The five member networks vote; the mask is the >=3 vote. The vote count
    # itself (0..5) is what the confidence map downstream needs, so capture it
    # on the way past. The mask that is written is unchanged.
    captured = {}

    def _extract_airway(self, image):
        a1, a2, a3 = self.airwayextractor.predict(image)
        aw = self.airwayextractor_wingsnet.predict(image)
        ab = self.airwayextractor_break.predict(image)
        ens = a1 + a2 + a3 + aw + ab
        captured["votes"] = np.asarray(InnerTransformer.ToNumpy(ens)[0, ...]).astype(np.uint8)
        ens[ens < 3] = 0
        ens[ens >= 3] = 1
        lcc = InnerTransformer.KeepLargestConnectedComponent(ens)
        lcc = InnerTransformer.CastToNumpyUINT8(InnerTransformer.ToNumpy(lcc)[0, ...])
        torch.cuda.empty_cache()
        self.logger.info("Airway segmentation complete.")
        return lcc

    seg_mod.AirwayAtlasBinaryAirwaySegmentator._extract_airway = _extract_airway

    t0 = time.time()
    print("loading AirMorph ensemble", flush=True)
    seg = AirwayAtlasBinaryAirwaySegmentator()
    orig_lobe = seg._extract_lobe

    def _extract_lobe(image):
        lobe, crop, bbox = orig_lobe(image)
        captured["bbox"], captured["shape"] = np.array(bbox), image.shape
        return lobe, crop, bbox
    seg._extract_lobe = _extract_lobe
    print(f"loaded in {time.time()-t0:.0f}s, segmenting", flush=True)
    seg({"patient": os.path.basename(d), "file_path": d})

    votes = restore_image_via_box(captured["shape"], captured["votes"], captured["bbox"])
    ref = sitk.ReadImage(os.path.join(d, "image.nii.gz"))
    img = sitk.GetImageFromArray(votes)
    img.CopyInformation(ref)
    sitk.WriteImage(img, os.path.join(d, "airmorph_votes.nii.gz"), True)

    out = os.path.join(d, "airway_bin.nii.gz")
    n = int((sitk.GetArrayFromImage(sitk.ReadImage(out)) > 0).sum())
    if n == 0:
        sys.exit("AirMorph produced an empty airway; the fusion would have no seed")
    print(f"airmorph done in {time.time()-t0:.0f}s, {n} seed voxels", flush=True)


if __name__ == "__main__":
    main()
