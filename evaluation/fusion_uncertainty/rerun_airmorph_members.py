#!/usr/bin/env python
"""Re-run AirMorph on a case and keep its five member votes, not just the vote.

The cohort run saved only AirMorph's final mask (>=3 of 5 members, largest
component). The uncertainty map needs the five member masks themselves. This
re-runs the unmodified segmentator with `_extract_airway` wrapped so the five
binary member predictions are captured, and writes them, together with the raw
>=3 vote and the post-processed mask, to an npz on the full image grid, in the
same layout as the ATM26 npz files (`member_*`, `raw`, `post`).

The case directory is not touched: the segmentator runs in a scratch directory
holding a symlink to image.nii.gz. The regenerated `post` mask is compared with
the stored airway_bin.nii.gz and the agreement is printed, as a determinism
check.

Runs in the AirMorph environment on the GPU.
    python rerun_airmorph_members.py --case-dir .../sub-XXXX_acc-YYYY --out dir/
"""
import argparse, os, sys, tempfile, time
import numpy as np
import SimpleITK as sitk

CODE = os.environ.get("OR3_CODE", "/fileserver/external/quin/ege/or_all3_run/code")
sys.path.insert(0, os.path.join(CODE, "AirMorph"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = os.path.abspath(a.case_dir)
    cid = os.path.basename(d)
    out = os.path.join(a.out, f"{cid}.npz")
    if os.path.exists(out):
        print(f"{cid}: exists, skipping"); return

    import torch
    from segmentator import airway_segmentator as seg_mod
    from util.utils import restore_image_via_box, InnerTransformer

    captured = {}

    def _extract_airway(self, image):
        a1, a2, a3 = self.airwayextractor.predict(image)
        aw = self.airwayextractor_wingsnet.predict(image)
        ab = self.airwayextractor_break.predict(image)
        members = [a1, a2, a3, aw, ab]
        cpu = [np.asarray(InnerTransformer.ToNumpy(m)[0, ...] > 0.5, dtype=np.uint8) for m in members]
        ens = a1 + a2 + a3 + aw + ab
        ens[ens < 3] = 0; ens[ens >= 3] = 1
        raw = np.asarray(InnerTransformer.ToNumpy(ens)[0, ...] > 0, dtype=np.uint8)
        lcc = InnerTransformer.KeepLargestConnectedComponent(ens)
        lcc = InnerTransformer.CastToNumpyUINT8(InnerTransformer.ToNumpy(lcc)[0, ...])
        captured["members"] = cpu; captured["raw"] = raw
        torch.cuda.empty_cache()
        return lcc

    seg_mod.AirwayAtlasBinaryAirwaySegmentator._extract_airway = _extract_airway

    # the segmentator's __call__ writes into file_path, so give it a scratch dir
    with tempfile.TemporaryDirectory(dir=a.out) as tmp:
        os.symlink(os.path.join(d, "image.nii.gz"), os.path.join(tmp, "image.nii.gz"))
        t0 = time.time()
        s = seg_mod.AirwayAtlasBinaryAirwaySegmentator()
        orig_lobe = s._extract_lobe

        def _extract_lobe(image):
            lobe, crop, bbox = orig_lobe(image)
            captured["bbox"] = np.array(bbox); captured["shape"] = image.shape
            return lobe, crop, bbox
        s._extract_lobe = _extract_lobe
        s({"patient": cid, "file_path": tmp})
        post = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(tmp, "airway_bin.nii.gz"))) > 0

    shape, bbox = captured["shape"], captured["bbox"]
    names = ["member_model1", "member_model2", "member_model3", "member_wingsnet", "member_break"]
    payload = {n: restore_image_via_box(shape, m, bbox) for n, m in zip(names, captured["members"])}
    payload["raw"] = restore_image_via_box(shape, captured["raw"], bbox)
    payload["post"] = post.astype(np.uint8)
    payload["original_shape"] = np.array(shape)

    ref = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(d, "airway_bin.nii.gz"))) > 0
    inter = int((ref & post).sum()); dice = 2 * inter / max(1, int(ref.sum()) + int(post.sum()))
    votes = sum(payload[n].astype(np.int32) for n in names)
    assert np.array_equal((votes >= 3).astype(np.uint8), payload["raw"]), "vote reconstruction mismatch"
    np.savez_compressed(out + ".partial.npz", **payload)
    os.replace(out + ".partial.npz", out)
    print(f"{cid}: done in {time.time()-t0:.0f}s; post vs stored airway_bin Dice {dice:.4f} "
          f"({int(post.sum())} vs {int(ref.sum())} voxels); members "
          f"{[int(payload[n].sum()) for n in names]}", flush=True)


if __name__ == "__main__":
    main()
