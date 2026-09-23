#!/usr/bin/env python
"""TfeNet airway segmentation, big and small networks, union of the two.

Same recipe as stage 04 of the cohort run: BAS checkpoints, sliding 128-cube
window at stride 64, overlaps averaged, threshold 0.5, raw union written before
any pruning. The one addition is an optional crop.

WHY CROP. A chest CT is mostly not lung, and a 512x512x900 volume is about 1400
forward passes per network. The airway cannot lie outside the box that holds
the lungs and the airway AirMorph already found, so restricting the sliding
window to that box, with a margin, removes most of the passes. Window origins
inside the box differ from a full-volume run, so voxels near window edges can
differ slightly; the recipe is otherwise identical. Pass --no-crop for the
exact cohort behaviour.

Runs in the TfeNet environment on the GPU. There is no CPU path: the
deformable convolutions ship only CUDA kernels.
    python stage_tfenet.py --case-dir job/case [--no-crop] [--margin 24]
"""
import argparse, os, sys, time
from importlib import import_module
import numpy as np
import torch
import SimpleITK as sitk

CODE = os.environ.get("OR3_CODE", "/fileserver/external/quin/ege/or_all3_run/code")
ROOT = os.path.join(CODE, "TfeNet")
CUBE, STEP, DEV, CK = 128, 64, "cuda:0", "BAS"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from data_ATM22 import lumTrans_hu           # noqa: E402
from utils import load_itk_image             # noqa: E402


def load_net(path):
    _, net = import_module("TfeNet").get_model()
    net.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    return net.to(DEV).eval()


def starts(n):
    k = (n - CUBE) // STEP + (1 if (n - CUBE) % STEP == 0 else 2)
    return [min(STEP * i, n - CUBE) for i in range(k)]


@torch.no_grad()
def run(net, vol, log):
    """Sliding window, one 128-cube per forward pass, averaged over overlaps."""
    tot = np.zeros(vol.shape[2:], np.float32)
    cnt = np.zeros_like(tot)
    vol = vol.to(DEV)
    coords = [(x, y, z) for x in starts(vol.shape[2])
              for y in starts(vol.shape[3]) for z in starts(vol.shape[4])]
    for i, (x, y, z) in enumerate(coords):
        # A slice is a view; the deformable-conv CUDA kernel demands contiguous.
        patch = vol[:, :, x:x + CUBE, y:y + CUBE, z:z + CUBE].contiguous()
        o = net(patch).float().cpu().numpy()
        tot[x:x + CUBE, y:y + CUBE, z:z + CUBE] += o[0, 0]
        cnt[x:x + CUBE, y:y + CUBE, z:z + CUBE] += 1
        if (i + 1) % 50 == 0 or i + 1 == len(coords):
            log(f"  window {i+1}/{len(coords)}")
    del vol
    torch.cuda.empty_cache()
    return (tot / cnt >= 0.5).astype(np.uint8)


def crop_box(case_dir, shape, margin):
    """Bounding box of lungs plus AirMorph airway, in array (z, y, x) order."""
    m = np.zeros(shape, bool)
    for f in ("lunglobe.nii.gz", "airway_bin.nii.gz"):
        p = os.path.join(case_dir, f)
        if os.path.exists(p):
            m |= sitk.GetArrayFromImage(sitk.ReadImage(p)) > 0
    if not m.any():
        return None
    sl = []
    for ax in range(3):
        idx = np.where(m.any(axis=tuple(i for i in range(3) if i != ax)))[0]
        lo, hi = int(idx.min()) - margin, int(idx.max()) + 1 + margin
        # every axis must still hold at least one window
        if hi - lo < CUBE:
            c = (lo + hi) // 2
            lo, hi = c - CUBE // 2, c + CUBE - CUBE // 2
        lo, hi = max(lo, 0), min(hi, shape[ax])
        if hi - lo < CUBE:                       # axis shorter than a cube
            lo, hi = 0, shape[ax]
        sl.append(slice(lo, hi))
    return tuple(sl)


def pad_to_cube(arr):
    pads, crop = [], []
    for n in arr.shape:
        miss = max(0, CUBE - n)
        pads.append((miss // 2, miss - miss // 2))
        crop.append(slice(miss // 2, miss // 2 + n))
    if any(b or a for b, a in pads):
        arr = np.pad(arr, pads, mode="constant", constant_values=-1024)
    return arr, tuple(crop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--margin", type=int, default=24)
    a = ap.parse_args()
    d = os.path.abspath(a.case_dir)
    log = lambda s: print(s, flush=True)

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable; TfeNet has no CPU path")

    t0 = time.time()
    nets = [load_net(os.path.join(ROOT, "checkpoint", CK, n))
            for n in ("TfeNet_checkpoint.ckpt", "TfeNetSmall_checkpoint.ckpt")]
    log(f"loaded TfeNet and TfeNetSmall in {time.time()-t0:.0f}s")

    ct = os.path.join(d, "image.nii.gz")
    full = load_itk_image(ct)[0]
    box = None if a.no_crop else crop_box(d, full.shape, a.margin)
    arr = full[box] if box is not None else full
    if box is not None:
        frac = arr.size / full.size
        log(f"cropped {full.shape} -> {arr.shape} ({frac*100:.0f}% of the volume)")
    arr, uncrop = pad_to_cube(arr)

    vol = torch.from_numpy(lumTrans_hu(arr).astype(np.float32) / 255.0)[None, None]
    u = np.zeros(vol.shape[2:], np.uint8)
    for name, net in zip(("TfeNet", "TfeNetSmall"), nets):
        log(f"{name}:")
        u |= run(net, vol, log)
    u = u[uncrop]

    out = np.zeros(full.shape, np.uint8)
    if box is not None:
        out[box] = u
    else:
        out = u

    ref = sitk.ReadImage(ct)
    img = sitk.GetImageFromArray(out)
    img.CopyInformation(ref)
    tmp = os.path.join(d, ".partial.tfenet_airway.nii.gz")
    sitk.WriteImage(img, tmp, True)
    os.replace(tmp, os.path.join(d, "tfenet_airway.nii.gz"))
    log(f"tfenet done in {time.time()-t0:.0f}s, {int(out.sum())} voxels")


if __name__ == "__main__":
    main()
