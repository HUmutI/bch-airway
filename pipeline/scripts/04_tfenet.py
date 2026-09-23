#!/usr/bin/env python
"""Stage 4 - TfeNet airway segmentation (BAS checkpoints, big + small union).

Faithful to LOCAL_COMMANDS.md section 6: BAS checkpoint, sliding 128-cube with
stride 64, average the overlaps, threshold at 0.5, and OR the big and small
networks together. Crucially this writes the RAW union BEFORE
postprocessing.py - the union prunes once, at the end, in the fuse step, and
TfeNet's detached distal fragments are exactly what the fusion exists to
reconnect (fuse_union3.py:19-23).

This is the expensive stage: a 512x512x901 volume is 14x7x7 = 686 windows per
network, 1372 forward passes per case, ~1.14M across the 849-scan cohort.

TfeNet has no CPU path - DAConv/DSConv raise NotImplementedError on non-CUDA
input (DAConv/DAConv_CUDA.py:27, DSConv/DSConv_CUDA.py:26) and ship only .cu
kernels.

Usage (tfenet venv, GPU required):
    $OR3_ENVS/tfenet/bin/python 04_tfenet.py --cases $OR3_CASES
"""
import argparse
import os
import sys
import time
import traceback
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
    """Window origins covering [0, n) with the last one flush to the end."""
    k = (n - CUBE) // STEP + (1 if (n - CUBE) % STEP == 0 else 2)
    return [min(STEP * i, n - CUBE) for i in range(k)]


def set_in_step(net, n):
    """Widen the deformable convs' internal batch chunking.

    DAConv/DSConv split their input batch into chunks of GET_STEP(batch,
    in_step) - which is the GCD of the two (DAConv/src/config.h:43) - and loop
    over them in C++ (deformable_conv3d.cu:319). model/TfeNet_model.py builds
    every DAConvPack/DSConvPack without in_step, so it defaults to 1
    (DAConv_CUDA.py:'class DAConv' signature) and the kernel processes ONE
    sample per iteration no matter how large the batch is. Feeding it batched
    windows would then only speed up the ordinary Conv3d layers.

    Setting in_step = batch makes the GCD equal the batch, so the whole batch is
    handled in a single pass. This only changes how work is grouped - each
    sample's result is independent of the others - but the im2col buffer scales
    with the chunk width, so it costs GPU memory.
    """
    changed = 0
    for m in net.modules():
        if hasattr(m, "in_step"):
            m.in_step = n
            changed += 1
    return changed


@torch.no_grad()
def run(net, vol, batch=1):
    """Sliding 128-cube inference, averaging the overlaps.

    batch=1 reproduces the original one-window-at-a-time loop exactly: a
    torch.cat of a single slice is that slice. Larger batches change only how
    windows are grouped into forward passes, never the arithmetic per window.
    """
    tot = np.zeros(vol.shape[2:], np.float32)
    cnt = np.zeros_like(tot)
    vol = vol.to(DEV)
    coords = [(x, y, z)
              for x in starts(vol.shape[2])
              for y in starts(vol.shape[3])
              for z in starts(vol.shape[4])]

    i = 0
    while i < len(coords):
        chunk = coords[i:i + batch]
        try:
            patch = torch.cat([vol[:, :, x:x + CUBE, y:y + CUBE, z:z + CUBE]
                               for x, y, z in chunk], 0).contiguous()
            o = net(patch).float().cpu().numpy()
            del patch
        except torch.cuda.OutOfMemoryError:
            # Don't lose the case to a transient OOM on a long unattended run:
            # halve the batch and retry the same windows.
            if batch == 1:
                raise
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            print(f"    CUDA OOM - dropping batch to {batch} and retrying",
                  flush=True)
            continue
        for j, (x, y, z) in enumerate(chunk):
            tot[x:x + CUBE, y:y + CUBE, z:z + CUBE] += o[j, 0]
            cnt[x:x + CUBE, y:y + CUBE, z:z + CUBE] += 1
        i += len(chunk)

    del vol
    torch.cuda.empty_cache()
    return (tot / cnt >= 0.5).astype(np.uint8)


def pad_to_cube(array):
    """Air-pad axes shorter than the model cube and return crop slices."""
    pads = []
    crop = []
    for size in array.shape:
        missing = max(0, CUBE - size)
        before = missing // 2
        after = missing - before
        pads.append((before, after))
        crop.append(slice(before, before + size))
    if any(before or after for before, after in pads):
        array = np.pad(array, pads, mode="constant", constant_values=-1024)
    return array, tuple(crop), pads


def predict(nets, ct, batch, pad_small_axes=False):
    """Big + small network union for one CT path."""
    arr = load_itk_image(ct)[0]
    if min(arr.shape) < CUBE:
        if not pad_small_axes:
            raise RuntimeError(f"axis under {CUBE}: shape {arr.shape}")
        original_shape = arr.shape
        arr, crop, pads = pad_to_cube(arr)
        print(f"    air-padding {original_shape} to {arr.shape}: {pads}", flush=True)
    else:
        crop = tuple(slice(0, size) for size in arr.shape)
    vol = torch.from_numpy(
        lumTrans_hu(arr).astype(np.float32) / 255.0)[None, None]
    u = np.zeros(vol.shape[2:], np.uint8)
    for net in nets:
        u |= run(net, vol, batch)
    del vol
    return u[crop]


def selftest(nets, case_dir, batch, in_step):
    """Prove batching changes speed and nothing else.

    Runs the same case twice - once exactly as the original recipe did
    (batch 1, in_step 1), once with the requested settings - and requires the
    two masks to be bit-identical. Windowed averaging then thresholding at 0.5
    is order-independent, so anything other than an exact match means the
    batched path is wrong and must not be used on the cohort.
    """
    ct = os.path.join(case_dir, "image.nii.gz")
    if not os.path.exists(ct):
        sys.exit(f"no image.nii.gz in {case_dir} - run 01_stage.py first")
    print(f"self-test on {os.path.basename(case_dir.rstrip('/'))}", flush=True)

    set_in_step(nets[0], 1), set_in_step(nets[1], 1)
    t0 = time.time()
    ref = predict(nets, ct, 1)
    t_ref = time.time() - t0
    print(f"  baseline  batch=1  in_step=1  : {t_ref/60:6.2f} min  "
          f"{int(ref.sum())} voxels", flush=True)

    n = set_in_step(nets[0], in_step) + set_in_step(nets[1], in_step)
    t0 = time.time()
    got = predict(nets, ct, batch)
    t_new = time.time() - t0
    print(f"  batched   batch={batch}  in_step={in_step} : {t_new/60:6.2f} min  "
          f"{int(got.sum())} voxels   ({n} deformable convs retuned)", flush=True)

    same = np.array_equal(ref, got)
    diff = int((ref != got).sum())
    print(f"\n  identical: {same}" + ("" if same else f"   ({diff} voxels differ)"))
    print(f"  speedup  : {t_ref/max(t_new, 1e-9):.2f}x")
    if not same:
        sys.exit("\nFAIL - batched output differs from the baseline. Do NOT use "
                 "these settings for the cohort.")
    print("\nPASS - identical mask, safe to use for the cohort.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch", type=int,
                    default=int(os.environ.get("OR3_TFENET_BATCH", "1")),
                    help="128-cube windows per forward pass (1 = original)")
    # MEASURED 2026-08-09 on sub-0428_acc-0488: --batch 4 --in-step 4 produced
    # 466,979 voxels against the baseline's 121,194 (582,723 differing) for a
    # mere 1.14x. in_step > 1 is the first thing that makes GET_STEP return
    # step > 1, activating a reshape path in DAConv_forward_cuda that the
    # shipped checkpoints were never run through. It is WRONG. Do not raise
    # this without re-running --selftest and getting an identical mask.
    ap.add_argument("--in-step", type=int, default=1,
                    help="deformable-conv internal chunk width. KEEP AT 1 - "
                         "in_step>1 gave provably wrong masks in testing")
    ap.add_argument("--selftest", metavar="CASE_DIR",
                    help="compare batched vs original on one case and exit")
    ap.add_argument("--pad-small-axes", action="store_true",
                    help="air-pad axes shorter than 128 for inference, then crop "
                         "the prediction back to the original geometry")
    args = ap.parse_args()
    in_step = args.in_step
    if in_step != 1:
        print(f"WARNING: in_step={in_step} activates the step>1 path in "
              f"DAConv/DSConv, which produced provably wrong masks on "
              f"2026-08-09. Verify with --selftest before trusting any output.",
              flush=True)

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA unavailable. TfeNet's DAConv/DSConv have no CPU "
                 "implementation; there is no or_all3_lcc without TfeNet.")

    if args.selftest:
        nets = [load_net(os.path.join(ROOT, "checkpoint", CK, n))
                for n in ("TfeNet_checkpoint.ckpt", "TfeNetSmall_checkpoint.ckpt")]
        selftest(nets, args.selftest, args.batch, in_step)
        return

    if not args.cases:
        sys.exit("--cases is required unless --selftest is given")
    dirs = [ln.strip() for ln in open(args.cases) if ln.strip()]
    todo = [d for d in dirs
            if args.force or not os.path.exists(os.path.join(d, "tfenet_airway.nii.gz"))]
    print(f"TfeNet: {len(todo)} to run, {len(dirs) - len(todo)} already done",
          flush=True)
    if not todo:
        return

    print("device:", torch.cuda.get_device_name(0), flush=True)

    nets = [load_net(os.path.join(ROOT, "checkpoint", CK, n))
            for n in ("TfeNet_checkpoint.ckpt", "TfeNetSmall_checkpoint.ckpt")]
    n_conv = sum(set_in_step(net, in_step) for net in nets)
    print(f"batch={args.batch}  in_step={in_step}  "
          f"({n_conv} deformable convs retuned)", flush=True)

    t0, failed = time.time(), []
    for i, d in enumerate(todo, 1):
        case = os.path.basename(d.rstrip("/"))
        out = os.path.join(d, "tfenet_airway.nii.gz")
        ct = os.path.join(d, "image.nii.gz")
        print(f"=== [{i}/{len(todo)}] {case}", flush=True)
        ts = time.time()
        try:
            u = predict(nets, ct, args.batch, args.pad_small_axes)

            ref = sitk.ReadImage(ct)
            m = sitk.GetImageFromArray(u)
            m.CopyInformation(ref)
            # Atomic: temp file then rename, so an interrupted write cannot
            # leave a truncated mask that resume treats as a finished case.
            # The temp name must KEEP the .nii.gz suffix - SimpleITK picks its
            # writer from the extension and rejects "....nii.gz.partial".
            tmp = os.path.join(os.path.dirname(out),
                               ".partial." + os.path.basename(out))
            sitk.WriteImage(m, tmp, True)
            os.replace(tmp, out)
            print(f"    {int(u.sum())} voxels, {(time.time()-ts)/60:.1f} min",
                  flush=True)
            del u
        except Exception:
            failed.append(case)
            print(f"    FAILED {case}\n{traceback.format_exc()}", flush=True)
            torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"    [{i}/{len(todo)}] {el/60:.1f} min elapsed, "
              f"~{el/i*(len(todo)-i)/3600:.1f} h left", flush=True)

    print(f"\nTfeNet complete. failed: {len(failed)}")
    for c in failed:
        print("   ", c)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
