#!/usr/bin/env python
"""Ablation 9: test-time augmentation (TTA) + model ensemble.

Each of the three members (AirMorph 5-net vote, MEDPSeg, TfeNet big+small) is
run a second time on the left-right mirrored CT and the prediction is mirrored
back, giving 6 binary passes per voxel instead of 3. Fusion candidates are
thresholds on the 6-pass count (k=1 is the union of everything, k=3 a majority
of the six) plus "AND within member, union across members", all followed by the
paper's seeded-component rule. The disagreement between the two passes of a
member is an uncertainty signal evaluated as a false-positive detector inside
the current union mask, against the calibrated vote-pattern confidence.

The GPU stages are subcommands so each runs in its own environment:

  airmorph env:  tta.py airmorph --split test --flip lr      (also prepares the
                 mirrored image.nii.gz + medpseg_in/ in a scratch case dir)
  medpseg  env:  tta.py medpseg  --split test --flip lr      (medpseg CLI per case)
  tfenet   env:  tta.py tfenet   --split test --flip lr      (sliding window, crop
                 to lungs+AirMorph airway with 24 vox margin as the web worker does)
  airmorph env:  tta.py collect  --split test --flip lr      (one npz per case:
                 airmorph_flip, airmorph_votes_flip, medpseg_flip, tfenet_flip)
  airmorph env:  tta.py fuse     --split test [--workers 2]  (sweep + scoring)
  airmorph env:  tta.py uncertainty --split test             (AUROC of TTA disagreement)

`--flip none` runs the unmodified image through the same driver, as a parity
check of this code path against the stored member masks.

The models ignore direction cosines and assume LPS (the ATM volumes are LPS),
so mirroring is a flip of the numpy x axis (axis 2 of the (z, y, x) array) with
the header left untouched; the output array is flipped back on the same axis.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

ROOT = "/fileserver/external/quin/ege"
OUT = os.path.join(ROOT, "fusion_uncertainty_run", "ablation", "tta")
CODE = os.environ.get("OR3_CODE", os.path.join(ROOT, "or_all3_run", "code"))
ENVS = os.path.join(ROOT, "or_all3_run", "envs")
MEDPSEG_BIN = os.path.join(ENVS, "medpseg", "bin", "medpseg")
IMAGES = {"test": os.path.join(ROOT, "fusion_uncertainty_run", "test_images", "{cid}_0000.mha"),
          "val": os.path.join(ROOT, "ATM26_VAL_RUN", "data", "ATM26", "imagesTr", "{cid}_0000.mha")}
FLIP_AXIS = {"none": None, "lr": 2, "ap": 1}       # numpy (z, y, x) axis


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


CACHE_DIRS = {"test": os.path.join(ROOT, "paper", "airway_paper_bundle", "08_masks_and_gt", "gt_cache_centerlines"),
              "val": os.path.join(ROOT, "ATM26_VAL_RUN", "data", "cache", "ATM26")}


def case_list(split, only=None):
    """common.cases(split) without importing common (the medpseg and tfenet
    envs lack its dependencies)."""
    cs = sorted(os.listdir(CACHE_DIRS[split]))
    if only:
        cs = [c for c in cs if c in set(only)]
    return cs


def scratch_dir(split, flip, cid):
    return os.path.join(OUT, "scratch", split, flip, cid)


def member_path(split, flip, cid, stage):
    return os.path.join(OUT, "members", split, f"{cid}.{flip}.{stage}.npz")


def flipped(arr, flip):
    ax = FLIP_AXIS[flip]
    return arr if ax is None else np.ascontiguousarray(np.flip(arr, ax))


def gpu_free_mib():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        used, tot = [int(x) for x in o.strip().split(",")]
        return tot - used, used
    except Exception:
        return None, None


def wait_for_gpu(need_mib, tag):
    """Shared card: do not start a stage while someone else holds most of it."""
    while True:
        free, used = gpu_free_mib()
        if free is None or free >= need_mib:
            log(f"{tag}: GPU used {used} MiB, free {free} MiB")
            return
        log(f"{tag}: only {free} MiB free on the GPU, waiting 60 s")
        time.sleep(60)


def prepare(split, flip, cid):
    """Scratch case dir with the (mirrored) image.nii.gz and medpseg_in/."""
    import SimpleITK as sitk
    d = scratch_dir(split, flip, cid)
    os.makedirs(os.path.join(d, "medpseg_in"), exist_ok=True)
    img_path = os.path.join(d, "image.nii.gz")
    if not os.path.exists(img_path):
        src = sitk.ReadImage(IMAGES[split].format(cid=cid))
        arr = sitk.GetArrayFromImage(src)
        out = sitk.GetImageFromArray(flipped(arr, flip))
        out.CopyInformation(src)
        out = sitk.Cast(out, sitk.sitkInt16)
        tmp = os.path.join(d, ".partial.image.nii.gz")
        sitk.WriteImage(out, tmp, True)
        os.replace(tmp, img_path)
    link = os.path.join(d, "medpseg_in", "image.nii.gz")
    if not os.path.lexists(link):
        os.symlink(os.path.join("..", "image.nii.gz"), link)
    return d


def save_member(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path + ".partial.npz", **arrays)
    os.replace(path + ".partial.npz", path)


# ---------------------------------------------------------------- GPU: AirMorph

def cmd_airmorph(a):
    import SimpleITK as sitk
    import torch
    sys.path.insert(0, os.path.join(CODE, "AirMorph"))
    os.chdir(os.path.join(CODE, "AirMorph"))
    from segmentator import airway_segmentator as seg_mod
    from util.utils import InnerTransformer, restore_image_via_box

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
        return lcc

    seg_mod.AirwayAtlasBinaryAirwaySegmentator._extract_airway = _extract_airway

    todo = [c for c in case_list(a.split, a.cases)
            if a.force or not os.path.exists(member_path(a.split, a.flip, c, "airmorph"))]
    log(f"airmorph {a.split}/{a.flip}: {len(todo)} to run")
    if not todo:
        return
    wait_for_gpu(a.need_gpu_mib, "airmorph")
    t0 = time.time()
    seg = seg_mod.AirwayAtlasBinaryAirwaySegmentator()
    orig_lobe = seg._extract_lobe

    def _extract_lobe(image):
        lobe, crop, bbox = orig_lobe(image)
        captured["bbox"], captured["shape"] = np.array(bbox), image.shape
        return lobe, crop, bbox
    seg._extract_lobe = _extract_lobe
    log(f"airmorph models loaded in {time.time()-t0:.0f}s")

    for i, cid in enumerate(todo, 1):
        ts = time.time()
        d = prepare(a.split, a.flip, cid)
        tp = time.time() - ts
        seg({"patient": cid, "file_path": d})
        votes = restore_image_via_box(captured["shape"], captured["votes"], captured["bbox"])
        raw = (votes >= 3).astype(np.uint8)
        post = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(d, "airway_bin.nii.gz"))) > 0
        peak = torch.cuda.max_memory_allocated() / 2**20
        save_member(member_path(a.split, a.flip, cid, "airmorph"),
                    votes=flipped(votes, a.flip), raw=flipped(raw, a.flip),
                    post=flipped(post.astype(np.uint8), a.flip),
                    seconds=np.float32(time.time() - ts - tp), prepare_seconds=np.float32(tp),
                    gpu_peak_mib=np.float32(peak))
        log(f"airmorph [{i}/{len(todo)}] {cid}: {time.time()-ts-tp:.0f}s (+{tp:.0f}s prepare), "
            f"raw {int(raw.sum())} vox, post {int(post.sum())} vox, GPU peak {peak:.0f} MiB")
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------- GPU: MEDPSeg

class _QueueShim:
    """Stands in for MEDPSeg's GUI queue; progress chatter is dropped."""
    def put(self, item):
        pass


def _medpseg_unflip(array, dir_array):
    dir_array = np.asarray(dir_array)
    if len(dir_array) != 9:
        return array
    axes = np.where(dir_array[[0, 4, 8]][::-1] < 0)[0]
    return array if len(axes) == 0 else np.flip(array, axes).copy()


def cmd_medpseg(a):
    """MEDPSeg airway head, the pipeline loaded once (ATM26_VAL_RUN/training/
    predict_medpseg.py path, which produced the stored masks). `--cli` runs
    the `medpseg` command per case instead (web-worker path; ~2 min slower
    per case because it reloads the models and segments the lungs too)."""
    import SimpleITK as sitk
    todo = [c for c in case_list(a.split, a.cases)
            if a.force or not os.path.exists(member_path(a.split, a.flip, c, "medpseg"))]
    log(f"medpseg {a.split}/{a.flip}: {len(todo)} to run ({'cli' if a.cli else 'in-process'})")
    if not todo:
        return
    if a.cli:
        return _medpseg_cli(a, todo, sitk)
    import torch
    from medpseg.image_read import read_preprocess
    from medpseg.poly_pipeline import PolySegmentationPipeline
    import medpseg
    weight = os.path.join(os.path.dirname(medpseg.__file__), "poly_medseg_25d_fix.ckpt")
    wait_for_gpu(a.need_gpu_mib, "medpseg")
    t0 = time.time()
    model = PolySegmentationPipeline(weight=weight, batch_size=a.batch_size, cpu=False,
                                     output_dir=os.path.join(OUT, "scratch"), post=False, cli=True)
    log(f"medpseg loaded in {time.time()-t0:.0f}s")
    for i, cid in enumerate(todo, 1):
        ts = time.time()
        d = prepare(a.split, a.flip, cid)
        tp = time.time() - ts
        data, _, _, _, directions, original_image = read_preprocess(os.path.join(d, "image.nii.gz"), norm=True)
        dir_array = np.asarray(directions)
        _, airway, _ = model(data.unsqueeze(0).unsqueeze(0), _QueueShim(), None, original_image, cid, dir_array, False)
        m = _medpseg_unflip(airway.astype(np.uint8), dir_array) > 0
        peak = torch.cuda.max_memory_allocated() / 2**20
        save_member(member_path(a.split, a.flip, cid, "medpseg"),
                    raw=flipped(m.astype(np.uint8), a.flip), seconds=np.float32(time.time() - ts - tp),
                    gpu_peak_mib=np.float32(peak))
        log(f"medpseg [{i}/{len(todo)}] {cid}: {time.time()-ts-tp:.0f}s (+{tp:.0f}s prepare), {int(m.sum())} vox, GPU peak {peak:.0f} MiB")
        del data, airway
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def _medpseg_cli(a, todo, sitk):
    for i, cid in enumerate(todo, 1):
        wait_for_gpu(a.need_gpu_mib, "medpseg")
        ts = time.time()
        d = prepare(a.split, a.flip, cid)
        out_dir = os.path.join(d, "medpseg_out")
        proc = subprocess.run([MEDPSEG_BIN, "-i", os.path.join(d, "medpseg_in"), "-o", out_dir,
                               "--disable_lobe"], capture_output=True, text=True)
        txt = proc.stdout + proc.stderr
        with open(os.path.join(d, "medpseg.log"), "w") as fh:
            fh.write(txt)
        produced = os.path.join(out_dir, "image_airway.nii.gz")
        if proc.returncode != 0 or "Aborting processing" in txt or not os.path.exists(produced):
            log(f"medpseg FAILED {cid}: exit {proc.returncode}; see {d}/medpseg.log")
            continue
        m = sitk.GetArrayFromImage(sitk.ReadImage(produced)) > 0
        save_member(member_path(a.split, a.flip, cid, "medpseg"),
                    raw=flipped(m.astype(np.uint8), a.flip), seconds=np.float32(time.time() - ts),
                    gpu_peak_mib=np.float32(0))
        log(f"medpseg [{i}/{len(todo)}] {cid}: {time.time()-ts:.0f}s, {int(m.sum())} vox")
        shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------- GPU: TfeNet

CUBE, STEP, DEV = 128, 64, "cuda:0"


def _tfe_starts(n):
    k = (n - CUBE) // STEP + (1 if (n - CUBE) % STEP == 0 else 2)
    return [min(STEP * i, n - CUBE) for i in range(k)]


def _tfe_run(net, vol, torch):
    tot = np.zeros(vol.shape[2:], np.float32)
    cnt = np.zeros_like(tot)
    vol = vol.to(DEV)
    with torch.no_grad():
        for x in _tfe_starts(vol.shape[2]):
            for y in _tfe_starts(vol.shape[3]):
                for z in _tfe_starts(vol.shape[4]):
                    patch = vol[:, :, x:x + CUBE, y:y + CUBE, z:z + CUBE].contiguous()
                    o = net(patch).float().cpu().numpy()
                    tot[x:x + CUBE, y:y + CUBE, z:z + CUBE] += o[0, 0]
                    cnt[x:x + CUBE, y:y + CUBE, z:z + CUBE] += 1
    del vol
    torch.cuda.empty_cache()
    return (tot / cnt >= 0.5).astype(np.uint8)


def _tfe_crop_box(case_dir, shape, margin, sitk):
    """stage_tfenet.crop_box: lungs + AirMorph airway, margin, >= one window per axis."""
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
        if hi - lo < CUBE:
            c = (lo + hi) // 2
            lo, hi = c - CUBE // 2, c + CUBE - CUBE // 2
        lo, hi = max(lo, 0), min(hi, shape[ax])
        if hi - lo < CUBE:
            lo, hi = 0, shape[ax]
        sl.append(slice(lo, hi))
    return tuple(sl)


def cmd_tfenet(a):
    import torch
    import SimpleITK as sitk
    from importlib import import_module
    TROOT = os.path.join(CODE, "TfeNet")
    os.chdir(TROOT)
    sys.path.insert(0, TROOT)
    from data_ATM22 import lumTrans_hu
    from utils import load_itk_image

    todo = [c for c in case_list(a.split, a.cases)
            if a.force or not os.path.exists(member_path(a.split, a.flip, c, "tfenet"))]
    log(f"tfenet {a.split}/{a.flip}: {len(todo)} to run (crop={'no' if a.no_crop else a.margin})")
    if not todo:
        return
    wait_for_gpu(a.need_gpu_mib, "tfenet")

    def load_net(path):
        _, net = import_module("TfeNet").get_model()
        net.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        return net.to(DEV).eval()
    t0 = time.time()
    nets = [load_net(os.path.join(TROOT, "checkpoint", "BAS", n))
            for n in ("TfeNet_checkpoint.ckpt", "TfeNetSmall_checkpoint.ckpt")]
    log(f"tfenet models loaded in {time.time()-t0:.0f}s")

    for i, cid in enumerate(todo, 1):
        ts = time.time()
        d = prepare(a.split, a.flip, cid)
        full = load_itk_image(os.path.join(d, "image.nii.gz"))[0]
        box = None if a.no_crop else _tfe_crop_box(d, full.shape, a.margin, sitk)
        arr = full[box] if box is not None else full
        frac = arr.size / full.size
        vol = torch.from_numpy(lumTrans_hu(arr).astype(np.float32) / 255.0)[None, None]
        u = np.zeros(vol.shape[2:], np.uint8)
        for net in nets:
            u |= _tfe_run(net, vol, torch)
        out = np.zeros(full.shape, np.uint8)
        if box is not None:
            out[box] = u
        else:
            out = u
        peak = torch.cuda.max_memory_allocated() / 2**20
        save_member(member_path(a.split, a.flip, cid, "tfenet"),
                    raw=flipped(out, a.flip), seconds=np.float32(time.time() - ts),
                    crop_fraction=np.float32(frac), gpu_peak_mib=np.float32(peak))
        log(f"tfenet [{i}/{len(todo)}] {cid}: {time.time()-ts:.0f}s, crop {frac*100:.0f}%, "
            f"{int(out.sum())} vox, GPU peak {peak:.0f} MiB")
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------- collect

def cmd_collect(a):
    """One npz per case with the un-flipped binary masks, and a timing table."""
    import pandas as pd
    rows = []
    for cid in case_list(a.split, a.cases):
        paths = {s: member_path(a.split, a.flip, cid, s) for s in ("airmorph", "medpseg", "tfenet")}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        final = os.path.join(OUT, "members", a.split, f"ATM26_{cid}_{a.flip}.npz")
        am, md, tf = (np.load(paths[s]) for s in ("airmorph", "medpseg", "tfenet"))
        row = {"case": cid, "flip": a.flip,
               "airmorph_s": float(am["seconds"]), "medpseg_s": float(md["seconds"]), "tfenet_s": float(tf["seconds"]),
               "tfenet_crop_frac": float(tf["crop_fraction"]),
               "airmorph_gpu_peak_mib": float(am["gpu_peak_mib"]), "medpseg_gpu_peak_mib": float(md["gpu_peak_mib"]) if "gpu_peak_mib" in md else float("nan"),
               "tfenet_gpu_peak_mib": float(tf["gpu_peak_mib"])}
        row["total_s"] = row["airmorph_s"] + row["medpseg_s"] + row["tfenet_s"]
        rows.append(row)
        if not os.path.exists(final) or a.force:
            save_member(final, **{f"airmorph_{a.flip}": am["raw"], f"airmorph_votes_{a.flip}": am["votes"],
                                  f"medpseg_{a.flip}": md["raw"], f"tfenet_{a.flip}": tf["raw"]})
        # the scratch image is regenerable; the member masks are kept
        img = os.path.join(scratch_dir(a.split, a.flip, cid), "image.nii.gz")
        if os.path.exists(img) and not a.keep_images:
            os.remove(img)
    df = pd.DataFrame(rows)
    out = os.path.join(OUT, "results", f"timing_{a.split}_{a.flip}.csv")
    df.to_csv(out, index=False)
    if len(df):
        log(f"collected {len(df)} cases -> {out}; mean seconds airmorph {df.airmorph_s.mean():.0f}, "
            f"medpseg {df.medpseg_s.mean():.0f}, tfenet {df.tfenet_s.mean():.0f}, total {df.total_s.mean():.0f}")


# ---------------------------------------------------------------- parity check

def cmd_parity(a):
    """Dice between this driver's un-augmented members and the stored masks."""
    import common
    for cid in case_list(a.split, a.cases):
        f = os.path.join(OUT, "members", a.split, f"ATM26_{cid}_none.npz")
        if not os.path.exists(f):
            continue
        c = common.load_cached(a.split, cid)
        p = np.load(f)
        msg = []
        for k in ("airmorph", "medpseg", "tfenet"):
            x, y = p[f"{k}_none"].astype(bool), c[k]
            dice = 2 * (x & y).sum() / max(1, x.sum() + y.sum())
            msg.append(f"{k} Dice {dice:.4f} ({int(x.sum())} vs stored {int(y.sum())})")
        log(f"parity {cid}: " + "; ".join(msg))


# ---------------------------------------------------------------- fusion

def load_tta(split, cid, flip="lr"):
    f = os.path.join(OUT, "members", split, f"ATM26_{cid}_{flip}.npz")
    if not os.path.exists(f):
        return None
    p = np.load(f)
    return {k: p[f"{k}_{flip}"].astype(bool) for k in ("airmorph", "medpseg", "tfenet")} | \
        {"airmorph_votes": p[f"airmorph_votes_{flip}"]}


def strategies(c, t):
    """Candidate masks (uint8) from the 3 original + 3 mirrored passes."""
    import common
    s6 = (c["airmorph"].astype(np.uint8) + c["medpseg"] + c["tfenet"]
          + t["airmorph"] + t["medpseg"] + t["tfenet"])
    seed = c["airmorph"] | t["airmorph"]
    out = {"union_seeded": c["M"].astype(np.uint8)}
    for k in (1, 2, 3, 4):
        out[f"tta_k{k}"] = common.keep_seeded_component(s6 >= k, seed)
    and_union = (c["airmorph"] & t["airmorph"]) | (c["medpseg"] & t["medpseg"]) | (c["tfenet"] & t["tfenet"])
    out["tta_and_union"] = common.keep_seeded_component(and_union, seed)
    # the current union plus what two or more passes confirm (additions only)
    out["tta_union_plus"] = common.keep_seeded_component(c["M"] | (s6 >= 2), seed)
    # the current union minus what no second pass confirms (removals only)
    out["tta_union_minus"] = common.keep_seeded_component(c["M"] & (s6 >= 2), seed)
    return out, s6, seed


_CFG = {}


def _fuse_case(cid):
    import common
    a = _CFG["a"]
    t0 = time.time()
    c = common.load_cached(a.split, cid)
    t = load_tta(a.split, cid, a.tta_flip)
    if t is None:
        return []
    masks, s6, seed = strategies(c, t)
    if a.hybrid_thresholds:
        conf = common.confidence_from_pattern(c["pattern"], "calibrated", common.pattern_table("val"))
        for th in a.hybrid_thresholds:
            keep = (s6 >= 2) | ((s6 == 1) & (conf >= th))
            masks[f"tta_k1_conf{th:.2f}"] = common.keep_seeded_component(keep, seed)
    if a.strategies:
        masks = {k: v for k, v in masks.items() if k in a.strategies}
    rows = []
    for name, m in masks.items():
        rows.append({"split": a.split, "case": cid, "strategy": name,
                     **common.score(m, c, cldice=(name in a.cldice_for))})
    if a.save_name:
        d = os.path.join(OUT, "masks", a.split)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, f"ATM26_{cid}.npz"), mask=masks[a.save_name].astype(np.uint8))
    log(f"fuse {cid}: " + ", ".join(f"{r['strategy']} DSC {r['DSC']:.2f}/BND {r['BND']:.2f}" for r in rows)
        + f" ({time.time()-t0:.0f}s)")
    return rows


def cmd_fuse(a):
    import pandas as pd
    from multiprocessing import Pool
    _CFG["a"] = a
    cs = [c for c in case_list(a.split, a.cases) if load_tta(a.split, c, a.tta_flip) is not None]
    log(f"fuse {a.split}: {len(cs)} cases with TTA members")
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(_fuse_case, cs) for r in rs]
    df = pd.DataFrame(rows).sort_values(["strategy", "case"])
    out = a.out or os.path.join(OUT, "results", f"percase_{a.split}.csv")
    df.to_csv(out, index=False)
    g = df.groupby("strategy")[["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "FPR"]].mean().round(2)
    pd.set_option("display.width", 250)
    print(g.to_string())
    g.to_csv(out.replace(".csv", "_summary.csv"))


# ---------------------------------------------------------------- uncertainty

def _auroc_hist(pos, neg):
    """AUROC of a score where HIGHER bins mean 'more positive' (uncertainty_eval)."""
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    cum_neg = np.cumsum(neg) - neg
    return float(((cum_neg + 0.5 * neg) * pos).sum() / (P * N))


def _unc_case(cid):
    import common
    a = _CFG["a"]
    c = common.load_cached(a.split, cid)
    t = load_tta(a.split, cid, a.tta_flip)
    if t is None:
        return None
    M = c["M"]; gt = c["gt"].astype(bool)
    fp = M & ~gt; tp = M & gt
    s6 = (c["airmorph"].astype(np.uint8) + c["medpseg"] + c["tfenet"]
          + t["airmorph"] + t["medpseg"] + t["tfenet"])[M]
    pat = c["pattern"][M].astype(np.int64)                      # 0..23
    # mirrored-pass pattern: AirMorph votes (flipped) 0..5 * 4 + 2*medpseg + tfenet
    patf = (np.minimum(t["airmorph_votes"], 5).astype(np.int64) * 4 + 2 * t["medpseg"] + t["tfenet"])[M]
    joint = pat * 24 + patf                                     # 576 cells
    fpm, tpm = fp[M], tp[M]
    h = {}
    h["s6_fp"] = np.bincount(s6[fpm], minlength=7)[:7]; h["s6_tp"] = np.bincount(s6[tpm], minlength=7)[:7]
    h["pat_fp"] = np.bincount(pat[fpm], minlength=24)[:24]; h["pat_tp"] = np.bincount(pat[tpm], minlength=24)[:24]
    h["joint_fp"] = np.bincount(joint[fpm], minlength=576); h["joint_tp"] = np.bincount(joint[tpm], minlength=576)
    # agreement of the mirrored pass with the original per member (0..3 members agree)
    agree = ((c["airmorph"] == t["airmorph"]).astype(np.uint8) + (c["medpseg"] == t["medpseg"]) + (c["tfenet"] == t["tfenet"]))[M]
    h["agree_fp"] = np.bincount(agree[fpm], minlength=4)[:4]; h["agree_tp"] = np.bincount(agree[tpm], minlength=4)[:4]
    # calibrated confidence x s6 for a combined score, as a joint hist (24 x 7)
    comb = pat * 7 + s6
    h["comb_fp"] = np.bincount(comb[fpm], minlength=168)[:168]; h["comb_tp"] = np.bincount(comb[tpm], minlength=168)[:168]
    log(f"unc {cid}: mask {int(M.sum())} vox, FP {int(fpm.sum())}")
    return cid, h


def _table_auroc(fp, tp, p):
    """AUROC of (1 - p[cell]) as an FP detector given per-cell FP/TP counts."""
    order = np.argsort(p)               # ascending P(airway) = descending uncertainty
    return _auroc_hist(fp[order][::-1], tp[order][::-1])


def cmd_uncertainty(a):
    import common, pandas as pd
    from multiprocessing import Pool
    _CFG["a"] = a
    cs = [c for c in case_list(a.split, a.cases) if load_tta(a.split, c, a.tta_flip) is not None]
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(_unc_case, cs) if r is not None]
    H = {cid: h for cid, h in res}
    keys = list(next(iter(H.values())).keys())
    tot = {k: sum(h[k] for h in H.values()) for k in keys}
    lut_val = common.pattern_table("val")["p"]
    p_cal = np.nan_to_num(lut_val, nan=0.5)

    def scores(h, calib):
        """calib: dict of P(airway) tables for pattern/joint/comb (fitted elsewhere)."""
        out = {}
        out["s6_vote_fraction"] = _auroc_hist(h["s6_fp"][::-1], h["s6_tp"][::-1])
        out["members_agreeing"] = _auroc_hist(h["agree_fp"][::-1], h["agree_tp"][::-1])
        out["pattern_calibrated_val"] = _table_auroc(h["pat_fp"], h["pat_tp"], p_cal)
        for k in ("pat", "joint", "comb"):
            if k in calib:
                out[f"{k}_calibrated_{calib['src']}"] = _table_auroc(h[f"{k}_fp"], h[f"{k}_tp"], calib[k])
        return out

    def fit(hs):
        c = {}
        for k in ("pat", "joint", "comb"):
            fp = sum(h[f"{k}_fp"] for h in hs); tp = sum(h[f"{k}_tp"] for h in hs)
            with np.errstate(invalid="ignore", divide="ignore"):
                c[k] = np.where(fp + tp > 0, tp / np.maximum(fp + tp, 1), 0.5)
        return c

    rows = []
    # pooled, with the joint tables fitted leave-one-case-out on this split
    # (the val split's TTA may be incomplete, so the fit source is stated)
    for cid, h in H.items():
        calib = fit([v for k, v in H.items() if k != cid]); calib["src"] = f"LOO_{a.split}"
        rows.append({"case": cid, **scores(h, calib)})
    per = pd.DataFrame(rows).sort_values("case")
    per.to_csv(os.path.join(OUT, "results", f"uncertainty_percase_{a.split}.csv"), index=False)
    # pooled: in-sample fit for the pattern-only LUT reproduces the study; the
    # joint tables are reported LOO-pooled by summing LOO-scored histograms is not
    # possible, so the pooled joint number uses the in-sample fit and is flagged.
    calib_in = fit(list(H.values())); calib_in["src"] = f"insample_{a.split}"
    pooled = scores(tot, calib_in)
    summary = {"split": a.split, "n_cases": len(H),
               "mask_precision_%": round(100 * sum(tot[k] for k in ("pat_tp",)).sum() / (tot["pat_tp"].sum() + tot["pat_fp"].sum()), 2),
               "pooled": {k: round(v, 4) for k, v in pooled.items()},
               "percase_mean": {k: round(float(per[k].mean()), 4) for k in per.columns if k != "case"},
               "percase_sd": {k: round(float(per[k].std(ddof=1)), 4) for k in per.columns if k != "case"}}
    # FP capture by the least-certain 5/10/20 % of the mask for the TTA count
    al = tot["s6_fp"] + tot["s6_tp"]
    cum_all = np.cumsum(al) / al.sum(); cum_fp = np.cumsum(tot["s6_fp"]) / tot["s6_fp"].sum()
    summary["s6_FP_capture"] = {f"lowest_{int(k*100)}%": round(100 * float(np.interp(k, np.r_[0.0, cum_all], np.r_[0.0, cum_fp])), 1)
                                for k in (0.05, 0.10, 0.20)}
    summary["s6_reliability"] = {int(k): {"voxels": int(al[k]), "precision_%": round(100 * tot["s6_tp"][k] / max(al[k], 1), 2),
                                          "share_of_FP_%": round(100 * tot["s6_fp"][k] / tot["s6_fp"].sum(), 2)} for k in range(1, 7)}
    ag = tot["agree_fp"] + tot["agree_tp"]
    summary["agree_reliability"] = {int(k): {"voxels": int(ag[k]), "precision_%": round(100 * tot["agree_tp"][k] / max(ag[k], 1), 2),
                                             "share_of_FP_%": round(100 * tot["agree_fp"][k] / tot["agree_fp"].sum(), 2)} for k in range(4)}
    with open(os.path.join(OUT, "results", f"uncertainty_{a.split}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez(os.path.join(OUT, "results", f"uncertainty_hist_{a.split}.npz"), **tot)
    print(json.dumps(summary, indent=2))


def cmd_xauroc(a):
    """Pooled test AUROC with every table fitted on val histograms only."""
    import common
    R = os.path.join(OUT, "results")
    v = dict(np.load(os.path.join(R, "uncertainty_hist_val.npz")))
    t = dict(np.load(os.path.join(R, "uncertainty_hist_test.npz")))
    res = {}
    for k in ("pat", "joint", "comb"):
        fp, tp = v[f"{k}_fp"], v[f"{k}_tp"]
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(fp + tp > 0, tp / np.maximum(fp + tp, 1), 0.5)
        res[f"{k}_calibrated_on_val"] = round(_table_auroc(t[f"{k}_fp"], t[f"{k}_tp"], p), 4)
    res["s6_vote_fraction (no fit)"] = round(_auroc_hist(t["s6_fp"][::-1], t["s6_tp"][::-1]), 4)
    res["members_agreeing (no fit)"] = round(_auroc_hist(t["agree_fp"][::-1], t["agree_tp"][::-1]), 4)
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(R, "uncertainty_test_valfitted.json"), "w"), indent=2)


# ---------------------------------------------------------------- report

METRICS = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision"]


def _ref(split):
    """union_seeded and topofuse (refined, kappa 0.4) per case from the study."""
    import pandas as pd
    f = os.path.join(ROOT, "fusion_uncertainty_run", "results", f"whole_{split}.csv")
    df = pd.read_csv(f)
    return df[df.strategy.isin(["union_seeded", "topofuse"])].replace({"strategy": {"topofuse": "union_refined"}})


def select_rule(summary, ref_union):
    """Brief's rule: max DSC subject to BND loss <= 1.0 point vs the union."""
    ok = summary[(summary.BND >= ref_union["BND"] - 1.0) & (summary.index.str.startswith("tta_"))]
    return None if ok.empty else ok.DSC.idxmax()


def cmd_report(a):
    import pandas as pd
    out = {}
    for split in ("val", "test"):
        fs = sorted(glob.glob(os.path.join(OUT, "results", f"percase_{split}*.csv")))
        fs = [f for f in fs if not f.endswith("_summary.csv")]
        if not fs:
            continue
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        # clDice may exist in one file and be NaN in another: keep the scored row
        df = df.sort_values("clDice", na_position="last").drop_duplicates(["case", "strategy"], keep="first")
        df = df[df.strategy != "union_seeded"]       # the reference file holds it (identical mask)
        if a.strategies:
            df = df[df.strategy.isin(a.strategies)]
        cases = sorted(df.case.unique())
        ref = _ref(split); ref = ref[ref.case.isin(cases)]
        allr = pd.concat([ref, df], ignore_index=True)
        summ = allr.groupby("strategy")[METRICS].mean().round(2)
        summ["n"] = allr.groupby("strategy").size()
        out[split] = (allr, summ)
        print(f"\n## {split} ({len(cases)} cases)\n" + summ.to_string())
        summ.to_csv(os.path.join(OUT, "results", f"summary_{split}.csv"))
    if a.pick is None:
        if "val" in out:
            allr, summ = out["val"]
            pick = select_rule(summ, summ.loc["union_seeded"])
            print(f"\nselection on val ({summ.loc['union_seeded','n']} cases): {pick}")
        else:
            pick = None
    else:
        pick = a.pick
    if pick and "test" in out:
        allr, summ = out["test"]
        w = allr.pivot(index="case", columns="strategy", values=METRICS)
        rows = []
        for refname in ("union_seeded", "union_refined"):
            for m in METRICS:
                d = (w[(m, pick)] - w[(m, refname)]).dropna()
                rows.append({"vs": refname, "metric": m, "mean_diff": round(d.mean(), 2), "sd": round(d.std(ddof=1), 2),
                             "improved": int((d > 0).sum()), "worse": int((d < 0).sum()), "tied": int((d == 0).sum()), "n": len(d)})
        pdf = pd.DataFrame(rows)
        print(f"\n## test: paired {pick} minus reference\n" + pdf.to_string(index=False))
        pdf.to_csv(os.path.join(OUT, "results", f"paired_test_{pick}.csv"), index=False)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("airmorph", "medpseg", "tfenet", "collect", "parity", "fuse", "uncertainty", "report", "xauroc"):
        p = sub.add_parser(name)
        p.add_argument("--split", required=(name not in ("report", "xauroc")), choices=("val", "test"))
        if name == "report":
            p.add_argument("--pick", default=None)
            p.add_argument("--strategies", nargs="*", default=None)
        p.add_argument("--flip", default="lr", choices=tuple(FLIP_AXIS))
        p.add_argument("--cases", nargs="*", default=None)
        p.add_argument("--force", action="store_true")
        p.add_argument("--need-gpu-mib", type=int, default=12000)
        if name == "medpseg":
            p.add_argument("--cli", action="store_true")
            p.add_argument("--batch-size", type=int, default=4)
        if name == "tfenet":
            p.add_argument("--no-crop", action="store_true")
            p.add_argument("--margin", type=int, default=24)
        if name == "collect":
            p.add_argument("--keep-images", action="store_true")
        if name in ("fuse", "uncertainty"):
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--tta-flip", default="lr", choices=tuple(FLIP_AXIS))
        if name == "fuse":
            p.add_argument("--cldice-for", nargs="*", default=[])
            p.add_argument("--hybrid-thresholds", nargs="*", type=float, default=[])
            p.add_argument("--save-name", default=None)
            p.add_argument("--strategies", nargs="*", default=None)
            p.add_argument("--out", default=None)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
