"""Ablation 10: learned error correction of the union with a tiny 3-D CNN.

A two-level U-Net (16/32 channels) sees, per voxel, the CT (HU clipped to
[-1024, 400], scaled to [0, 1]), the three member masks (AirMorph, MEDPSeg,
TfeNet), AirMorph's internal vote count / 5 and, for variant A, the union mask
M itself; it predicts P(airway). Variant A (refinement) adds a fixed logit
offset 4 * (2M - 1) to a zero-initialised head, so the network starts out
reproducing the union exactly and learns only the correction. Variant B
(stacking) drops M and the offset and learns the fusion from scratch.

Training on the 30 ATM'22 validation cases only: 64^3 patches drawn around the
union, BCE + soft Dice, Adam. Model selection (epoch, threshold) by 5-fold
cross-validation by case within val under the brief's rule (maximise DSC with
BND loss <= 1 point vs the union, on full volumes with sliding-window
inference and the seeded-component rule); then one model is trained on all 30
val cases with the chosen schedule and run once on test and pediatric.

Subcommands (run with the airmorph env python, cwd anywhere):
  prep <split>        crop CT + members + GT to the padded union bbox -> npz
  cv  --variant A|B   5-fold CV, writes per-checkpoint prob crops
  cveval --variant    threshold sweep over the CV probs (CPU, 3 workers)
  final --variant     train on all val, run on test + pediatric, score, save masks
"""
import argparse, json, os, sys, time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402

OUT = os.path.join(common.STUDY, "ablation", "refine_cnn")
PREP = os.path.join(OUT, "prep")
PAD = 24           # voxels of context around the union bbox
HU_LO, HU_HI = -1024.0, 400.0
PATCH = 64
LOGIT_M = 4.0      # fixed residual offset of variant A

CT_PATH = {
    "val": os.path.join(common.VAL_RUN, "data/ATM26/imagesTr/{cid}_0000.mha"),
    "test": os.path.join(common.STUDY, "test_images/{cid}_0000.mha"),
}


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(OUT, "log.txt"), "a") as fh:
        fh.write(line + "\n")


# ----------------------------------------------------------------- prep

def ct_array(split, cid):
    import SimpleITK as sitk
    if split == "pediatric":
        p = os.path.join(common.ped_case_dir(cid), "image.nii.gz")
    else:
        p = CT_PATH[split].format(cid=cid)
    return sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.int16)


def prep_case(split, cid):
    f = os.path.join(PREP, split, cid + ".npz")
    if os.path.exists(f):
        return
    c = common.load_cached(split, cid)
    ct = ct_array(split, cid)
    assert ct.shape == c["gt"].shape, (cid, ct.shape, c["gt"].shape)
    idx = np.where(c["M"] | c["gt"].astype(bool))
    sl = tuple(slice(max(0, int(i.min()) - PAD), min(n, int(i.max()) + PAD + 1))
               for i, n in zip(idx, c["gt"].shape))
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez_compressed(f, ct=ct[sl], pattern=c["pattern"][sl],
                        airmorph=c["airmorph"][sl].astype(np.uint8), gt=c["gt"][sl],
                        lo=np.array([s.start for s in sl]), hi=np.array([s.stop for s in sl]),
                        shape=np.array(c["gt"].shape), spacing=c["spacing"])


def load_prep(split, cid):
    with np.load(os.path.join(PREP, split, cid + ".npz")) as z:
        return {k: z[k] for k in z.files}


def channels(d, variant):
    """(C, z, y, x) float32 input from a prep dict."""
    pat = d["pattern"]
    M = pat != 255
    votes = np.where(M, pat // 4, 0).astype(np.float32) / 5.0
    med = np.where(M, (pat // 2) % 2, 0).astype(np.float32)
    tfe = np.where(M, pat % 2, 0).astype(np.float32)
    ct = (np.clip(d["ct"].astype(np.float32), HU_LO, HU_HI) - HU_LO) / (HU_HI - HU_LO)
    ch = [ct, d["airmorph"].astype(np.float32), med, tfe, votes]
    if variant == "A":
        ch.append(M.astype(np.float32))
    return np.stack(ch, 0)


# ----------------------------------------------------------------- network

def build_net(variant):
    import torch, torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.InstanceNorm3d(o, affine=True),
                             nn.LeakyReLU(0.01, inplace=True),
                             nn.Conv3d(o, o, 3, padding=1), nn.InstanceNorm3d(o, affine=True),
                             nn.LeakyReLU(0.01, inplace=True))

    class TinyUNet(nn.Module):
        def __init__(self, cin, residual):
            super().__init__()
            self.residual = residual
            self.enc1 = block(cin, 16)
            self.down = nn.Conv3d(16, 32, 2, stride=2)
            self.enc2 = block(32, 32)
            self.up = nn.ConvTranspose3d(32, 16, 2, stride=2)
            self.dec = block(32, 16)
            self.head = nn.Conv3d(16, 1, 1)
            if residual:
                nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down(e1))
            d = self.dec(torch.cat([self.up(e2), e1], 1))
            out = self.head(d)
            if self.residual:
                out = out + LOGIT_M * (2.0 * x[:, -1:] - 1.0)
            return out

    return TinyUNet(6 if variant == "A" else 5, residual=(variant == "A"))


# ----------------------------------------------------------------- training

class PatchSampler:
    """Holds the prepped crops of the training cases and draws 64^3 patches:
    40 % centred on a union voxel, 40 % on a union/GT disagreement voxel,
    20 % anywhere in the crop; random flips."""

    def __init__(self, cids, variant, rng):
        self.rng = rng
        self.variant = variant
        self.cases = []
        for cid in cids:
            d = load_prep("val", cid)
            x = channels(d, variant)
            y = d["gt"].astype(np.float32)
            M = d["pattern"] != 255
            err = M ^ d["gt"].astype(bool)
            self.cases.append({"x": x, "y": y, "M": np.argwhere(M), "err": np.argwhere(err),
                               "shape": np.array(y.shape)})

    def patch(self):
        c = self.cases[self.rng.integers(len(self.cases))]
        u = self.rng.random()
        if u < 0.4 and len(c["M"]):
            ctr = c["M"][self.rng.integers(len(c["M"]))]
        elif u < 0.8 and len(c["err"]):
            ctr = c["err"][self.rng.integers(len(c["err"]))]
        else:
            ctr = np.array([self.rng.integers(n) for n in c["shape"]])
        lo = np.clip(ctr - PATCH // 2, 0, np.maximum(c["shape"] - PATCH, 0))
        sl = tuple(slice(int(a), int(a) + PATCH) for a in lo)
        x = c["x"][(slice(None),) + sl]; y = c["y"][sl]
        if x.shape[1:] != (PATCH,) * 3:                       # crop thinner than a patch
            px = np.zeros((x.shape[0],) + (PATCH,) * 3, np.float32); py = np.zeros((PATCH,) * 3, np.float32)
            s = tuple(slice(0, n) for n in y.shape)
            px[(slice(None),) + s] = x; py[s] = y; x, y = px, py
        for ax in range(3):
            if self.rng.random() < 0.5:
                x = np.flip(x, ax + 1); y = np.flip(y, ax)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def batch(self, n):
        xs, ys = zip(*[self.patch() for _ in range(n)])
        return np.stack(xs), np.stack(ys)[:, None]


def loss_fn(logit, y):
    import torch, torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logit, y)
    p = torch.sigmoid(logit)
    inter = (p * y).sum(dim=(1, 2, 3, 4)); den = p.sum(dim=(1, 2, 3, 4)) + y.sum(dim=(1, 2, 3, 4))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return bce + dice


def train(cids, variant, epochs, iters_per_epoch, batch, lr, seed, ckpt_epochs, ckpt_fn):
    """ckpt_fn(epoch, net) is called after each epoch listed in ckpt_epochs."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    sampler = PatchSampler(cids, variant, rng)
    log(f"  sampler ready ({len(cids)} cases) in {time.time() - t0:.0f}s")
    net = build_net(variant).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    total = epochs * iters_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (1 - it / total) ** 0.9)
    for ep in range(1, epochs + 1):
        net.train(); run = 0.0; t1 = time.time()
        for _ in range(iters_per_epoch):
            x, y = sampler.batch(batch)
            x = torch.from_numpy(x).cuda(non_blocking=True); y = torch.from_numpy(y).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = net(x)
            loss = loss_fn(logit.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item()
        log(f"  ep {ep:3d} loss {run / iters_per_epoch:.4f}  {time.time() - t1:.0f}s  "
            f"gpu {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
        if ep in ckpt_epochs:
            ckpt_fn(ep, net)
    return net


# ----------------------------------------------------------------- inference

def predict_crop(net, x, win=96, stride=64):
    """Sliding-window P(airway) over a (C, z, y, x) crop; mean of overlapping windows."""
    import torch
    C, Z, Y, X = x.shape
    shp = np.array([Z, Y, X]); pad = np.maximum(win - shp, 0)
    xp = np.pad(x, ((0, 0),) + tuple((0, int(p)) for p in pad)) if pad.any() else x
    Zp, Yp, Xp = xp.shape[1:]
    prob = torch.zeros((Zp, Yp, Xp), dtype=torch.float32, device="cuda")
    cnt = torch.zeros((Zp, Yp, Xp), dtype=torch.float32, device="cuda")
    xt = torch.from_numpy(np.ascontiguousarray(xp)).cuda()

    def starts(n):
        s = list(range(0, max(n - win, 0) + 1, stride))
        if s[-1] + win < n:
            s.append(n - win)
        return s
    net.eval()
    with torch.no_grad():
        for z in starts(Zp):
            for y in starts(Yp):
                for xx in starts(Xp):
                    sl = (slice(None), slice(z, z + win), slice(y, y + win), slice(xx, xx + win))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = net(xt[sl][None])[0, 0]
                    prob[sl[1:]] += torch.sigmoid(lg.float()); cnt[sl[1:]] += 1
    prob = (prob / cnt).cpu().numpy()[:Z, :Y, :X]
    return prob


def core_mask(c, core_frac=0.5):
    """Union voxels closer to the union skeleton than core_frac of their branch
    radius (at least one voxel): the topology guard of refine.py, re-used so the
    CNN may only edit the boundary shell and never cut a branch."""
    r = c["tab"]["r_mean_mm"].to_numpy(np.float32).copy(); r[0] = 0
    vox = float(np.max(c["spacing"]))
    return c["M"] & (c["dist"] <= np.maximum(core_frac * r[c["branch_map"]], vox))


def full_mask(d, prob_crop, thr, airmorph_full, core=None):
    """Threshold, embed in the full grid, optionally add the union core,
    seeded component (seed = AirMorph)."""
    full = np.zeros(tuple(d["shape"]), np.uint8)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(d["lo"], d["hi"]))
    full[sl] = prob_crop >= thr
    if core is not None:
        full |= core.astype(np.uint8)
    return common.keep_seeded_component(full, airmorph_full)


def save_prob(f, prob):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez_compressed(f, prob=np.round(prob * 255).astype(np.uint8))


def load_prob(f):
    with np.load(f) as z:
        return z["prob"].astype(np.float32) / 255.0


# ----------------------------------------------------------------- CV

def folds(cids, k=5):
    return [[c for i, c in enumerate(cids) if i % k == f] for f in range(k)]


def cmd_cv(a):
    cids = common.cases("val")
    ck = [int(e) for e in a.ckpt.split(",")]
    for f, held in enumerate(folds(cids)):
        tr = [c for c in cids if c not in held]
        log(f"[cv {a.variant}] fold {f}: train {len(tr)}, held {held}")
        held_d = {cid: load_prep("val", cid) for cid in held}

        def ckpt_fn(ep, net):
            for cid, d in held_d.items():
                f_out = os.path.join(OUT, "cv", a.variant, f"ep{ep}", cid + ".npz")
                if not os.path.exists(f_out):
                    save_prob(f_out, predict_crop(net, channels(d, a.variant)))
        train(tr, a.variant, a.epochs, a.iters, a.batch, a.lr, seed=1000 + f, ckpt_epochs=ck, ckpt_fn=ckpt_fn)


def _eval_one(args):
    variant, cid, eps, thrs, guards = args
    d = load_prep("val", cid); c = common.load_cached("val", cid)
    rows = []
    cores = {g: core_mask(c, g) for g in guards if g > 0}
    for ep in eps:
        prob = load_prob(os.path.join(OUT, "cv", variant, f"ep{ep}", cid + ".npz"))
        for thr in thrs:
            for guard in guards:
                m = full_mask(d, prob, thr, c["airmorph"], cores.get(guard))
                s = common.score(m, c, cldice=False)
                rows.append(dict(variant=variant, case=cid, epoch=ep, thr=thr, guard=guard, **s))
    return rows


def cmd_cveval(a):
    import pandas as pd
    from multiprocessing import Pool
    eps = [int(e) for e in a.ckpt.split(",")]
    thrs = [float(t) for t in a.thr.split(",")]
    guards = [float(g) for g in a.guard.split(",")]
    cids = common.cases("val")
    with Pool(a.workers) as pool:
        rows = sum(pool.map(_eval_one, [(a.variant, cid, eps, thrs, guards) for cid in cids]), [])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"cv_{a.variant}{a.tag}_percase.csv"), index=False)
    base = pd.concat([pd.read_csv(os.path.join(common.CACHE, "val", cid, "baseline_score.csv")).assign(case=cid)
                      for cid in cids])
    if "method" in base.columns:
        base = base[base["method"] == base["method"].iloc[0]]
    log(f"[cveval {a.variant}] baseline columns {list(base.columns)}")
    uni = {"DSC": base["DSC"].mean(), "BND": base["BND"].mean()}
    g = df.groupby(["epoch", "thr", "guard"])[["DSC", "BND", "TLD", "Sensitivity", "Precision"]].mean().reset_index()
    g["dDSC"] = g["DSC"] - uni["DSC"]; g["dBND"] = g["BND"] - uni["BND"]
    g.to_csv(os.path.join(OUT, f"cv_{a.variant}{a.tag}_summary.csv"), index=False)
    ok = g[g["dBND"] >= -1.0]
    pick = ok.sort_values("DSC", ascending=False).iloc[0] if len(ok) else None
    log(f"[cveval {a.variant}] union val DSC {uni['DSC']:.2f} BND {uni['BND']:.2f}\n" + g.to_string())
    log(f"[cveval {a.variant}] pick: {None if pick is None else pick.to_dict()}")
    if pick is not None:
        with open(os.path.join(OUT, f"cv_{a.variant}{a.tag}_pick.json"), "w") as fh:
            json.dump({"epoch": int(pick["epoch"]), "thr": float(pick["thr"]), "guard": float(pick["guard"]),
                       "val_cv_DSC": float(pick["DSC"]),
                       "val_cv_BND": float(pick["BND"])}, fh, indent=1)


# ----------------------------------------------------------------- final

def cmd_final(a):
    import torch, pandas as pd
    with open(os.path.join(OUT, f"cv_{a.variant}_pick.json")) as fh:
        pick = json.load(fh)
    ep, thr, guard = pick["epoch"], pick["thr"], pick.get("guard", 0)
    cids = common.cases("val")
    wfile = os.path.join(OUT, f"final_{a.variant}_ep{ep}.pt")
    if os.path.exists(wfile):
        net = build_net(a.variant).cuda(); net.load_state_dict(torch.load(wfile))
        log(f"[final {a.variant}] loaded {wfile}")
    else:
        log(f"[final {a.variant}] training on all 30 val cases for {ep} epochs, thr {thr}")
        t0 = time.time()
        net = train(cids, a.variant, ep, a.iters, a.batch, a.lr, seed=7, ckpt_epochs=[], ckpt_fn=None)
        torch.save(net.state_dict(), wfile)
        log(f"[final {a.variant}] train time {time.time() - t0:.0f}s")
    rows = []
    for split in a.splits.split(","):
        for cid in common.cases(split):
            d = load_prep(split, cid); c = common.load_cached(split, cid)
            t0 = time.time()
            x = channels(d, a.variant)
            torch.cuda.synchronize(); t1 = time.time()
            prob = predict_crop(net, x)
            torch.cuda.synchronize(); t_gpu = time.time() - t1
            m = full_mask(d, prob, thr, c["airmorph"], core_mask(c, guard) if guard > 0 else None)
            t_tot = time.time() - t0
            s = common.score(m, c, cldice=True)
            rows.append(dict(variant=a.variant, split=split, case=cid, sec_total=t_tot, sec_gpu=t_gpu, **s))
            log(f"[final {a.variant}] {split} {cid} " + " ".join(f"{k} {v:.2f}" for k, v in s.items()) +
                f"  ({t_tot:.1f}s, gpu {t_gpu:.1f}s)")
            if split != "val":
                f = os.path.join(OUT, "masks", split, f"ATM26_{cid}.npz") if split == "test" else \
                    os.path.join(OUT, "masks", split, f"{cid}.npz")
                os.makedirs(os.path.dirname(f), exist_ok=True)
                np.savez_compressed(f, mask=m.astype(np.uint8))
                save_prob(os.path.join(OUT, "probs", split, cid + ".npz"), prob)
    df = pd.DataFrame(rows)
    out = os.path.join(OUT, f"final_{a.variant}_percase.csv")
    if os.path.exists(out):
        old = pd.read_csv(out); old = old[~old["split"].isin(df["split"].unique())]
        df = pd.concat([old, df])
    df.to_csv(out, index=False)
    log(f"[final {a.variant}]\n" + df.groupby("split")[["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision"]]
        .mean().round(2).to_string())


# ----------------------------------------------------------------- report

COLS = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision"]


def cmd_report(a):
    """Tables for RESULTS.md: means per split with the union / refined rows, and
    paired per-case differences on test. Also scores a secondary operating point
    (no guard, thr 0.5: the DSC-maximal CV point) from the saved test probability
    maps, labelled as such."""
    import pandas as pd
    ref = {sp: pd.read_csv(os.path.join(os.path.dirname(HERE), "results", f"whole_{sp}.csv"))
           for sp in ("val", "test", "pediatric")}
    out = []
    for variant in a.variant.split(","):
        f = os.path.join(OUT, f"final_{variant}_percase.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        with open(os.path.join(OUT, f"cv_{variant}_pick.json")) as fh:
            pick = json.load(fh)
        cvs = pd.read_csv(os.path.join(OUT, f"cv_{variant}_summary.csv"))
        cvrow = cvs[(cvs.epoch == pick["epoch"]) & (np.isclose(cvs.thr, pick["thr"])) & (np.isclose(cvs.guard, pick["guard"]))].iloc[0]
        out.append(f"\n### Variant {variant}: epoch {pick['epoch']}, thr {pick['thr']}, guard core_frac {pick['guard']}\n")
        out.append("| split | method | DSC | clDice | TLD | BND | Sens | Prec |\n|---|---|---|---|---|---|---|---|")
        out.append(f"| val (5-fold CV, no clDice) | refine_cnn {variant} | {cvrow.DSC:.2f} | - | {cvrow.TLD:.2f} | {cvrow.BND:.2f} | {cvrow.Sensitivity:.2f} | {cvrow.Precision:.2f} |")
        for sp in ("val", "test", "pediatric"):
            for strat, name in (("union_seeded", "union, seeded"), ("topofuse", "union, refined")):
                r = ref[sp][ref[sp].strategy == strat][COLS].mean()
                out.append(f"| {sp} | {name} | " + " | ".join(f"{r[k]:.2f}" for k in COLS) + " |")
            if sp in df.split.unique():
                r = df[df.split == sp][COLS].mean()
                out.append(f"| {sp} | refine_cnn {variant} | " + " | ".join(f"{r[k]:.2f}" for k in COLS) + " |")
        # paired differences on test
        t = df[df.split == "test"].set_index("case")
        out.append(f"\nPaired per-case differences on test, refine_cnn {variant} minus reference (n={len(t)}):\n")
        out.append("| vs | metric | mean diff | cases improved | cases worse |\n|---|---|---|---|---|")
        for strat, name in (("union_seeded", "union"), ("topofuse", "refined")):
            r = ref["test"][ref["test"].strategy == strat].set_index("case").loc[t.index]
            for k in ("DSC", "clDice", "TLD", "BND"):
                d = t[k] - r[k]
                out.append(f"| {name} | {k} | {d.mean():+.2f} | {(d > 0).sum()} | {(d < 0).sum()} |")
        if "pediatric" in df.split.unique():
            t = df[df.split == "pediatric"].set_index("case")
            out.append(f"\nPaired per-case differences on pediatric, refine_cnn {variant} minus reference (n={len(t)}):\n")
            out.append("| vs | metric | mean diff | cases improved | cases worse |\n|---|---|---|---|---|")
            for strat, name in (("union_seeded", "union"), ("topofuse", "refined")):
                r = ref["pediatric"][ref["pediatric"].strategy == strat].set_index("case").loc[t.index]
                for k in ("DSC", "clDice", "TLD", "BND"):
                    d = t[k] - r[k]
                    out.append(f"| {name} | {k} | {d.mean():+.2f} | {(d > 0).sum()} | {(d < 0).sum()} |")
        out.append(f"\nCompute per scan (test, mean): total {df[df.split == 'test'].sec_total.mean():.1f} s wall "
                   f"(GPU sliding window {df[df.split == 'test'].sec_gpu.mean():.1f} s, rest = channel build, "
                   f"threshold, seeded component on CPU), excluding loading the CT and member masks.")
        if a.secondary:
            thr2, ep2 = 0.5, pick["epoch"]
            rows = []
            for sp in ("test",):
                for cid in common.cases(sp):
                    d = load_prep(sp, cid); c = common.load_cached(sp, cid)
                    prob = load_prob(os.path.join(OUT, "probs", sp, cid + ".npz"))
                    m = full_mask(d, prob, thr2, c["airmorph"])
                    rows.append(dict(split=sp, case=cid, **common.score(m, c, cldice=True)))
            s2 = pd.DataFrame(rows); s2.to_csv(os.path.join(OUT, f"final_{variant}_secondary_noguard_thr0.5_percase.csv"), index=False)
            r = s2[COLS].mean()
            out.append(f"\nSecondary (DSC-maximal CV point, no guard, thr {thr2}; same test probability maps, no extra tuning): "
                       f"test | " + " | ".join(f"{k} {r[k]:.2f}" for k in COLS))
    txt = "\n".join(out)
    with open(os.path.join(OUT, "report_tables.md"), "w") as fh:
        fh.write(txt)
    print(txt)


# ----------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("split"); p.add_argument("--workers", type=int, default=3)
    for name in ("cv", "cveval", "final", "report"):
        p = sub.add_parser(name)
        p.add_argument("--variant", required=True, help="A or B (report: comma-separated)")
        p.add_argument("--epochs", type=int, default=20)
        p.add_argument("--iters", type=int, default=150)
        p.add_argument("--batch", type=int, default=4)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--ckpt", default="5,10,15,20")
        p.add_argument("--thr", default="0.05,0.1,0.2,0.3,0.5")
        p.add_argument("--guard", default="0,0.5,0.75", help="core_frac of the union core kept; 0 = none")
        p.add_argument("--workers", type=int, default=3)
        p.add_argument("--tag", default="", help="suffix for the cveval output files")
        p.add_argument("--secondary", action="store_true", help="report: also score the unguarded thr-0.5 point")
        p.add_argument("--splits", default="test,pediatric")
    a = ap.parse_args()
    if a.cmd == "prep":
        from multiprocessing import Pool
        cids = common.cases(a.split)
        with Pool(a.workers) as pool:
            pool.starmap(prep_case, [(a.split, cid) for cid in cids])
        log(f"prep {a.split}: {len(cids)} cases")
    elif a.cmd == "cv":
        cmd_cv(a)
    elif a.cmd == "cveval":
        cmd_cveval(a)
    elif a.cmd == "final":
        cmd_final(a)
    elif a.cmd == "report":
        cmd_report(a)
