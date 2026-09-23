# Run `or_all3_lcc` on the local machine

`or_all3_lcc` is three pipelines plus a fuse step, so all three have to come down:

| tarball | what | size | md5 |
|---|---|---|---|
| `airmorph_seg.tar.gz` | AirMorph code + 6 checkpoints + lungmask | 145 MB | `8ed1d39a68233f25f93eb1690693bdf3` |
| `medpseg_airway.tar.gz` | MEDPSeg code + airway/poly weights | 289 MB | `abd5fd0efb9c68a01d90ecb04c27fd16` |
| `tfenet_bas.tar.gz` | TfeNet code + BAS checkpoints | 63 MB | `929f4ce36d1268ec9523d3ec8aa9ec3e` |

~500 MB total. Excluded: training data, MEDPSeg's `lober.ckpt` (523 MB lobes head, unused
for airway) and `data_poly.zip` (733 MB training data), TfeNet's ATM22 checkpoint
(contaminated — trained on ATM26) and its prebuilt `.so` files, which have to be
recompiled locally anyway.

**Check this first.** TfeNet's DAConv/DSConv are CUDA extensions with no CPU fallback, so
`or_all3_lcc` needs an NVIDIA GPU (~11 GB VRAM at peak) and `nvcc` to build them. Without
one, AirMorph and MEDPSeg can be coaxed onto CPU but TfeNet cannot, and there is no
`or_all3_lcc` without TfeNet.

```bash
nvidia-smi
nvcc --version
```

---

## 1. Download

```bash
mkdir -p ~/or_all3 && cd ~/or_all3
```

```bash
scp <user>@<gpu-server>:<gpu-server-path>/UMUT/transfer/airmorph_seg.tar.gz .
scp <user>@<gpu-server>:<gpu-server-path>/UMUT/transfer/medpseg_airway.tar.gz .
scp <user>@<gpu-server>:<gpu-server-path>/UMUT/transfer/tfenet_bas.tar.gz .
scp <user>@<gpu-server>:<gpu-server-path>/UMUT/transfer/fuse_union3.py .
scp <user>@<gpu-server>:<gpu-server-path>/UMUT/transfer/LOCAL_COMMANDS.md .
```

One line instead of five:

```bash
scp <user>@<gpu-server>:<gpu-server-path>/UMUT/transfer/{airmorph_seg.tar.gz,medpseg_airway.tar.gz,tfenet_bas.tar.gz,fuse_union3.py,LOCAL_COMMANDS.md} .
```

```bash
md5sum *.tar.gz
# 8ed1d39a68233f25f93eb1690693bdf3  airmorph_seg.tar.gz
# abd5fd0efb9c68a01d90ecb04c27fd16  medpseg_airway.tar.gz
# 929f4ce36d1268ec9523d3ec8aa9ec3e  tfenet_bas.tar.gz
```

```bash
tar xzf airmorph_seg.tar.gz
tar xzf medpseg_airway.tar.gz
tar xzf tfenet_bas.tar.gz
ls
```

Gives `AirMorph/`, `medpseg/`, `TfeNet/`, `UMUT/transfer/fuse_union3.py`.

---

## 2. Environments

Three separate envs — they conflict (AirMorph needs monai 1.4.0, TfeNet needs 1.3.2).
Swap `cu121` for the CUDA build matching the local driver.

### 2a. AirMorph

```bash
conda create -n airmorph python=3.10 -y
conda activate airmorph
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
grep -v '^pycuda' ~/or_all3/AirMorph/requirements.txt > /tmp/req_airmorph.txt
pip install -r /tmp/req_airmorph.txt
```

`pycuda` is stripped because it is only needed for branch labeling, not segmentation.

```bash
python -c "
import sys; sys.path.insert(0, '$HOME/or_all3/AirMorph')
from segmentator.airway_segmentator import AirwayAtlasBinaryAirwaySegmentator
print('AirMorph OK')"
conda deactivate
```

### 2b. MEDPSeg

```bash
conda create -n medpseg python=3.10 -y
conda activate medpseg
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
cd ~/or_all3/medpseg && pip install -e .
medpseg --help && conda deactivate
```

### 2c. TfeNet — needs compiling

```bash
conda create -n tfenet python=3.10 -y
conda activate tfenet
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r ~/or_all3/TfeNet/requirements.txt
```

```bash
cd ~/or_all3/TfeNet/DAConv && pip install -e .
cd ~/or_all3/TfeNet/DSConv && pip install -e .
```

```bash
cd ~/or_all3/TfeNet
python -c "import DACONV_CUDA, DSCONV_CUDA; print('TfeNet CUDA ops OK')"
conda deactivate
```

If the build fails, it is almost always a torch/nvcc CUDA version mismatch — make the
torch wheel's CUDA match `nvcc --version`.

---

## 3. Prepare the scans

```bash
export RAW=~/data/raw          # <-- folder of source .nii.gz
export CASES=~/data/cases
conda activate airmorph
```

Check orientation and size. AirMorph and TfeNet both ignore direction cosines and assume
LPS; a non-LPS scan is seen A-P mirrored and every left/right name flips. TfeNet's window
is a 128 cube and fails on any axis smaller than that.

```bash
python - <<'PY'
import glob, os, SimpleITK as sitk
for p in sorted(glob.glob(os.path.join(os.path.expanduser(os.environ["RAW"]), "*.nii.gz"))):
    i = sitk.ReadImage(p)
    code = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(i.GetDirection())
    small = [n for n in i.GetSize() if n < 128]
    print(f"{os.path.basename(p):40s} {code}  size={i.GetSize()}"
          f"{'  <-- AXIS UNDER 128, TfeNet WILL FAIL' if small else ''}")
PY
```

Lay out one folder per case, reorienting to LPS and recording the source code. Safe to run
whether or not the scans are already LPS.

```bash
python - <<'PY'
import glob, json, os, SimpleITK as sitk
raw = os.path.expanduser(os.environ["RAW"]); cases = os.path.expanduser(os.environ["CASES"])
f = sitk.DICOMOrientImageFilter(); f.SetDesiredCoordinateOrientation("LPS")
for p in sorted(glob.glob(os.path.join(raw, "*.nii.gz"))):
    cid = os.path.basename(p)[:-len(".nii.gz")]
    d = os.path.join(cases, cid); os.makedirs(d, exist_ok=True)
    img = sitk.ReadImage(p)
    code = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(img.GetDirection())
    sitk.WriteImage(f.Execute(img), os.path.join(d, "image.nii.gz"), True)
    json.dump({"case": cid, "source": p, "source_orientation": code},
              open(os.path.join(d, "orient.json"), "w"), indent=2)
    print(cid, code, "-> LPS")
PY
```

---

## 4. AirMorph

```bash
cat > ~/or_all3/run_airmorph.py <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "AirMorph"))
from segmentator.airway_segmentator import AirwayAtlasBinaryAirwaySegmentator
seg = AirwayAtlasBinaryAirwaySegmentator()
for case in sys.argv[1:]:
    print("===", case, flush=True)
    seg({"patient": os.path.basename(case), "file_path": case})
PY
```

One case first:

```bash
conda activate airmorph
CUDA_VISIBLE_DEVICES=0 python ~/or_all3/run_airmorph.py $CASES/$(ls $CASES | head -1)
```

Look at it before batching — trachea on the midline, not mirrored:

```bash
itksnap -g $CASES/$(ls $CASES | head -1)/image.nii.gz \
        -s $CASES/$(ls $CASES | head -1)/airway_bin.nii.gz
```

```bash
CUDA_VISIBLE_DEVICES=0 python ~/or_all3/run_airmorph.py $CASES/*
conda deactivate
```

---

## 5. MEDPSeg

`--post` is off by default, which is what the union needs. `--disable_lobe` skips the
lobes head, which is not part of `or_all3_lcc`.

```bash
conda activate medpseg
for d in $CASES/*/; do
  medpseg -i "$d" -o "$d/medpseg_out" --disable_lobe
  cp "$d/medpseg_out/image_airway.nii.gz" "$d/medpseg_airway.nii.gz"
done
conda deactivate
```

If that filename is different on this install, look:

```bash
ls $CASES/*/medpseg_out/ | head
```

---

## 6. TfeNet

BAS checkpoint, and the raw big+small union **before** `postprocessing.py` — the union
prunes once, at the end.

```bash
conda activate tfenet
export TFENET_ROOT=~/or_all3/TfeNet
```

```bash
python - <<'PY'
import glob, os, sys
import numpy as np, torch, SimpleITK as sitk

root = os.path.expanduser(os.environ["TFENET_ROOT"])
cases = os.path.expanduser(os.environ["CASES"])
os.chdir(root); sys.path.insert(0, root)
from importlib import import_module
from data_ATM22 import lumTrans_hu
from utils import load_itk_image

CUBE, STEP, DEV, CK = 128, 64, "cuda:0", "BAS"

def load(p):
    _, net = import_module("TfeNet").get_model()
    net.load_state_dict(torch.load(p, map_location="cpu")["state_dict"])
    return net.to(DEV).eval()

nets = [load(os.path.join(root, "checkpoint", CK, n)) for n in
        ("TfeNet_checkpoint.ckpt", "TfeNetSmall_checkpoint.ckpt")]

def starts(n):
    k = (n - CUBE)//STEP + (1 if (n - CUBE) % STEP == 0 else 2)
    return [min(STEP*i, n - CUBE) for i in range(k)]

@torch.no_grad()
def run(net, vol):
    tot = np.zeros(vol.shape[2:], np.float32); cnt = np.zeros_like(tot)
    vol = vol.to(DEV)
    for x in starts(vol.shape[2]):
        for y in starts(vol.shape[3]):
            for z in starts(vol.shape[4]):
                o = net(vol[:, :, x:x+CUBE, y:y+CUBE, z:z+CUBE].contiguous()).cpu().numpy()[0,0]
                tot[x:x+CUBE, y:y+CUBE, z:z+CUBE] += o
                cnt[x:x+CUBE, y:y+CUBE, z:z+CUBE] += 1
    del vol; torch.cuda.empty_cache()
    return (tot/cnt >= 0.5).astype(np.uint8)

for d in sorted(glob.glob(os.path.join(cases, "*/"))):
    out = os.path.join(d, "tfenet_airway.nii.gz")
    if os.path.exists(out):
        print("skip", d); continue
    ct = os.path.join(d, "image.nii.gz")
    vol = torch.from_numpy(lumTrans_hu(load_itk_image(ct)[0]).astype(np.float32)/255.0)[None, None]
    u = np.zeros(vol.shape[2:], np.uint8)
    for net in nets:
        u |= run(net, vol)
    del vol
    ref = sitk.ReadImage(ct)
    m = sitk.GetImageFromArray(u); m.CopyInformation(ref)
    sitk.WriteImage(m, out, True)
    print(os.path.basename(d.rstrip("/")), int(u.sum()), "voxels")
PY
conda deactivate
```

---

## 7. Fuse — this is `or_all3_lcc`

```bash
conda activate airmorph
for d in $CASES/*/; do
  echo "=== $(basename $d)"
  python ~/or_all3/UMUT/transfer/fuse_union3.py \
    --airmorph "$d/airway_bin.nii.gz" \
    --medpseg  "$d/medpseg_airway.nii.gz" \
    --tfenet   "$d/tfenet_airway.nii.gz" \
    --out      "$d/union3_lcc.nii.gz"
done
```

Default `--connectivity 6` matches the reported scores (DSC 93.06, clDice 90.24,
TLD 95.22, BND 92.31). `--connectivity 26` matches the BCH_LIAM clinical runs. Pick one
and keep it fixed across the cohort.

`SEEDING MATTERED` in the log means TfeNet produced a blob larger than the airway on that
case. The output is still correct — that is exactly the collapse the seeded rule prevents
— but the case is worth a look.

---

## 8. Rotate back to the source orientation

```bash
python - <<'PY'
import glob, json, os, SimpleITK as sitk
for d in sorted(glob.glob(os.path.join(os.path.expanduser(os.environ["CASES"]), "*/"))):
    oj = os.path.join(d, "orient.json")
    if not os.path.exists(oj): continue
    code = json.load(open(oj))["source_orientation"]
    if code == "LPS": continue
    f = sitk.DICOMOrientImageFilter(); f.SetDesiredCoordinateOrientation(code)
    p = os.path.join(d, "union3_lcc.nii.gz")
    sitk.WriteImage(f.Execute(sitk.ReadImage(p)),
                    os.path.join(d, "union3_lcc_srcorient.nii.gz"), True)
    print(os.path.basename(d.rstrip("/")), "->", code)
PY
```

---

## 9. Check

```bash
for d in $CASES/*/; do
  python -c "
import SimpleITK as s, numpy as np
i=s.ReadImage('$d/union3_lcc.nii.gz'); a=s.GetArrayFromImage(i)>0
print(f'$(basename $d)  {a.sum():8d} vox  {a.sum()*np.prod(i.GetSpacing())/1000:7.2f} mL')"
done
```

Adult airway is roughly 20–50 mL. A few mL means something upstream failed.

```bash
itksnap -g $CASES/<case>/image.nii.gz -s $CASES/<case>/union3_lcc.nii.gz
```

---

## 10. Optional — prove the local install is correct

Reproduces a known-good shipped mask bit for bit.

```bash
mkdir -p ~/or_all3/verify && cd ~/or_all3/verify
scp <user>@<gpu-server>:<gpu-server-path>/BCH_LIAM/itk_review/0-1/{14_airway_airmorph,02_airway_medpseg,03_airway_tfenet,19_airway_union3_lcc}.nii.gz .
```

```bash
python ~/or_all3/UMUT/transfer/fuse_union3.py \
  --airmorph 14_airway_airmorph.nii.gz \
  --medpseg  02_airway_medpseg.nii.gz \
  --tfenet   03_airway_tfenet.nii.gz \
  --connectivity 26 \
  --out mine.nii.gz
```

```bash
python -c "
import SimpleITK as s, numpy as np
a=s.GetArrayFromImage(s.ReadImage('mine.nii.gz'))>0
b=s.GetArrayFromImage(s.ReadImage('19_airway_union3_lcc.nii.gz'))>0
print('mine',a.sum(),'shipped',b.sum(),'identical:',np.array_equal(a,b))"
# expect: mine 55522 shipped 55522 identical: True
```

That only exercises the fuse step. To check the whole local install end to end, also pull
`00_CT_source.nii.gz` from the same folder, run sections 4–7 on it, and compare against
`19_airway_union3_lcc.nii.gz` — expect close but not identical, since that mask was built
with the older largest-component rule.
