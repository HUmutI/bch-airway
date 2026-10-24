# The three models

The pipeline runs three published airway segmentation models and one
published labelling architecture. Their code and trained weights ship as
three tarballs, ~520 MB, too large for git. They live as assets of the
GitHub Release `models-v1` of this repository (on both the organisation
repo and the mirror).

```bash
gh auth login                          # once, your own account, with access to the repo
./models/download_models.sh            # -> models/tarballs/, verifies md5
# MODELS_REPO=HUmutI/bch-airway ./models/download_models.sh   to use the mirror
```

| tarball | contents | size | md5 |
|---|---|---|---|
| `airmorph_seg.tar.gz` | AirMorph code, 5 segmentation checkpoints, lungmask `unet_r231`, labelling checkpoint + `configs/class2anno.json` | 151 MB | `8ed1d39a68233f25f93eb1690693bdf3` |
| `medpseg_airway.tar.gz` | MEDPSeg code and `poly_medseg_25d_fix.ckpt` (airway head; the lobe head is left out on purpose) | 302 MB | `abd5fd0efb9c68a01d90ecb04c27fd16` |
| `tfenet_bas.tar.gz` | TfeNet code, DAConv/DSConv CUDA sources, BAS checkpoints big + small | 66 MB | `929f4ce36d1268ec9523d3ec8aa9ec3e` |

`LOCAL_COMMANDS.md` is the original set-up note written when the tarballs
were first assembled: what was excluded from each upstream release and why,
how each model is invoked, and the pitfalls (LPS orientation, MEDPSeg reading
a whole folder, TfeNet's CUDA build). `pipeline/scripts/00_setup.sh` is the
executable version of it.

## Local changes to upstream code

Everything the lab changed in the upstream code is either a `sed` line in
`00_setup.sh` / `webapp/deploy/Dockerfile.worker`, or in one patch file:

- **`webapp/deploy/airmorph_cpu_path.patch`**: AirMorph's tree parsing and
  feature extraction import `pycuda` at module level. The patch makes it
  optional and adds NumPy equivalents of the kernels, so branch labelling
  runs without pycuda (and on CPU). Written 2026-09-01. On izmir the patched
  copy is `or_all3_run/code/AirMorph`; the tarball is upstream. Regenerate
  with `diff -ruN <upstream> <lab copy> -x __pycache__ -x '*.pyc' -x '*.pth' -x '*.pt' -x '*.ckpt'`
  if the lab copy changes.
- **MEDPSeg `setup.py`**: the `lober.ckpt` existence check is commented out
  (the lobe head is not shipped) and `SimpleITK==2.0.2` becomes `2.4.1` (no
  cp310 wheel for 2.0.2).
- **AirMorph environment**: `pycuda` and `pyradiomics` dropped from
  `requirements.txt`; `torch-geometric==2.6.1` and `openpyxl` added (the
  labelling code imports the first, the morphometry scripts the second;
  neither is in upstream's list).
- **Per-environment CUDA builds**: AirMorph and MEDPSeg get torch cu121
  (their pinned CUDA runtime libs are 12.1), TfeNet gets cu124 to match the
  nvcc that compiles its ops. `config.sh` explains the failure mode if you
  mix them.

## Licences

What the lab wrote is MIT licensed (`LICENSE` at the repository root). That
includes the branch-labelling side of AirMorph: the authors published the
architecture but not the training procedure or the weights, so the trainer,
the loss, the data handling (`evaluation/airmorph_reconstruction/`) and the
labelling checkpoint are the lab's own work.

The tarballs also carry third-party material under its authors' terms: the
MEDPSeg and TfeNet code and weights, and the AirMorph segmentation code with
its five checkpoints and the `lungmask` model. The AirMorph code is
redistributed with its authors' permission; the weights trained on ATM'22/ATM'26
data are CC BY-NC 4.0 (`LICENSE-WEIGHTS.md`); everything else is MIT or
Apache-2.0 (`NOTICE.md`, `webapp/deploy/THIRD_PARTY_NOTICES.md`).

## Provenance

The tarballs were assembled from the upstream releases on the METU server
(`LOCAL_COMMANDS.md` §1) in July 2026 and copied to izmir as
`/fileserver/external/quin/ege/all_3_lcc/`. The Release assets are those
files unchanged; the md5s above are the ones recorded at assembly.
