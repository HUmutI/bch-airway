# Running `or_all3_lcc` over the airway-dataset cohort

849 scans, 590 subjects, ages 0–18. Everything here is driven by
`scripts/config.sh`; each stage is resumable and skips cases whose output
already exists.

> **Run these OUTSIDE `claude-safe.sh`.** That sandbox uses `--dev /dev`
> (line 22), which builds a minimal `/dev` with no `/dev/nvidia*` nodes, so
> CUDA cannot initialise inside it. The GPU itself is fine — an RTX A5000 at
> PCI `41:00.0`, driver 550.163.01, `nvcc` 12.4. If you want an agent to drive
> the run from inside the sandbox, change that line to `--dev-bind /dev /dev`.

---

## Layout

| path | what |
|---|---|
| `or_all3_run/scripts/` | the pipeline |
| `or_all3_run/code/` | AirMorph, medpseg, TfeNet (extracted by stage 0) |
| `or_all3_run/envs/` | three venvs |
| `or_all3_run/logs/` | per-stage timestamped logs |
| `or_all3_run/manifest.csv` | every scan: geometry, orientation, output dir |
| `or_all3_run/qc.csv` | per-case volumes and flags |
| `airway-seg/` | **results, mirroring the dataset tree** |

Results mirror the dataset one-for-one:

```
airway-dataset/age_00-01/sub-0012/sub-0012_acc-0295_source.nii.gz
airway-seg/    age_00-01/sub-0012/sub-0012_acc-0295/
```

and each case directory holds:

| file | from |
|---|---|
| `image.nii.gz` | stage 1, reoriented to LPS |
| `orient.json` | source path, source orientation, geometry |
| `airway_bin.nii.gz`, `lunglobe.nii.gz`, `airway_skeleton.nii.gz` | AirMorph |
| `medpseg_airway.nii.gz` | MEDPSeg |
| `tfenet_airway.nii.gz` | TfeNet, raw big+small union, pre-postprocessing |
| **`union3_lcc.nii.gz`** | **`or_all3_lcc`, in LPS** |
| **`union3_lcc_srcorient.nii.gz`** | **same mask, rotated back to LAS — use this to overlay on the original scans** |
| `fuse.log` | fusion log, including any `SEEDING MATTERED` warning |

All three member masks are kept deliberately: re-fusing later (different
connectivity, different rule) then costs seconds instead of re-running days of
GPU work.

---

## Procedure

```bash
cd /fileserver/external/quin/ege/or_all3_run/scripts

./00_setup.sh                 # once: extract, build 3 venvs, compile CUDA ops
./run_pipeline.sh pilot       # 13 cases — DO THIS FIRST
```

Then read the stage 6 and 7 output before going further. When it looks right:

```bash
nohup ./run_pipeline.sh all > ../logs/full_$(date +%F).log 2>&1 &
tail -f ../logs/full_$(date +%F).log
```

Resume after any interruption — finished cases are skipped automatically:

```bash
./run_pipeline.sh all         # or: ./run_pipeline.sh all 04   to start at stage 4
```

### Stages

| # | script | env | GPU | what |
|---|---|---|---|---|
| 1 | `01_stage.py` | airmorph | no | mirror the tree, reorient LAS→LPS, build case lists |
| 2 | `02_airmorph.py` | airmorph | yes | 5-model ensemble + lungmask → the fusion seed |
| 3 | `03_medpseg.py` | medpseg | yes | MEDPSeg airway head |
| 4 | `04_tfenet.py` | tfenet | yes | sliding 128-cube, big+small, **the long stage** |
| 5 | `05_fuse.py` | airmorph | no | `fuse_union3.py`, then rotate back to LAS |
| 6 | `06_qc.py` | airmorph | no | volumes, per-bucket outliers |
| 7 | `07_verify_orientation.py` | airmorph | no | proves masks aren't mirrored |

---

## Why the pilot matters

**Every one of the 849 scans is LAS, not LPS.** AirMorph and TfeNet ignore
direction cosines and assume LPS (`LOCAL_COMMANDS.md:134-136`), so an error in
the staging step produces masks mirrored front-to-back. A mirrored airway tree
still looks like an airway tree in a viewer — this failure is silent, and it
would only surface after days of compute.

Stage 7 tests it quantitatively instead of by eye: airway lumen is air
(≈ −1000 HU), so it samples the **original dataset file** at the final mask's
voxels. Correct geometry ⇒ mean HU ≤ −850 with ≥90% of voxels below −800.
A mirrored mask lands on mediastinum and parenchyma and the mean rises sharply.
Because it reads the untouched source scan, it validates the LPS staging and
the rotate-back together.

The 13 pilot cases are chosen to cover the cohort's risks: the whole `age_00-01`
bucket (smallest airways, furthest from these networks' adult training data),
the largest volume in the cohort (`sub-0150_acc-0542`, 823×512×1025 — 2,688
TfeNet windows and the peak-memory case), a non-square in-plane scan, and two
adolescents where volumes should look most adult-like.

---

## Do not batch TfeNet's windows without re-testing

Tried and rejected on 2026-08-09. `04_tfenet.py` takes `--batch` and
`--in-step`; both default to 1, which is the configuration the pilot validated.

`GET_STEP(batch, in_step)` is the GCD of the two (`DAConv/src/config.h:43`), so
with `in_step=1` the deformable convs always process one sample per internal
iteration however large the batch is — meaning window batching alone speeds up
only the ordinary `Conv3d` layers. Raising `in_step` is what would parallelise
them, and it is **wrong**:

```
sub-0428_acc-0488   baseline  batch=1 in_step=1   2.54 min   121,194 voxels
                    batched   batch=4 in_step=4   2.23 min   466,979 voxels
                    582,723 voxels differ; speedup only 1.14x
```

`in_step>1` is the first thing that makes `step>1`, which activates a reshape
path in `DAConv_forward_cuda` the shipped checkpoints were never run through.
The measured ceiling was 1.14x even while computing the wrong answer, so there
is little to win here. If you revisit it, `--selftest CASE_DIR` runs the case
both ways and requires a bit-identical mask:

```bash
../envs/tfenet/bin/python 04_tfenet.py --batch 4 --in-step 1 --selftest <case_dir>
```

## Runtime and storage

TfeNet dominates: a median 512×512×901 volume is 686 windows per network, 1,372
forward passes per case, ~1.14M across the cohort. On one A5000 expect roughly
**1–2 weeks of continuous GPU** for all 849 scans, with TfeNet perhaps 3–6 days
of that. The pilot will give you a real per-case number — multiply it out before
committing.

Storage: ~239 GB for the LPS working copies plus ~20 GB of masks. There is 58 TB
free, so this is not a constraint. If you later want the space back, the
`image.nii.gz` files are regenerable from the dataset at any time.

---

## Decisions baked in

**Connectivity = 6** (`config.sh`, `OR3_CONNECTIVITY`). This is
`fuse_union3.py`'s default and the rule the published `or_all3_lcc` scores were
measured with (DSC 93.06, clDice 90.24, TLD 95.22, BND 92.31). Switch to 26 if
these results must line up with the BCH_LIAM clinical runs — but pick one and
keep it fixed across the whole cohort. Because the member masks are retained,
re-fusing the cohort the other way is a minutes-long job:

```bash
OR3_CONNECTIVITY=26 ./05_fuse.py --cases ../cases_all.txt --force
```

**MEDPSeg input is isolated.** `medpseg -i` takes a *folder* and derives its
output ID from every file basename it finds (`medpseg/pipeline.py:161`). Running
`LOCAL_COMMANDS.md` §5 verbatim after §4 would therefore make MEDPSeg re-segment
AirMorph's `airway_bin.nii.gz` and `lunglobe.nii.gz` as if they were CTs. Stage 1
creates `medpseg_in/` holding only a symlink to `image.nii.gz`, so stage order
cannot matter.

**AirMorph's seed is already pruned.** `airway_bin.nii.gz` is a majority vote
followed by `KeepLargestConnectedComponent`
(`segmentator/airway_segmentator.py:66-73`). `fuse_union3.py`'s docstring warns
against feeding pre-pruned masks and points at a `RUN_ON_TARGET.md` tweak to save
the pre-LCC ensemble — that file is not in the shipped tarball, so we follow
`LOCAL_COMMANDS.md` §4/§7 as written. AirMorph is the fusion *seed*, and a pruned
seed still seeds the correct component; what is lost is only AirMorph-only distal
fragments, which MEDPSeg and TfeNet largely re-supply. Worth revisiting if you
can get `RUN_ON_TARGET.md`.

**QC thresholds are relative, not clinical.** `LOCAL_COMMANDS.md:353` offers
"adult airway ≈ 20–50 mL", which does not transfer to ages 0–18. Stage 6 instead
flags anything under 0.5 mL absolutely, and otherwise scores each case against
its own age bucket by robust z-score (median/MAD, |z| > 3.5).

---

## Known data quirks

- **All 849 scans are LAS.** Handled; see above.
- **No scan has an axis under 128**, so there are no TfeNet size failures.
  Stage 4 still checks per case and fails that case loudly rather than silently.
- **Mixed dtypes.** 797 scans are int16 with `scl_inter = -1024`; 52 are uint16
  with `scl_inter = -8192`. SimpleITK applies the scaling, so HU should come out
  right, but the −8192 group is unusual — stage 7's mean-HU numbers are a free
  check that windowing behaved on those scans too.
- **39 subjects span more than one age bucket.** Each *scan* still lives in
  exactly one bucket, so the mirrored tree is unambiguous; just don't assume one
  subject means one directory.

---

## If something fails

Each stage prints `FAILED <case>` with a traceback and continues to the next
case, then exits non-zero with a summary. Fix the cause and re-run the same
command — completed cases are skipped.

```bash
grep -c FAILED ../logs/04_tfenet_*.log
grep -A5 FAILED ../logs/04_tfenet_*.log | head -40
```

Cases where fusion logged `SEEDING MATTERED` are collected in
`seeding_mattered.txt`. Those outputs are correct — that warning is the seeded
rule doing its job, catching a TfeNet blob larger than the airway — but they are
worth a look (`LOCAL_COMMANDS.md:316-318`).

Build failures in stage 0 are almost always a torch/nvcc CUDA mismatch
(`LOCAL_COMMANDS.md:121`). `config.sh` pins torch cu124 against the installed
nvcc 12.4; if you change one, change the other.
