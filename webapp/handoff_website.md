# Handoff: airway segmentation web app

Written 2026-09-22, updated 2026-09-23, for whoever picks this up next. Only the website is
covered here. The paper, the cohort run and the GitHub repos are separate
work and are not described.

## What it is

A site where lab radiologists upload a chest CT (NIfTI only) and get back the
full pipeline output of the airway method: fused airway mask, lobar and
segmental labels, named branches, an ITK-SNAP-style viewer, and airway
morphometry. It runs the real models, nothing is approximated.

Lives at `/fileserver/external/quin/ege/airway_web/` (a symlink to `bch-airway/webapp/`). Currently served from
the lab GPU machine izmir at `http://izmir:8000`. The professor has said izmir
cannot be used and no tunnels are allowed, so the next step is moving it to
AWS. That work is prepared but not executed (see "Where it stands").

Owner: Umut Isik (umut.isik@childrens.harvard.edu). Users: lab radiologists.
Rules from the professor: research use only with warnings on every page, no
PHI, de-identified NIfTI only, UI simple and professional, **no blue colours**.

## Layout

```
airway_web/
  config.py              every path and switch; all overridable by AIRWAY_* env vars
  run_api.sh             uvicorn on 0.0.0.0:8000, runs anywhere
  run_worker.sh          GPU worker, must run OUTSIDE the sandbox; --mock fakes the GPU stages
  api/app.py             FastAPI: upload, status, retry, log, result files, job list
  api/cloud.py           starts the EC2 GPU instance when a job is queued (no-op unless AIRWAY_WORKER_INSTANCE_ID set)
  worker/worker.py       polls jobs/ for state=queued, claims, runs; --exit-when-idle MIN, --retain-days N
  worker/pipeline.py     stage sequence, subprocess spawning, progress, process-group kill
  worker/stages/         one script per stage, each in the conda/venv it needs
  web/index.html         upload page (banner, drop zone, acknowledgement checkbox)
  web/job.html           progress + viewer + morphometry page
  web/app.js             all front-end logic (upload, polling, NiiVue viewer, morphometry tables)
  web/app.css            palette: bg #f6f4f0, text #1f1d1a, accent terracotta #b5533c, ok green, no blue
  web/palette.json       the ONE colour table for labels (used by JS and by the packaging stage)
  deploy/                Docker images, compose, Caddy, systemd unit, AWS runbook, vendor tarballs
  jobs/<id>/             one directory per upload: input.nii.gz, status.json, case/, logs/, results/
  .venv/                 FastAPI + uvicorn only (the model envs live elsewhere)
```

The three model environments are **not** in this directory:
`/fileserver/external/quin/ege/or_all3_run/envs/{airmorph,medpseg,tfenet}`,
built by `or_all3_run/scripts/00_setup.sh`. They have conflicting
dependencies, so every stage is a subprocess in its own interpreter.

## Pipeline, stage by stage

| stage | env | GPU | what | time on A5000, 512×512×794 |
|---|---|---|---|---|
| prepare | airmorph | no | validate 3D int scan, HU sanity check, reorient to LPS, write `case/image.nii.gz`, `orient.json` | 20 s |
| airmorph | airmorph | yes | AirMorph 5-member vote: `airway_bin`, `lunglobe`, `airway_skeleton` | 3 min |
| medpseg | medpseg | yes | MEDPSeg CLI `--disable_lobe` → `medpseg_airway.nii.gz` | 3.5 min |
| tfenet | tfenet | yes | TfeNet big+small, 128-cube stride 64, cropped to lung∪airway box + 24 vox | 4.3 min |
| fuse | airmorph | no | `all_3_lcc/fuse_union3.py`: union, seeded 6-connected component → `union3_lcc.nii.gz` | 14 s |
| label | airmorph | no (CPU path) | TEASAR skeleton + graph-transformer classifier → `case_parse`, `case_pred_lob/seg`, `case_anno.json` | 5.5 min |
| package | airmorph | no | rotate everything to the upload's orientation, display copies, meshes, ITK-SNAP label files, `summary.json`, zip | 25 s |
| morphometry | airmorph | no | per-branch and central-airway measurements, non-fatal | 3.3 min (16 s for 7 branches) |

medpseg and tfenet run concurrently (`PARALLEL_MEMBERS`, 11+5 GB on a 24 GB
card). Total to viewer ≈ 14 min; morphometry fills in afterwards.

`status.json` carries `state` (queued/claimed/running/done/failed),
`progress`, `message`, `stages[]` with per-stage state and timestamps,
`results` (= summary.json) and `results_ready: true` as soon as package is
done. The job page opens the viewer on `results_ready`, not on `done`.
Morphometry failure sets `morphometry_error` and the job still ends `done`.

## Result files (`jobs/<id>/results/`)

Full resolution, in the upload's orientation: `image`, `airway_mask`,
`airway_lobar`, `airway_segmental`, `airway_branches`, `airway_centreline`
(all `.nii.gz`). Label volumes store class+1, 0 = background. Lobar values:
1 trachea, 2 Left Upper, 3 Left Lower, 4 Right Upper, 5 Right Middle,
6 Right Lower. `labels.json` maps voxel value → name and branch id → names.

`*_view.nii.gz`: the same, resampled to an isotropic grid of at most
`VIEW_MAX_VOXELS` (60 M) voxels, because NiiVue keeps the CT as RGBA8 on the
GPU (4 bytes/voxel). These are what the browser loads. Not in the zip.

`mesh_airway.mz3`, `mesh_lobe_<k>.mz3`: marching-cubes surfaces in world mm
(RAS), verified to lie on the mask surface. `itksnap_*_labels.txt`: ITK-SNAP
label descriptions in the site's colours. `morphometry.json`,
`morphometry_table.json`, `morphometry_branches.csv`. `airway_results.zip`.

## The viewer (web/app.js, function `viewer`)

NiiVue 0.44.0 from jsdelivr. ITK-SNAP arrangement: 2×2 grid (axial, coronal,
sagittal, 3D), one crosshair. Airway loads first, alone, in red. Overlay
radios: Airway / Lobar / Segmental / Branches (discrete label LUTs via
`NVImage.setColormapLabel`). Window presets, right-drag W/L, opacity
sliders, centreline toggle, radiological convention, orientation cube,
screenshot (`saveScene`). Readout bar shows voxel, mm, HU and the branch
under the cursor. Layout buttons: 2×2 / Axial / Coronal / Sagittal / 3D.
**The 3D button shows the airway surface alone by default** (the user was
emphatic); "Slice planes in 3D view" is an opt-in checkbox.

Morphometry card below: headline stats, central airways table (trachea, RMB,
LMB: length, inscribed diameter, cross-section diameter, angle), ratios
(eccentricity, taper, length/diameter, daughter/parent, (R−L)/(R+L)
asymmetry, interbronchial angle proxy, lung volume), per-generation,
per-lobe, all branches (click a row → crosshair jumps there).

## Hard-won gotchas (each cost real time)

1. **NiiVue picks the file reader from the `name` option**, for volumes and
   meshes. Names must keep their extension (`airway_mask_view.nii.gz`,
   `airway.mz3`). A bare name gives `undefined is not an object (evaluating
   'ext.toUpperCase')`.
2. **Label colours must be discrete.** `nv.addColormap` with a label-shaped
   map is interpolated and looks terrible. Use `img.setColormapLabel(cm)`.
3. **The slice-plane render shader ignores volume opacity.** To show only
   the mesh in 3D, switch to `setVolumeRenderIllumination(0)` with background
   opacity 0 and `opts.renderOverlayBlend = 0`. `-1` gives slice planes.
4. Wire UI controls before any `await`, and print progress and errors into
   the readout bar. A 74 MB display volume otherwise looks like a dead page.
5. Static URLs carry `?v=N` in the HTML. Bump N on every JS/CSS change or
   browsers keep the old file. Pages themselves are read from disk per
   request, so HTML/JS/CSS edits need no API restart. `api/*.py`,
   `worker/*.py`, `config.py` edits DO need a restart of `run_api.sh` /
   `run_worker.sh`. Stage scripts are subprocesses and reload on their own.
6. `class2anno.json` is nested (`lob`, `seg`, `subseg`), 0-based; label
   NIfTIs store class+1. Never hardcode label order; read `labels.json`.
7. `sitk.GetArrayViewFromImage(sitk.ReadImage(p))` on a temporary reads
   zeros. Keep the image in a variable or use `GetArrayFromImage`.
8. TfeNet window slices must be `.contiguous()` before the deformable convs.
9. MEDPSeg is a bash wrapper; kill by process group (`start_new_session`,
   `os.killpg`) or it holds the GPU after a sibling fails.
10. The mock (`run_worker.sh --mock`) copies a finished cohort case
    (`AIRWAY_MOCK_CASE`, default `airway-seg/age_00-01/sub-0012/sub-0012_acc-0295`)
    and must NOT overwrite `input.nii.gz`; the upload must be that case's
    own source file (`airway-dataset/age_00-01/sub-0012/sub-0012_acc-0295_source.nii.gz`).
11. Importing the cohort morphometry scripts by file path needs
    `sys.modules[name] = mod` before `exec_module`, or their dataclasses break.
12. TfeNet batching is NOT a speed lever (`in_step=1` processes one window
    regardless; `in_step>1` documented as wrong). The two real levers are
    `PARALLEL_MEMBERS` and `TFENET_CROP` in config.py.
13. Sandbox (`claude-safe.sh`) hides `/dev/nvidia*` and the Docker socket.
    GPU jobs and `docker build` run outside it.

## Morphometry sources (reused, not copied)

- `or_all3_run/scripts/09_airmorph_morphometry.py` → `measure_case(case_dir)`:
  40 columns per branch. Needs `case_{skel_parsing,parse}.nii.gz`,
  `case_airway_{graph,feature_cls,graph_cls}.npy`, `case_anno.json`,
  `union3_lcc.nii.gz`, all present in `jobs/<id>/case/`.
- `FINAL_AIRWAY_90_RESULTS_2026-09-16/reproducibility_scripts/axis_geometry.py`
  → `measure_case(case_id, case_dir, root, lmb, rmb)`: perpendicular cross
  sections, intrathoracic tracheal length, lung volume. Needs `lunglobe.nii.gz`.
- Root/LMB/RMB selection re-implemented in `stage_morphometry.py`
  (`central_topology`), mirroring `16_build_meeting_package.py`.
- Definitions match the 90-case report: asymmetry (R−L)/(R+L); the
  interbronchial angle is a proxy (sum of two local angles) and is labelled so.

## Where it stands

Working and browser-verified by the user on izmir: upload, progress,
retry, viewer, 3D airway-only, morphometry, downloads. Two real GPU runs on
2026-09-22 completed all eight stages.

**Cloud move: prepared, not executed.** `deploy/`:
- `Dockerfile.worker`: multi-stage, replicates `00_setup.sh` (cu121 torch for
  AirMorph/MEDPSeg, cu124 for TfeNet, DAConv/DSConv compiled for arch
  7.5–9.0 so it runs on T4/A100/A10G/L4/H100). No pycuda: the lab's
  AirMorph edits that make it optional are applied from
  `deploy/airmorph_cpu_path.patch`. Runtime image on `nvidia/cuda:12.4.1-base`. ≈ 20 GB.
- `Dockerfile.api`, `compose.yml` (api + caddy + worker), `Caddyfile` (HTTPS
  + basic auth), `airway-worker.service` (worker exits when idle → instance
  stops), `prepare.sh` (already run; `deploy/vendor/` holds the three model
  tarballs, `fuse_union3.py`, the two morphometry scripts, 496 MB).
- `README_AWS.md`: full runbook. Design: t3.small web instance always on,
  g5.xlarge GPU instance started by `api/cloud.py` per queued job and
  self-stopping after 15 idle minutes, EFS as the shared jobs directory.
  ≈ $35/month at 50 scans; $730/month if the GPU is left running.

**Images built and verified on izmir, 2026-09-23.** `airway-worker` (19.9 GB)
and `airway-api` exist in izmir's local Docker. A full run of sub-0053 in the
container produced results bit-identical to the native run (Dice 1.0000 on
every label volume, 239 branches, same morphometry). `docker stop` mid-job
was tested: stages killed, GPU freed, job marked failed with Retry.
Four fixes came out of it, all committed (see `README_AWS.md`, step 1):
`--shm-size 8g` is mandatory, the AirMorph CPU-path patch, torch-geometric +
openpyxl in the AirMorph env, and a 45-min stage timeout in the worker.

**Code is in git:** https://github.com/HUmutI/bch-airway, directory `webapp/`
(private, account HUmutI / huseyinumut2004@gmail.com; commits must use that
email). The whole airway project lives in that one repository; its top-level
README is the entry point. On izmir `airway_web/` is a symlink into the
checkout (`bch-airway/webapp/`); `jobs/`, `.venv/`, `deploy/vendor/` live
there and are ignored by git. The earlier `HUmutI/airway-web` repo is archived.

**Worker robustness added 2026-09-23:** a worker stopped mid-job (Ctrl-C,
SIGTERM, instance stop) now cancels its stages and marks the job failed; a
job found `running` at worker start is marked failed too. Before this, a
restart left the job "running" forever (that is what job `f081be85` was).

Decisions: personal AWS account, us-east-1, DuckDNS subdomain for now.
Next concrete steps, in order:
1. Owner: AWS account + IAM key, **G-instance vCPU quota request** (the long
   pole, 1–2 days), DuckDNS subdomain + token, lab password.
2. `aws configure` on izmir (`~/.local/bin/aws` is installed), push both
   images to ECR (`README_AWS.md` step 1).
3. `README_AWS.md` steps 2–4. The web instance and DNS can be set up before
   the GPU quota arrives.
4. Replace basic auth with institutional SSO when IT provides it.

## Ideas not yet built (agreed as worthwhile, none started)

1. Age field at upload → measurements as percentiles against the 90-scan
   normative cohort (`FINAL_AIRWAY_90_RESULTS_2026-09-16/cohort_90.csv` has
   age_years; `or_all3_run/normal_airway_branch_morphometry.csv` has per-
   generation rows).
2. QC flags: lobe with no branches, trachea not reaching the top slice, tree
   much shallower than the cohort median, member blob larger than the fused mask.
3. "Compare members" toggle: raw AirMorph / MEDPSeg / TfeNet masks.
4. Job history page (`GET /api/jobs` already exists).
5. Morphometry speed: `measure_case` does one `np.argwhere` per branch over
   the full volume; a single pass grouped by label would cut 180 s to ~20 s.

## How to run locally (izmir)

```
cd /fileserver/external/quin/ege/airway_web
./run_api.sh                 # http://izmir:8000
./run_worker.sh              # second terminal, outside the sandbox, needs the GPU
./run_worker.sh --mock       # no GPU; upload the sub-0012_acc-0295 source file
```
Useful: `GET /api/jobs` lists jobs; `jobs/<id>/logs/<stage>.log` per stage;
`jobs/<id>/log.txt` the job log; `window.airwayViewer` in the browser console
is the NiiVue instance.
