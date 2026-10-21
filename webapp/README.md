# Airway segmentation web app

Upload a chest CT, get the fused airway mask and the branch labels back, with
an in-browser 3D view. It runs the paper's pipeline exactly: AirMorph, MEDPSeg
and TfeNet, union, seeded component selection, skeletonisation, graph
transformer naming. Nothing is approximated.

```
airway_web/
  api/app.py            FastAPI: upload, status, results
  worker/worker.py      takes queued jobs one at a time and runs them
  worker/pipeline.py    the stage sequence and the progress file
  worker/stages/        one script per stage, each in the environment it needs
  web/                  the two pages, no framework; palette.json is the one colour table
  jobs/                 one directory per upload (created at runtime)
  config.py             paths, and the two speed switches
```

## What the page shows

The viewer is laid out like ITK-SNAP: axial, coronal and sagittal slices plus a
3D pane, one crosshair shared by all four. The 3D pane shows the three slice
planes with the airway surface cutting through them, not a volume render. The
airway comes up first, alone, in red. Lobar, segmental and branch labels are
overlays you switch to; in lobar mode each lobe has its own surface and its own
checkbox. Window presets (lung, soft tissue, bone) plus right-drag for free
window and level. The readout under the canvas gives voxel, mm, HU and the name
of the branch under the cursor. Screenshot button saves the four panes as PNG.

Below it, **Airway morphometry**: branch count, total centreline length,
deepest generation, lumen volume, tracheal length and diameter, airway/lung
volume, skeleton fractal dimension; tables for the central airways (trachea and
both main bronchi with inscribed and cross-sectional diameters, eccentricity,
taper, carina ratios), per generation, per lobe, and every branch (click a row
and the crosshair jumps there). It is computed by the cohort's own code
(`or_all3_run/scripts/09_airmorph_morphometry.py` and the 90-case
`axis_geometry.py`), imported, not copied, so the numbers match the tables in
the reports.

Stage order: prepare, airmorph, medpseg | tfenet, fuse, label, package,
morphometry. The viewer opens as soon as `package` is done
(`status.results_ready`); morphometry runs afterwards (15 s for a sparse tree,
about 3 min for 240 branches) and cannot fail the job.

`results/` per job: full-resolution NIfTIs as uploaded (`image`, `airway_mask`,
`airway_lobar`, `airway_segmental`, `airway_branches`, `airway_centreline`),
the same as `*_view.nii.gz` resampled to an isotropic grid of at most
`VIEW_MAX_VOXELS` for the browser (the CT costs 4 bytes a voxel on the GPU),
`mesh_airway.mz3` and `mesh_lobe_<k>.mz3` surfaces in world mm,
`itksnap_*_labels.txt` label descriptions in the site's colours,
`labels.json`, `summary.json`, `morphometry.json`, `morphometry_table.json`,
`morphometry_branches.csv`, and `airway_results.zip` with everything but the
display copies.

## Running it on izmir

Two processes. The API can run anywhere, the worker needs the GPU and must run
**outside the sandbox** (it hides `/dev/nvidia*`).

```
cd /fileserver/external/quin/ege/airway_web
./run_api.sh            # http://izmir:8000
./run_worker.sh         # in a second terminal, or tmux
```

To try the pages on a machine with no GPU, `./run_worker.sh --mock` fakes the
stages and packages a finished case from the cohort so the viewer shows real
output.

## Speed

The cohort recipe runs the three models one after another and TfeNet slides its
window over the whole volume. Two changes, both in `config.py`:

- `PARALLEL_MEMBERS` runs MEDPSeg and TfeNet at the same time on the GPU. They
  fit together on a 24 GB card.
- `TFENET_CROP` restricts TfeNet's sliding window to the box holding the lungs
  and the airway AirMorph already found, plus a margin. The airway cannot lie
  outside that box, and the box is typically a third of the volume. Window
  origins inside the box differ from a full-volume run, so voxels near window
  edges can differ slightly. Set it to 0 for the exact cohort behaviour.

Expected wall time on the A5000 drops from about 11 minutes to about 6.

## Running it on your own machine

On any Ubuntu machine with an NVIDIA GPU (16 GB or more), the hosted site's two
containers run with one command, results stay local:
`docker compose -f deploy/compose.local.yml up -d`, then http://localhost:8000.
Setup (Docker, NVIDIA toolkit, access to the worker image) is in
`deploy/README_LOCAL.md`.

## Moving to AWS

See `deploy/README_AWS.md` for the full runbook. Short version:

The API and the worker only share the jobs directory and the status file.
Point `AIRWAY_JOBS` at an S3 mount, run the worker on a GPU instance, and the
same code works. A proper deployment replaces the directory poll in
`worker.py` with an SQS consumer and lets the instance scale to zero between
jobs; nothing in `pipeline.py` changes.

## Not for clinical use

The site says so on every page and requires an acknowledgement before upload.
Keep it that way.
