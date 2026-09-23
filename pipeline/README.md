# pipeline/

The cohort pipeline: three model environments, seven segmentation stages,
then QC, morphometry and the table/report builders.

- **`README_RUN.md`**: read this first. Layout, stage table, the pilot run,
  the LAS/LPS orientation trap, the TfeNet batching result, runtime and
  storage, decisions baked in, known data quirks, what to do when a stage
  fails.
- **`MORPHOMETRY_README.md`**: `09_airmorph_morphometry.py`, one row per
  branch, and its outputs.
- **`scripts/`**: `00_setup.sh` builds the environments from the model
  tarballs (`models/README.md`); `01`–`07` are the segmentation stages;
  `run_pipeline.sh {pilot|all} [start_stage]` runs them; `08`–`19` are
  integrity checks, morphometry, radiologist QC deliveries, analysis tables,
  age distributions, the paper's Table 1, the meeting package and report,
  and the correction of one case with a tracheal tube.
- **`fuse_union3.py`**: the fusion rule (union of the three masks, one
  6-connected component seeded from AirMorph). Used unchanged by the cohort
  pipeline and by the web app.

All paths default to izmir's and are overridable through `OR3_*` variables
(`scripts/config.sh`, `docs/izmir.md`). Case lists (`cases_*.txt`),
`manifest.csv`, `qc.csv` and every output stay on izmir; they are derived
from the dataset and are not in git.
