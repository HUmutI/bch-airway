# AirMorph morphometry for the 83 normal cases

`scripts/09_airmorph_morphometry.py` anatomically labels the fused `or_all3_lcc`
masks and writes one spreadsheet row per physical parsed branch.

The installed copy includes CPU fallbacks for AirMorph's original PyCUDA tree-parsing
helpers, so this workflow does not require PyCUDA or a visible NVIDIA GPU. The small
anatomical labeling network also runs on CPU.

```bash
cd /fileserver/external/quin/ege/or_all3_run
envs/airmorph/bin/pip install openpyxl

# Pilot first
envs/airmorph/bin/python scripts/09_airmorph_morphometry.py \
  --case sub-0012_acc-0295 --phase all

# All 83 normal cases; completed classifier products are skipped on resume
envs/airmorph/bin/python scripts/09_airmorph_morphometry.py --phase all
```

Outputs:

- `normal_airway_branch_morphometry.xlsx`: Branches, Cases, and Data Dictionary sheets
- `normal_airway_branch_morphometry.csv`: portable copy of the branch table
- `normal_airway_morphometry_status.csv`: resumable per-case status

The input masks remain unchanged. Classification products and
`union3_lcc_skeleton.nii.gz` are saved beside each case in `airway-seg/`.

## Measurement definitions

Lengths, radii, diameters, angles, and volumes are recalculated in physical units from
the fused mask. AirMorph's saved neural-network feature matrix is deliberately not used
for physical measurements because several of its columns are normalized before saving.
Anatomical names are model predictions and must undergo visual/anatomical QC before the
spreadsheet is used as a pediatric reference dataset.
