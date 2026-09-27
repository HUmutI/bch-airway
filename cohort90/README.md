# cohort90/

The 90-scan balanced normative cohort (10 per age group, one scan per
patient, sex recorded), measured with the pipeline's morphometry and written
up as a report for the radiologists. Code only; the cohort tables, the
report and the figures stay on izmir in
`cohort90_results/`.

| script | what |
|---|---|
| `00_select_cohort.py` | integer-programming selection of the 90 from the normals of two source cohorts |
| `01_measure_cohort.py` | runs `pipeline/scripts/09_airmorph_morphometry.py` and `axis_geometry.py` over the 90 |
| `axis_geometry.py` | perpendicular cross-sections of trachea / main bronchi, intrathoracic tracheal length, lung volume; `test_axis_geometry.py` covers it |
| `topology_qc.py` | tree sanity flags per case |
| `02_build_package.py`, `03_build_report.py`, `retypeset_report.py` | the tables, figures and LaTeX report |
| `fetch_patient_sex.py` | joins the anonymisation mapping with the clinical atlas **in memory** and writes only anonymised id + sex. Runs only on a host that can read both; never commit its inputs or the mapping. |
| `finish_remaining_four_after_gpu.py` | one-off: finished four cases after a GPU outage |

Definitions used in the report (and by the web app's morphometry card):
asymmetry is (R−L)/(R+L); the interbronchial angle is a proxy (sum of two
local angles) and is labelled as such; branch generation is counted from the
trachea as 0.
