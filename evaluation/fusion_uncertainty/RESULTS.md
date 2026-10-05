# Results: confidence map and uncertainty-aware fusion (2026-10-04)

Everything below is computed from the stored member masks of the paper's
benchmarks with the paper's metric code. Validation = 30 ATM'22 volumes (used
for every choice: the calibration table, the logistic scorer, the operating
point). Test = 30 ATM'22 volumes, pediatric = 9 expert-annotated scans; both
touched once, with the validation choices frozen.

## 1. The confidence map works, and it is calibrated

P(airway | vote pattern) estimated on validation, applied unchanged:

| split | AUROC for false-positive voxels | expected calibration error | FP voxels inside the least-confident 10 % / 20 % of the mask |
|---|---|---|---|
| validation (fit) | 0.898 (per case 0.908 ± 0.048) | 0.0 | 59 % / 86 % |
| **test** | **0.901** (per case 0.917 ± 0.041) | **0.9 points** | 61 % / 86 % |
| pediatric | 0.917 | 11.2 points (see below) | 78 % / 95 % |

The raw vote fraction (7 predictors, equal weight) gets AUROC 0.876 on test;
one-vote-per-model 0.894; the calibrated table 0.901. On the test split the
predicted and observed precisions agree to within a point averaged over the
mask (`figures/fig_reliability.png`). 79 % of the fused mask is marked by all
seven predictors and is 98 % correct; the remaining 21 % holds 86 % of all
false positives.

What the table says about the members (validation, inside the fused mask):

| pattern | share of mask | P(airway) |
|---|---|---|
| all seven agree | 78.3 % | 97.9 % |
| MEDPSeg + TfeNet, AirMorph silent | 2.4 % | 98.3 % |
| AirMorph 5/5 + MEDPSeg | 3.7 % | 70.4 % |
| AirMorph 5/5 + TfeNet | 1.5 % | 68.8 % |
| AirMorph 5/5 alone | 1.2 % | 52.6 % |
| MEDPSeg alone | 1.2 % | 53.9 % |
| AirMorph 4/5 alone | 1.9 % | 34.3 % |
| AirMorph 3/5 alone | 3.1 % | 17.3 % |

AirMorph's internal agreement is weak evidence on its own; agreement *between*
models is what matters, and MEDPSeg + TfeNet without AirMorph is almost
always real distal airway. These are the voxels the union was kept for.

Pediatric: the expert annotations are generous (the union's precision against
them is 99.85 %), so every pattern is observed near 100 % and the adult
probabilities are too pessimistic there. The *ranking* still holds (AUROC
0.92; 78 % of the few false positives sit in the least-confident 10 %).

## 2. Branch confidence

Mean confidence along a branch's centreline, against whether the branch is
real (>= 50 % of its centreline within one voxel of the GT):

| band (test, 10 305 branches) | branches | false |
|---|---|---|
| all predictors agree | 5 225 | 2.0 % |
| 0.9 – 1 | 1 289 | 10.3 % |
| 2/3 – 0.9 | 2 066 | 15.5 % |
| 1/2 – 2/3 | 987 | 29.8 % |
| below 1/2 | 738 | 32.7 % |

By number of supporting models (test): one model 35.7 % false, two 28.0 %,
three 5.6 %. The 12-feature logistic scorer reaches AUROC 0.90 per case
(0.82 pooled) for false branches; the plain centreline confidence 0.89 (0.79).
Pediatric: 0 false branches among 758 with full agreement, 7 % among the 43
below 1/3.

## 3. Where the union's error is, and what pruning can do

Pruning every false branch *with the ground truth itself* (oracle, leaf by
leaf) raises validation DSC by 1.0 point and costs 3.3 points of branch
detection. So false branches are not where the union's 12 % false-positive
volume is. It is in the boundary shell of true branches: the union keeps the
fattest of three boundaries. The logistic leaf pruning at tau 0.5 removes
~6 branches per volume for +0.02 DSC / −0.5 BND on validation; it is kept as
an option (`--prune-tau`) and switched off in the default.

## 4. Branch-relative consensus: the operating point

Rule: keep a voxel when its calibrated confidence is at least kappa times the
confidence along its branch's centreline; always keep the core within
max(0.5 r, 1.5 voxels) of the centreline; seeded component again. Chosen on
validation by "maximum DSC with at most one point of branch-detection loss":
**kappa 0.4, core 1.5 voxels, no pruning**. Full grids:
`results/refine_sweep_{val,test}_summary.csv`; the frontier against the 51
earlier voxel rules: `figures/fig_frontier_{val,test}.png`.

Whole tree, mean ± SD over volumes (`results/whole_tree_summary.csv`):

| split | method | DSC | clDice | TLD | BND | Sens | Prec |
|---|---|---|---|---|---|---|---|
| val | MEDPSeg alone | 92.86 | 85.33 | 88.02 | 76.68 | 94.86 | 91.11 |
| val | majority vote, seeded | 94.23 | 89.94 | 88.08 | 80.19 | 95.07 | 93.62 |
| val | union, seeded (current) | 92.45 | 89.19 | 95.80 | 93.23 | 98.34 | 87.44 |
| val | **proposed** | **93.57** | 89.31 | 95.27 | 92.30 | 97.15 | 90.44 |
| test | MEDPSeg alone | 93.29 | 85.82 | 87.50 | 76.10 | 94.78 | 92.02 |
| test | majority vote, seeded | 94.38 | 90.48 | 87.84 | 80.35 | 95.00 | 94.04 |
| test | union, seeded (current) | 93.06 | 90.24 | 95.22 | 92.31 | 98.23 | 88.71 |
| test | **proposed** | **94.04** | 90.18 | 94.60 | 91.33 | 97.09 | 91.46 |
| pediatric | MEDPSeg alone | 80.40 | 74.51 | 76.60 | 53.17 | 83.20 | 80.83 |
| pediatric | majority vote, seeded | 90.93 | 91.49 | 80.17 | 59.26 | 83.48 | 99.95 |
| pediatric | union, seeded (current) | **94.74** | 97.06 | 93.91 | 86.40 | 90.21 | 99.85 |
| pediatric | proposed | 92.67 | 96.87 | 92.01 | 80.16 | 86.54 | 99.86 |

Paired per volume, proposed minus union: test DSC +0.98 (better in 26 of
30), precision +2.74, sensitivity −1.14, BND −0.98; validation DSC +1.13
(28 of 30). On the test split this is a point no earlier rule reaches: at
DSC 94.0 the best voxel rule has BND 86 (wvote 1.0), the proposed 91.3.

Per generation on test (`results/per_generation_test_summary.csv`): the
proposed fusion has the best DSC of the four at generations 1–4 and sits
between the union and the majority vote beyond; branch detection stays
within 1–1.6 points of the union at every generation (gen 7: 90.2 vs 91.0;
gen 10: 84.2 vs 84.9) while the majority vote is 13–16 points lower.

**Pediatric is the caveat.** The experts' masks are wider than any of the
members, so the union has nothing to trim there (precision 99.85 %) and the
refinement only removes true wall voxels: DSC −2.1, BND −6.2 (worse in 9 of
9). The web app therefore keeps the union as its default and exposes the
refinement as a mode (`AIRWAY_FUSION=topofuse`); the confidence map itself is
always produced. For the paper the honest statement is: the refinement is an
adult-benchmark gain that does not transfer to the pediatric annotation
convention, and the confidence map is what transfers.

## 5. Figures

* `figures/fig_example_ATM_043.png`, `fig_example_sub-0012.png`: mask coloured
  by confidence beside the ground-truth error map and an axial slice.
* `figures/fig_reliability.png`: predicted vs observed precision per pattern.
* `figures/fig_frontier_val.png`, `fig_frontier_test.png`: DSC against branch
  detection / tree length for the 51 earlier rules and the proposed curve.

## 6. Reproduce

```
PY=or_all3_run/envs/airmorph/bin/python   # kimimaro, scikit-image, pandas
$PY rerun_airmorph_members.py --case-dir airway-seg/.../sub-X --out fusion_uncertainty_run/pediatric_members   # GPU, pediatric only
$PY extract.py --split val; $PY extract.py --split test; $PY extract.py --split pediatric
$PY cache_volumes.py --split val  (test, pediatric)
$PY sweep.py --split val --out results/sweep_val.csv --logit-json results/logit_val.json
$PY sweep_refine.py --split val --out results/refine_val.csv   (test)
$PY apply_final.py --split test --scheme calibrated --prune-tau 0 --kappa 0.4 --core-min-vox 1.5 --out results/whole_test.csv
$PY uncertainty_eval.py --splits val test pediatric --logit-json results/logit_val.json --out-dir results/unc
AIRWAY_ROOT=... $PY ../benchmark_adult/per_generation.py --optional vote2of3_seeded --extra topofuse=<masks dir>
```
