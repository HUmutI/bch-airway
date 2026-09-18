# bch-airway

Code for the paper *Airway Segmentation and Anatomical Branch Assignment in
Pediatric Chest CT*.

The pipeline segments the airway with an ensemble of three public models,
retains one connected component chosen by a seed rather than by size, and names
every branch with a graph transformer. This repository holds the evaluation
code and the result tables. It does not hold patient data.

## What is here

| directory | contents |
|---|---|
| `benchmark_adult/` | scoring on the ATM'22 test split, whole tree and per generation, and the tables the paper reports |
| `rule_sweep/` | the 51 combination rules compared on the validation split, and the summariser that ranks them |
| `benchmark_pediatric/` | the same comparison against expert-annotated pediatric volumes |
| `figures/` | the render scripts for the figures in the paper |

`benchmark_adult/shim/` holds the metric implementations as imported, so that
tree length, branch detection and clDice mean the same thing in every table
here. Metric definitions differ between publications, and the paper states the
ones used.

## Data

None of it is included.

The adult benchmark is the Airway Tree Modeling challenge (ATM'22), which is
public and must be obtained from the organisers. The pediatric examinations are
from a clinical cohort and cannot be shared. The pediatric result table is
included with case identifiers replaced by positional labels.

## Running it

Set `AIRWAY_ROOT` to the directory holding the masks and references, then

```
python benchmark_adult/recompute.py --workers 16 --out results.csv
python rule_sweep/sweep.py --split val --workers 12 --out val.csv
python rule_sweep/summarise.py val.csv
python benchmark_pediatric/bench_pediatric.py --gt-dir <annotations> --out ped.csv
```

The scripts are CPU only. Skeletonisation inside clDice dominates the runtime,
at roughly half a core-minute per case per rule.

## Two things worth knowing before reusing this

Component selection is seeded, not largest. A model that emits a spurious
structure larger than the airway will otherwise have that structure kept and the
airway discarded, and the result still looks like an airway tree. On the adult
benchmark this affects 3 of 30 examinations. In children it affects 6 of 83, and
every one is an infant under one year.

Tree length and branch detection are defined on the reference centreline, so the
skeletonisation used to build that centreline changes the numbers. The pediatric
driver uses the same TEASAR skeleton as the adult one for exactly that reason.

## Citation

To be added on publication.
