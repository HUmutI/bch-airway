# Uncertainty-aware fusion: confidence map, branch confidence, topology-aware pruning

Study code for making the three-model ensemble say *how sure it is*, and for
using that to tidy the fused mask without losing the distal branches the union
is kept for. Numbers in `RESULTS.md`; everything here runs from stored member
masks, no GPU.

## What the ensemble can tell us without retraining

Every fused voxel is marked by some subset of seven predictors: AirMorph's five
member networks (its mask is their >= 3 vote), MEDPSeg and TfeNet. That gives
24 possible *vote patterns*. On the 30 ATM'22 validation volumes we count, per
pattern, how often the voxel is airway in the ground truth. The table
(`confidence_lut.json`) is the confidence map: **P(airway | pattern)**.

Two things the table shows that nothing in the pipeline knew before:

* AirMorph's internal agreement is weak evidence on its own: 5/5 members
  without MEDPSeg or TfeNet is still 47 % wrong; 3/5 alone is 83 % wrong.
* MEDPSeg and TfeNet agreeing without AirMorph is 98 % right (2.4 % of the
  mask): the union is recovering real distal airway that AirMorph misses.

A per-branch confidence is the mean of the map along the branch's centreline.

## Topology-aware fusion (`topofuse`)

1. **Branch tree of the union.** Skeletonise the fused mask, cut at branch
   points (`skeleton_parsing`, as the benchmark does), assign every voxel to
   its nearest branch, join branches into a tree rooted at the widest one.
2. **Branch probability.** A 12-feature logistic regression (confidence along
   the centreline and over the voxels, how many models support it, member
   fractions, length, radius, radius ratio to the parent, tubularity,
   generation, leaf flag) fitted on the validation branches
   (`branch_logit.json`), target = branch centreline within one voxel of GT.
3. **Leaf pruning.** Repeatedly cut leaf branches with probability < tau
   (generations 1-3 never cut). Topology-preserving: a false subtree vanishes
   from its tips inward; the tree never disconnects.
4. **Branch-relative consensus.** Keep a voxel when its confidence is at least
   kappa times its branch's centreline confidence; always keep the core within
   max(0.5 r, 1.5 voxels) of the centreline. A branch all models see is trimmed
   towards majority consensus; a distal branch one model found keeps all its
   voxels. Then the seeded-component rule once more.

## Files

| file | role |
|---|---|
| `common.py` | paths, loaders (ATM val/test, pediatric), metrics (verbatim), confidence schemes, cache loader |
| `rerun_airmorph_members.py` | GPU: re-run AirMorph keeping the five member votes (pediatric cases) |
| `extract.py` | per case: union+seeded, branch tree, per-branch evidence & GT overlap, confidence histograms |
| `cache_volumes.py` | per case: vote pattern volume, distance to skeleton, GT (so sweeps need no member volumes) |
| `tree.py`, `refine.py` | the branch tree and the branch-relative refinement |
| `sweep.py` | leaf-pruning sweep from counts (exact, no volumes); fits the logistic scorer |
| `sweep_refine.py` | refinement sweep (scheme x tau x kappa x core) on cached volumes |
| `apply_final.py` | chosen rule with the full metric set beside the references; writes masks + branch tables |
| `uncertainty_eval.py` | voxel reliability / AUROC / FP capture; branch false-rate by confidence band |
| `fig_frontier.py`, `fig_example.py` | figures |
| `confidence_lut.json` | the calibration table (copied to `webapp/worker/`) |

Data and caches live on izmir under `fusion_uncertainty_run/` (not in git).
Per-generation scoring reuses `benchmark_adult/per_generation.py --extra`.
