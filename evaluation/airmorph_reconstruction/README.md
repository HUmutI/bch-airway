# AirMorph training code — segmentation + anatomical labeling

Reproduces **both** stages of the AirMorph pipeline from the public ATM26 challenge data:
the 5-model binary airway segmentation ensemble, and the anatomical-labeling transformer.
Written because the released repo is inference-only: no train loop, optimiser, loss,
dataset or augmentation was ever committed (`git log --diff-filter=A` confirms), and the
published checkpoints are behind an inaccessible Google Drive link.

**Nothing in the original repo is modified.** All code here is new, under `training/`, and
reuses the provided networks / graph-build / metrics unchanged. Trained checkpoints are
written into `checkpoints/` in the exact formats the released inference code loads, so
`airwayatlas_pipeline.py` runs unchanged afterwards.

## Scope

In scope (both trainable from the ATM26 data the user has):
- **Segmentation:** the 5-model binary ensemble (299 ATM26 Task-1 cases).
- **Anatomical labeling:** the transformer `our_net` (118 ATM26 Task-2 cases, whose
  branch-level ground truth ships in `...-NewTrainLabels.zip`). This became trainable only
  because those annotations are now available — earlier they were not.

Out of scope: the AirwaySign radiomics / branching-pattern signature stages, and any
external test cohorts (LIDC etc.) the user does not have.

## Data (ATM26, `.mha`)

| Source | Contents | Used for |
|---|---|---|
| Task-1 TrainBatch | 299 CT + 299 binary airway masks | segmentation training |
| Task-2 NewTrainLabels | 118 branch anatomical-label volumes | labeling targets |
| Task-2 TrainBatch | 118 CT (⊂ Task-1) + old labels | **skipped** (redundant / superseded) |

Images carry a `_0000` suffix, labels do not; `discover_cases` pairs them by id. The 118
labeling cases are a subset of the 299; their CT and binary masks come from Task-1.

## The model ↔ loss mapping

Derived from the source, not guessed. The repo expects exactly 5 airway checkpoints and
the paper describes exactly 5 models, each with a distinct topology-preserving loss.

| # | Checkpoint | Network | `layer_order` | out ch | Ckpt format | Loss |
|---|---|---|---|---|---|---|
| 1 | `airway_model1.pth` | `UNet3D` | `cip` | 2 | raw `state_dict` | Dice + Focal |
| 2 | `airway_model2.pth` | `UNet3D` | `cpi` | 2 | raw `state_dict` | Connectivity-Aware (CAL) |
| 3 | `airway_model3.pth` | `UNet3D` | `cip` | 2 | raw `state_dict` | CAL + local-sensitive distance |
| 4 | `wingsnet.ckpt` | `WingsNet_v3` | — | 1 | `{"state_dict": …}` | General Union |
| 5 | `break1.ckpt` | `WingsNet_v3` | — | 1 | `{"state_dict": …}` | Breakage-Sensitive |

`networks/airway_network_break.py` is byte-identical to
`networks/airway_network_wingsnet.py`, confirming models 4 and 5 differ only by loss —
"break" names the *breakage-sensitive loss*, not a repair architecture.

The `layer_order` assignment is forced by `models/airway_model.py`: `weight_path1` →
`self.net` (`cip`), `weight_path2` → `self.net_for_model2` (`cpi`), `weight_path3` →
`self.net` (`cip`). Getting model 2 wrong fails loudly at `PReLU.weight`.

Ensemble (`segmentator/airway_segmentator.py:65-68`): sum the 5 binary maps, keep voxels
with **≥ 3 votes**, then `KeepLargestConnectedComponent`.

## The anatomical labeling stage

The released transformer `networks/transformer.py::our_net` is reused **unmodified** — its
`forward(x, mask_top, spd, p)` already returns all six classification-head logits
(`x1_1,x2_1,x3_1` stage-1 "give" and `x1_2,x2_2,x3_2` stage-2 "accecpt", for lobar-6 /
segmental-20 / subseg-82) plus `node_pair*` (learned topology prior) and `outlier*`. So no
subclass is needed (unlike WingsNet). Inference reads the stage-2 heads.

Training data flow (`training/data/prepare_labeling.py`), identical to the deployed
`classifier/airway_classifier.py` graph build so features match at train and inference:

```
GT binary airway (Task-1)  -> MPC-Skel skeleton  -> tree_parse (branches, parent/child,
generation, trachea)  -> feature_extraction_cuda (11 features, edges)  -> multitask_dataset
(x[N,11], spd[N,N] Floyd, mask_top[N,N] ancestor->descendant)          [pycuda + torch_geometric]
```

**The ATM26 Task-2 public labels are SEGMENTAL, not subsegmental** (verified on the data,
not assumed): each airway voxel carries `v ∈ 0..20`, and `v` maps to `class2anno["seg"]`
id `v−1` (`v=0` background/unlabeled; ids 1-8 = left segments, 9-18 = right, 19 = trachea,
20 = LB7). Confirmed by per-label centroids (left/right split, trachea topmost). So the
**82-class subsegmental ground truth was never released** — only lobar and segmental are
learnable from real labels.

Per-node ground truth (`training/data/label_transfer.py`): each node takes the **majority
segmental value** over its parsed branch's voxels (`airway_parse == node_idx[j]+1`, the
inverse of `transform_pred_data`) → **segmental** target `v−1` (real), **lobar** target by
anatomy (`SEG_TO_LOB`, real). The **subsegmental** head has no real target; `subseg_policy`
chooses between `"base_segment"` (supervise toward each segment's base subseg class —
segment-granularity, keeps the head trained) and `"ignore"` (leave it unsupervised).
Verified via `python -m training.data.label_transfer`.

Loss (`training/losses/labeling_loss.py`): the paper gives training details but no loss, so
cross-entropy deep-supervision over the six heads,
`Σ_level w·[CE(stage2)+aux·CE(stage1)]`, `aux=0.5`, `ignore_index` for unlabeled nodes /
the unsupervised subseg level. `node_pair`/`outlier` get no explicit target — they sit in
the forward path and train implicitly.

## Runbook

```bash
# 0. dependencies (additive install into `clip`) + lungmask weight.
#    Now also installs pycuda + torch_geometric for the labeling stage (non-fatal if nvcc
#    is missing — segmentation still works).
bash training/setup_env.sh

# 1. extract the data and point training/configs/train_config_base.py at it:
#      seg_images / seg_labels  -> Task-1 imagesTr / labelsTr   (299)
#      label_branch_dir         -> Task-2 NewTrainLabels/labelsTr (118)
#    (Skip the Task-2 TrainBatch zip: images are redundant, labels superseded.)

# --- Segmentation ---------------------------------------------------------------
python -m training.data.prepare_dataset --limit 2 --skip-splits   # smoke test
python -m training.data.prepare_dataset                           # full (hours; disk-heavy)
python -m training.train --model model1 --overfit 4 --max-epochs 20   # sanity
CUDA_VISIBLE_DEVICES=0 python -m training.train --model model1   &
CUDA_VISIBLE_DEVICES=1 python -m training.train --model model2   &
CUDA_VISIBLE_DEVICES=2 python -m training.train --model model3   &
CUDA_VISIBLE_DEVICES=3 python -m training.train --model wingsnet &
wait
CUDA_VISIBLE_DEVICES=0 python -m training.train --model break
python -m training.evaluate --split test                          # per-model + ensemble

# --- Anatomical labeling --------------------------------------------------------
python -m training.data.prepare_labeling --inspect ATM_001        # validate label codes
python -m training.data.prepare_labeling --limit 2                # smoke test
python -m training.data.prepare_labeling                          # full 118-case graph cache
python -m training.train_labeling --overfit 1 --max-epochs 300    # sanity (memorise 1 graph)
python -m training.train_labeling                                 # full (Adam 5e-4, 600 epochs)
python -m training.evaluate_labeling --split test                 # node acc + TreeCons

# --- Both (decoupled orchestration) --------------------------------------------
python -m training.train_pipeline --parallel-seg --gpus 0 1 2 3   # seg ensemble, then labeling

# The released pipeline now runs unmodified with all six checkpoints in place:
python airwayatlas_pipeline.py
```

The three trainings the user asked for map to: `training.train` (×5 models) = segmentation;
`training.train_labeling` = labeling; `training.train_pipeline` = both.

## Why one model per GPU instead of DDP

Batch size 1 is an explicitly stated paper hyperparameter. DDP across 4 GPUs would make
the effective batch 4 and change the gradient-noise scale that lr = 2e-3 was tuned
against — a deviation on a *specified* value, which the fidelity priority rules out.
Five independent models make the embarrassingly-parallel axis free anyway. It also makes
`droplayer`'s bare `.cuda()` call (`networks/airway_network_wingsnet.py:101`) correct by
construction, since each process sees exactly one device.

## Verified before shipping

- All 5 networks build, forward, and backward; **zero parameters without gradient**.
- Every exported checkpoint loads into the **provided** `UNet3D` / `WingsNet_v3` classes
  with `strict=True` — the format contract holds.
- All 5 losses: perfect prediction scores below both an inverted prediction and an
  all-foreground prediction.
- Initial gradient norms 0.9–7.0, all under `grad_clip = 12`.
- Every module compiles; all non-`monai` modules import.

Labeling stage:
- `our_net` (3.89 M params) builds, forwards to the six heads (shapes 6/20/82), and
  backprops; a synthetic graph overfits (loss 14.5 → 3.5 in 8 steps).
- `label_transfer` self-check passes for all 86 raw codes: trachea → (lob0, seg18, sub0),
  LB1+2a → (lob1, seg0, sub2), etc.
- Four params (`give/accecpt.dense_linear.3`) get no gradient — they are **dead in the
  released architecture** (`Stage.forward` loops `range(3)` and never uses index 3). Left
  untouched to keep the `state_dict` loadable by the provided `AirwaylabelingModel`.
- **Validated on real ATM26 data** (ATM_001): full lungmask → kimimaro → pycuda
  `tree_parse`/`feature_cuda` → graph → GT-transfer chain runs; 211 nodes, 208 resolved,
  trachea node → seg 18 (correct). Branch-label values are `0..20` = segmental scheme.
- **nvcc host-compiler fix**: CUDA 12.3 rejects gcc > 12; this box has gcc 13 and no g++-12,
  so `prepare_labeling.py` sets `NVCC_PREPEND_FLAGS=-allow-unsupported-compiler` (host stub
  only; device code unaffected). The provided inference pipeline needs the same export.

---

# Deviation register

Every departure from the paper, with its justification.

## A. Corrections — the paper as printed does not run

| # | Deviation | Why it is necessary |
|---|---|---|
| A1 | **Dice+Focal implemented as `(1-Dice) + Focal`** rather than the printed `-(Dice - Focal)` | The literal form expands to `-Dice + (1/\|X\|)Σ(1-p)²log p`. Since `log p ≤ 0` the second term is minimised by driving `p → 0` on the foreground: the loss is unbounded below and diverges. |
| A2 | **CAL's `Σ αx CE` normalised by `Σ αx`** (weighted mean) | Unnormalised, a sum over ~8.7 M voxels is ~10⁶× the overlap term, making that term inert. `ce_reduction="sum"` restores the literal form. |
| A3 | **LSD's `‖·‖₂` normalised by `√numel`** | The raw L2 over 8.7 M voxels is O(10³) and would swamp `L_CAL` entirely. |
| A4 | **Model 5 = Dice + Breakage-Sensitive, not BS alone** | `L_BS = 1 - Σ p·c / Σ c` is soft recall on the centreline with no false-positive term: `p ≡ 1` everywhere gives `L_BS ≈ 0`. Trained alone it collapses to all-foreground. In its source it is an auxiliary term. |
| A5 | **GUL clamps `p ≥ 1e-6` before `pow(0.7)`** | `d/dp p^0.7 = 0.7 p^-0.3 → ∞` at `p = 0`; NaN on the first iteration otherwise. |
| A6 | **All loss reductions forced to fp32 under bf16 autocast** | Summing 8.7 M values in reduced precision discards exactly the faint peripheral-branch contributions these losses exist to preserve. bf16 (not fp16) is used so `p^0.7` and the large sums do not overflow. |
| A7 | **`clDice` reimplemented in `training/eval_patch.py`** | `util/airwayatlas_eval.py:141` reads `skeletonize(gt_mask, method="lee\`")` — a stray backtick that makes **every** 3D `clDice` call raise. Original file untouched; see "Proposed upstream patches". |
| A8 | **Tree-parsing helpers copied into `training/eval_patch.py`** | `data_process/tree_parse_skelbased.py:5` imports `pycuda.driver` at module scope, so the three pure-scipy helpers `branch_detected_calculation` needs cannot be imported without pycuda. Copied verbatim. |
| A9 | **`WingsNetTrain` subclass restores the `dc0_0` aggregation head** | `WingsNet_v3.forward` constructs `dc0_0` and `dropout1` but never calls them, discarding the 12 encoder side outputs. The paper's Fig. 6b3 draws WingsNet with **two** aggregation heads, so the authors must have supervised both — otherwise those tensors stay at their random init. `state_dict` remains key-for-key compatible. |
| A10 | **U-Net losses defined on `sigmoid(logits[:,1])`, not softmax** | Forced by `finalsigmoid=1` (→ `nn.Sigmoid`, `networks/airway_network.py:211-214`) plus `AsDiscrete(pred[:,1,...])` in `models/airway_model.py:62`. A softmax would silently produce checkpoints that are wrong at inference. |

## B. Interpretations where the paper is ambiguous

| # | Deviation | Basis |
|---|---|---|
| B1 | **CAL combines its two groups by SUM, not product** | Checked against the PDF text layer, not eyeballed: on the formula line the first group's `)` is at column 51, the `+` at column 52, and the second group's `(` at column 53. The operator is unambiguously between the groups. A product would also zero the gradient whenever either factor → 0. `combine="product"` retained as a flag. |
| B2 | **CAL's `βt = 0.9`** | The paper prints "αt = 0.1, and αt = 0.9". The formula uses both `αt` and `βt`; the second is plainly `βt`. |
| B3 | **`αx = 1 + gain·exp(-d/τ)·g`, τ = 3 mm, gain = 4** | The paper says only "the weight of each airway voxel based on the Euclidean distance to the centerline" — it does not say in which direction. Centreline-peak was chosen because the loss is *connectivity*-aware: the centreline core is what must not break. |
| B4 | **GUL's `w_x` set equal to `α_x`** | The paper uses two symbols in one formula and defines neither. |
| B5 | **Differentiable `Dist()` = iterated soft erosion (min-pool), 5 steps** | The paper defers entirely to ref [22]. This is the standard differentiable distance transform and is a.e.-differentiable. |
| B6 | **Peripheral over-sampling = uniform over centreline points, reweighted by `radius^-1`** | "over-sampling guided by the spatial distribution of centerline points" gives no function. Two effects combine: most centreline points already lie in distal branches, and the radius term biases further toward thin ones. |
| B7 | **Flip on the last axis; rotation in the axial plane only, ±10°** | "random horizontal flipping" and "random rotation in [-10°, +10°]" name no axis. L-R flip and in-plane axial rotation are the standard chest-CT reading. |
| B8 | **Stage-1 lung attention applied to the LOSS, not the input** | The deployed pipeline only *crops* to the lung bounding box; it never multiplies the CT by the lung mask. Masking the input would create a train/test mismatch, so the mask is applied to the loss instead — same false-positive suppression, zero mismatch. |
| B9 | **Hard-patch mining schedule** (from epoch 150, refresh every 10, 25 % rotating subset, top-256 centres/case, `p_hard = 0.5`, FN weighted 3× FP) | "early-stage model predictions are used to identify and resample hard patches in later epochs" gives no schedule or criterion. FN is weighted above FP because breakage is what the topology losses target. |
| B10 | **300 epochs × 250 iters, cosine decay, 500-iter warmup, grad-clip 12, weight decay 0** | "Epochs and LR schedule not stated." lr = 2e-3 at batch size 1 is aggressive enough that warmup is close to mandatory. |
| B11 | **PyTorch default initialisation (Kaiming OFF)** | The paper names no initialisation. Measurement also favours the default: enabling Kaiming fan_out raised initial loss on all five models, saturated outputs (mean probability 0.69-0.74 vs a healthy ~0.55), and blew WingsNet's gradient norm from ~0.9 to ~1100 — which `grad_clip=12` would have throttled to nothing. Cause: WingsNet's unnormalised 1×1 heads (`SSEConv.conv2` emits 2 channels, `conv_se` 1), where Kaiming's fan_out assumption does not hold. Available as `kaiming_init=True`, which now skips 1×1 convs. |
| B12 | **Seeded 80/10/10 split, single ATM26 stratum; labeling inherits it** | The paper never states its split sizes. Written once to a manifest so all 5 seg models and every eval share it. The 118 labeling cases inherit the fold their id already occupies in the seg split (`training/data/labeling_splits.py`), so no case is seg-train but labeling-test — otherwise data would leak across the two stages of the decoupled "both" run. |
| B13 | **No resampling; native spacing** | The inference path does not resample. The paper's 0.625 mm resampling is stated only for LIDC-IDRI. |
| B14 | **All cases kept regardless of slice thickness** | The paper's primary-dataset criterion is z ≤ 0.675 mm, which would discard much of AIIB23 given the smaller public case count. |

## C. Limits on comparability with the paper

| # | Issue |
|---|---|
| C1 | **Far fewer cases than the paper's cohorts.** The user has ATM26 Task-1 (299 CT + binary masks) and Task-2 (118 branch-labeled). The paper trains segmentation on 620 volumes and reports external cohorts totalling 3,023; labeling is learned on the authors' full primary set. 299 / 118 is smaller, so expect weaker absolute numbers than the paper. Nothing in the code depends on the count: discovery pairs whatever files exist and the split partitions whatever it finds. |
| C2 | **Table 1's columns are not reproducible.** Its TLD/BND/DSC/clDice/Sensitivity are reported separately under "Lobar", "Segmental" and "Subsegmental" — binary metrics restricted to branches at each anatomical level, which requires the out-of-scope manual labels. `training/evaluate.py` reports **whole-tree** metrics, which are systematically *higher*. The paper's 84.40 "lobar" DSC is not comparable to a whole-tree DSC (typical ATM22 whole-tree SOTA is ~93). |
| C3 | **What *is* falsifiable:** the ≥3-of-5 ensemble should beat every individual member on TLD and BND. That is the claim to check. |

## D. Anatomical labeling stage

| # | Deviation | Basis |
|---|---|---|
| D1 | **Loss = cross-entropy deep-supervision over the six heads**, `Σ_level w·[CE(stage2)+aux·CE(stage1)]`, `aux=0.5` | The paper gives training details but no labeling loss. CE is the standard per-node classification loss; supervising both stages matches the two-stage give→accept architecture. `node_pair`/`outlier` get no explicit target and train implicitly through these terms. |
| D2 | **DropHead `p = 0.1` train / `0` eval** | The 4th `our_net.forward` argument drives the stochastic whole-head drop in `Attention_spd` (`transformer.py:62-77`). The paper never names it; the released code plainly intends a train-time head-drop. |
| D3 | **`heads = 4, dim_head = 32` (shipped config), not the paper's "32 heads"** | The paper text says "32 heads with a hidden dimension of 128", which conflicts with the shipped `heads=4, dim_head=32` (inner_dim = 128). Following the config keeps a trained checkpoint loadable by the provided `our_net`; "32 heads" is read as a slip for `dim_head=32`. |
| D4 | **ATM26 Task-2 labels are SEGMENTAL, decoded as `class2anno` seg id = `v − 1`** | Verified on the data: values are `0..20` (0 = background), and per-label centroids match the segmental anatomy (ids 1-8 left, 9-18 right, 19 trachea, 20 LB7). Per-node GT = majority value over each parsed branch (inverting `transform_pred_data`). Segmental target = `v−1`; lobar target = `SEG_TO_LOB[seg]`. Both real. |
| D5 | **Subsegmental (82) head has no ground truth → `subseg_policy`** | The a/b/c subtype labels were never released publicly. `"base_segment"` (default) supervises the subseg head toward each segment's base subseg class (segment-granularity; keeps the head trained and the released pipeline's subseg output sane); `"ignore"` leaves it unsupervised. Lobar + segmental are unaffected. **This is the one real gap vs. the paper's 3-level claim, forced by the public data.** |
| D6 | **Labeling graph built from the GT binary airway; value 0 = ignored** | The annotations live on the airway, so the GT binary mask gives the cleanest topology (the decoupled "both" run uses it too). A branch whose majority value is 0 (background / unlabeled) is left `ignore_index`; these are rare and reported by `prepare_labeling`. |
| D7 | **Adam lr 5e-4, 600 epochs (paper); one graph/step, grad-accum 4, grad-clip 5, wd 0** | lr and epochs are the paper's; the batching (one variable-size graph per step, accumulate 4) and the clip/wd guards are unspecified and chosen for stability. |
| D8 | **Model-selection metric = subsegmental node accuracy; TreeCons via the provided `calculate_CS`** | The finest level is the hardest and most clinically relevant. TreeCons reuses the released consistency metric unchanged. |

## Proposed upstream patches (not applied)

Offered as one-line diffs; the code here works around all of them.

1. `util/airwayatlas_eval.py:141` — `method="lee\`"` → `method="lee"`. One character; makes the shipped `clDice` work in 3D. **Recommended.**
2. `data_process/tree_parse_skelbased.py:5-6` — move `import pycuda…` inside `adjacent_map_cuda`, so the pure-scipy parsing helpers import without pycuda. Optional.
3. `networks/airway_network_wingsnet.py:101` — `.cuda()` → `.to(x.device)` in `droplayer`. Not needed under one-GPU-per-process, but would make the module device-safe.

No patch is needed for the hardcoded `torch.zeros((1,32,1,128,128))` fallback in the
WingsNet forward: that branch is unreachable when every patch dimension is divisible
by 8, which 128×224×304 satisfies.
