# -*- coding: utf-8 -*-
"""Models 2 and 3: Connectivity-Aware Loss, and CAL + local-sensitive distance.

Paper, verbatim:

    L_CAL = { 1 - sum_x(p_x g_x) / (alpha_t * sum_x(p_x) + beta_t * sum_x(g_x)) }
            + { sum_x alpha_x * CE(p_x, g_x) }

    "where alpha_t = 0.1, and alpha_t = 0.9"

    L_CAL w/ LSD = L_CAL + || Dist(g_x) - Dist(p_x) ||_2

The two brace groups are combined by ADDITION, not multiplication. This was checked
against the PDF's text layer rather than eyeballed: on the formula line the closing
paren of the first group sits at column 51, the `+` at column 52, and the opening paren
of the second group at column 53. The operator is unambiguously between the groups.
(A product would also be pathological - the gradient vannotatores whenever either factor
approaches zero.) `combine="product"` is retained as a flag for ablation.

Two further documented readings:

1. The paper prints `alpha_t` twice. The second is beta_t - the formula uses both, and
   0.1 / 0.9 are the standard Tversky-style asymmetric weights. Read as
   alpha_t = 0.1 (on the prediction), beta_t = 0.9 (on the ground truth).

2. The second group is printed as a bare sum over voxels. Taken literally that is a sum
   over ~8.7M voxels per patch, which is ~10^6 times the first group's magnitude and
   would make the overlap term inert. We use the weighted mean
   `sum(alpha_x * CE) / sum(alpha_x)` instead, so the two terms are comparable.
   `ce_reduction="sum"` preserves the literal form.
"""

from typing import Optional

import torch
import torch.nn as nn

from training.losses.distance_transform import LocalSensitiveDistanceLoss
from training.losses.weights import apply_loss_mask, build_voxel_weights


class ConnectivityAwareLoss(nn.Module):
    """L_CAL, optionally with the local-sensitive distance term (model 3)."""

    def __init__(
        self,
        alpha_t: float = 0.1,
        beta_t: float = 0.9,
        combine: str = "sum",
        ce_reduction: str = "mean",
        use_lsd: bool = False,
        lsd_iterations: int = 5,
        lsd_weight: float = 1.0,
        weight_tau: float = 3.0,
        weight_fg_gain: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if combine not in ("sum", "product"):
            raise ValueError("combine must be 'sum' or 'product'")
        if ce_reduction not in ("mean", "sum"):
            raise ValueError("ce_reduction must be 'mean' or 'sum'")
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.combine = combine
        self.ce_reduction = ce_reduction
        self.use_lsd = use_lsd
        self.lsd_weight = lsd_weight
        self.weight_tau = weight_tau
        self.weight_fg_gain = weight_fg_gain
        self.eps = eps
        self.lsd = LocalSensitiveDistanceLoss(iterations=lsd_iterations) if use_lsd else None

    def forward(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        meta: Optional[dict] = None,
    ) -> torch.Tensor:
        meta = meta or {}
        loss_mask = meta.get("loss_mask")
        dist_to_centerline = meta.get("dist_to_centerline")

        prob_m, target_m = apply_loss_mask(prob, target, loss_mask)

        # --- first factor: asymmetric overlap term
        intersection = (prob_m * target_m).sum()
        denominator = self.alpha_t * prob_m.sum() + self.beta_t * target_m.sum()
        overlap_factor = 1.0 - intersection / (denominator + self.eps)

        # --- second factor: centreline-distance-weighted cross entropy
        alpha_x = build_voxel_weights(
            target,
            dist_to_centerline,
            tau=self.weight_tau,
            fg_gain=self.weight_fg_gain,
        )
        if loss_mask is not None:
            alpha_x = alpha_x * loss_mask

        p = prob.clamp(min=self.eps, max=1.0 - self.eps)
        cross_entropy = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
        weighted_ce = (alpha_x * cross_entropy).sum()
        if self.ce_reduction == "mean":
            weighted_ce = weighted_ce / (alpha_x.sum() + self.eps)

        if self.combine == "sum":
            loss = overlap_factor + weighted_ce
        else:
            loss = overlap_factor * weighted_ce

        if self.use_lsd:
            loss = loss + self.lsd_weight * self.lsd(prob, target)

        return loss
