# -*- coding: utf-8 -*-
"""Model 4: General Union Loss (WingsNet, Zheng et al. 2021).

Paper, verbatim:

    L_GU = 1 - sum_x(w_x * p_x^rl * g_x) / sum_x(alpha_x * (alpha * p_x + beta * g_x))

    with rl = 0.7, alpha = 0.2, beta = 0.7

The symbol `alpha` is overloaded in the printed formula - it names both the scalar 0.2
and the per-voxel weight `alpha_x`. In the original General Union formulation the same
per-voxel weight appears in numerator and denominator, so we read `w_x` and `alpha_x`
as the same map and build it from the cached distance-to-centreline (see weights.py).

The `p^rl` exponent with rl < 1 is what makes this loss sensitive to faint peripheral
branches: it inflates the contribution of low-confidence foreground predictions.
"""

from typing import Optional

import torch
import torch.nn as nn

from training.losses.weights import apply_loss_mask, build_voxel_weights


class GeneralUnionLoss(nn.Module):
    """L_GU, per AirMorph model 4."""

    def __init__(
        self,
        rl: float = 0.7,
        alpha: float = 0.2,
        beta: float = 0.7,
        weight_tau: float = 3.0,
        weight_fg_gain: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.rl = rl
        self.alpha = alpha
        self.beta = beta
        self.weight_tau = weight_tau
        self.weight_fg_gain = weight_fg_gain
        self.eps = eps

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

        w_x = build_voxel_weights(
            target,
            dist_to_centerline,
            tau=self.weight_tau,
            fg_gain=self.weight_fg_gain,
        )
        if loss_mask is not None:
            w_x = w_x * loss_mask

        # clamp before the fractional power: p^0.7 has an infinite derivative at 0.
        p = prob_m.clamp(min=self.eps, max=1.0)

        numerator = (w_x * torch.pow(p, self.rl) * target_m).sum()
        denominator = (w_x * (self.alpha * prob_m + self.beta * target_m)).sum()

        return 1.0 - numerator / (denominator + self.eps)
