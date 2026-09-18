# -*- coding: utf-8 -*-
"""Model 1: Dice with Focal loss.

Paper, verbatim:

    L_Dice w/ Focal = -( 2*sum(p_x g_x) / sum(p_x + g_x)
                         - (1/|X|) * sum((1 - p_x)^2 * log(p_x)) )

Sign note (documented deviation): as printed, expanding the outer minus gives
`-Dice + (1/|X|) sum((1-p)^2 log p)`. Because log(p) <= 0, that second term is
minimised by driving p -> 0 on the foreground, i.e. the printed expression is
degenerate. The intended objective is plainly Dice loss plus the standard focal loss,
`-Dice - (1/|X|) sum((1-p)^2 log p)`, which is what we implement. Focal gamma = 2 as
printed. We use `1 - Dice` rather than `-Dice`; the two differ by a constant and have
identical gradients, but the loss stays non-negative and is easier to read in logs.
"""

from typing import Optional

import torch
import torch.nn as nn

from training.losses.weights import apply_loss_mask


class DiceFocalLoss(nn.Module):
    """Dice + focal, per AirMorph model 1."""

    def __init__(self, gamma: float = 2.0, smooth: float = 1e-5, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps

    def forward(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        meta: Optional[dict] = None,
    ) -> torch.Tensor:
        meta = meta or {}
        loss_mask = meta.get("loss_mask")

        prob_m, target_m = apply_loss_mask(prob, target, loss_mask)

        intersection = (prob_m * target_m).sum()
        denominator = prob_m.sum() + target_m.sum()
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice

        # Standard binary focal term on the probability of the true class.
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        p_t = p_t.clamp(min=self.eps, max=1.0 - self.eps)
        focal = -((1.0 - p_t) ** self.gamma) * torch.log(p_t)
        if loss_mask is not None:
            focal = focal * loss_mask
            focal_loss = focal.sum() / loss_mask.sum().clamp(min=1.0)
        else:
            focal_loss = focal.mean()

        return dice_loss + focal_loss
