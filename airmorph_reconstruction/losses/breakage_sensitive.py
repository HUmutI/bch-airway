# -*- coding: utf-8 -*-
"""Model 5: Breakage-Sensitive loss.

Paper, verbatim:

    L_BS = 1 - sum_i(p_x * c_i) / (sum_i(c_i) + eps)

    "where c_i denotes the i-th voxel value on the centerline map."

IMPORTANT correction (documented deviation). Taken alone this term is degenerate: it is
soft recall over the centreline and contains no penalty for false positives. Setting
p == 1 everywhere gives sum(p*c) == sum(c), hence L_BS ~= 0 - a perfect score for
predicting the entire volume as airway. A model trained on this term by itself collapses
to all-foreground.

In the source it comes from, the breakage-sensitive term is an *auxiliary* loss added to
a base segmentation objective; the paper prints only the auxiliary part. We therefore
train model 5 as

    L = L_dice + bs_weight * L_BS

which preserves the breakage-sensitive behaviour the paper is after (heavy emphasis on
maintaining connectivity along the centreline) while keeping the objective well posed.
Set `bs_weight` high enough that the centreline term dominates the gradient on thin
branches; the Dice term exists to anchor precision.
"""

from typing import Optional

import torch
import torch.nn as nn

from training.losses.weights import apply_loss_mask


class BreakageSensitiveLoss(nn.Module):
    """L_dice + bs_weight * L_BS, per AirMorph model 5."""

    def __init__(
        self,
        bs_weight: float = 1.0,
        smooth: float = 1e-5,
        eps: float = 1e-6,
        require_centerline: bool = True,
    ):
        super().__init__()
        self.bs_weight = bs_weight
        self.smooth = smooth
        self.eps = eps
        self.require_centerline = require_centerline

    def forward(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        meta: Optional[dict] = None,
    ) -> torch.Tensor:
        meta = meta or {}
        loss_mask = meta.get("loss_mask")
        centerline = meta.get("centerline")

        if centerline is None:
            if self.require_centerline:
                raise ValueError(
                    "BreakageSensitiveLoss needs meta['centerline']; the cache must be "
                    "built with skeletons (see training/data/prepare_dataset.py)")
            centerline = target

        prob_m, target_m = apply_loss_mask(prob, target, loss_mask)

        # --- base term: keeps the objective from collapsing to all-foreground
        intersection = (prob_m * target_m).sum()
        denominator = prob_m.sum() + target_m.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        # --- breakage-sensitive term: soft recall along the centreline
        c = centerline
        if loss_mask is not None:
            c = c * loss_mask
        bs_loss = 1.0 - (prob * c).sum() / (c.sum() + self.eps)

        return dice_loss + self.bs_weight * bs_loss
