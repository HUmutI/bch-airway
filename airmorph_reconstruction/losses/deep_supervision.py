# -*- coding: utf-8 -*-
"""WingsNet group supervision wrapper (models 4 and 5).

WingsNet supervises two aggregation heads plus the individual side outputs. The provided
`networks.airway_network_wingsnet.WingsNet_v3.forward` exposes only the decoder
aggregation, so `training/nets/wingsnet_train.py` subclasses it to return
`(main, aux, sides)` without editing the released file. See that module for why the
encoder head must be supervised too.

All side outputs are already upsampled to full resolution by their `SSEConv` /
`SSEConv2` blocks, and carry 2 channels; channel 0 is used, matching the aggregated
head's convention (`models/airway_model_wingsnet.py` reads `pred[:, 0, ...]`). They are
raw conv outputs, so a sigmoid is applied before scoring.

Weighting across heads is not specified by the paper. Full weight on the aggregation
inference actually reads, and smaller equal weights on the auxiliary head and each side
output, is the WingsNet default. Documented deviation.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionLoss(nn.Module):
    """Apply a base loss to both WingsNet aggregation heads and every side output."""

    def __init__(
        self,
        base_loss: nn.Module,
        main_weight: float = 1.0,
        aux_weight: float = 0.5,
        side_weight: float = 0.1,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.main_weight = main_weight
        self.aux_weight = aux_weight
        self.side_weight = side_weight

    def _score(self, raw: torch.Tensor, target: torch.Tensor, meta: Optional[dict],
               already_prob: bool = False) -> torch.Tensor:
        prob = raw[:, 0:1, ...]
        if prob.shape[2:] != target.shape[2:]:
            prob = F.interpolate(
                prob, size=target.shape[2:], mode="trilinear", align_corners=False)
        if not already_prob:
            prob = torch.sigmoid(prob)
        return self.base_loss(prob, target, meta)

    def forward(
        self,
        main_prob: torch.Tensor,
        aux_prob: Optional[torch.Tensor],
        side_outputs: Optional[Sequence[torch.Tensor]],
        target: torch.Tensor,
        meta: Optional[dict] = None,
    ) -> torch.Tensor:
        loss = self.main_weight * self._score(
            main_prob, target, meta, already_prob=True)

        if aux_prob is not None and self.aux_weight > 0:
            loss = loss + self.aux_weight * self._score(
                aux_prob, target, meta, already_prob=True)

        if side_outputs and self.side_weight > 0:
            for side in side_outputs:
                loss = loss + self.side_weight * self._score(side, target, meta)

        return loss
