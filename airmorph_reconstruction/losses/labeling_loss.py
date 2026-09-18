# -*- coding: utf-8 -*-
"""Loss for the AirMorph anatomical-labeling transformer.

The paper (p25) gives the labeling training details (Adam, lr 5e-4, 600 epochs) but **no
explicit loss**. The faithful, standard choice for hierarchical per-node classification is
cross-entropy deep-supervision over the six classification heads `our_net.forward` returns:

    L = sum_level w_level * [ CE(stage2_level, y_level) + aux * CE(stage1_level, y_level) ]

with level in {lobar, segmental, subsegmental}. Stage-2 (`x*_2`) is what inference reads,
so it carries full weight; stage-1 (`x*_1`, the "give" stage) is an auxiliary deep-supervision
head weighted by `aux`. The `node_pair*` (learned topology prior) and `outlier*` tensors are
NOT given an explicit target — they lie in the forward path and are trained implicitly
through these six CE terms, exactly as the released architecture wires them.

`ignore_index` drops nodes with no resolvable ground-truth label (see label_transfer).
Optional inverse-frequency class weighting is available but off by default (the paper
mentions none).
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelingDeepSupervisionLoss(nn.Module):

    def __init__(
        self,
        aux_weight: float = 0.5,
        level_weights: Sequence[float] = (1.0, 1.0, 1.0),
        ignore_index: int = -1,
        class_weights: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ):
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.level_weights = tuple(float(w) for w in level_weights)
        self.ignore_index = int(ignore_index)
        # class_weights: optional (w_lob, w_seg, w_sub) tensors, or None entries.
        self.class_weights = class_weights

    def _ce(self, logits: torch.Tensor, target: torch.Tensor, level: int) -> torch.Tensor:
        # If every node at this level is ignored (e.g. the subseg head under the "ignore"
        # policy, or a graph with no resolved labels), F.cross_entropy would average over
        # zero elements and return NaN. Contribute a real zero instead.
        if bool((target != self.ignore_index).sum() == 0):
            return logits.sum() * 0.0
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights[level]
            if weight is not None:
                weight = weight.to(logits.device, dtype=logits.dtype)
        return F.cross_entropy(
            logits, target, weight=weight, ignore_index=self.ignore_index)

    def forward(
        self,
        outputs: Tuple[torch.Tensor, ...],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Args:
            outputs: the 10-tuple from our_net.forward
                     (x1_1, x2_1, x3_1, x1_2, x2_2, x3_2, node_pair1, node_pair2,
                      outlier_1, outlier_2). Each x*_* is [N, C].
            targets: {"y_lob": [N], "y_seg": [N], "y_sub": [N]}.
        """

        x1_1, x2_1, x3_1, x1_2, x2_2, x3_2 = outputs[:6]
        y = (targets["y_lob"], targets["y_seg"], targets["y_sub"])
        stage2 = (x1_2, x2_2, x3_2)
        stage1 = (x1_1, x2_1, x3_1)

        total = x1_2.new_zeros(())
        logs: Dict[str, float] = {}
        names = ("lob", "seg", "sub")

        for level in range(3):
            main = self._ce(stage2[level], y[level], level)
            aux = self._ce(stage1[level], y[level], level)
            term = self.level_weights[level] * (main + self.aux_weight * aux)
            total = total + term
            logs[f"ce_{names[level]}"] = float(main.detach().item())

        logs["loss"] = float(total.detach().item())
        return total, logs


def build_class_weights(dataset, num_classes: Tuple[int, int, int],
                        ignore_index: int = -1):
    """Inverse-frequency class weights per level from a LabelingGraphDataset (optional)."""

    import numpy as np

    counts = [np.zeros(c, dtype=np.float64) for c in num_classes]
    for i in range(len(dataset)):
        item = dataset[i]
        for level, key in enumerate(("y_lob", "y_seg", "y_sub")):
            y = item[key].numpy()
            y = y[y != ignore_index]
            if y.size:
                counts[level] += np.bincount(y, minlength=num_classes[level])

    weights = []
    for c in counts:
        freq = c / max(c.sum(), 1.0)
        w = 1.0 / np.clip(freq, 1e-6, None)
        w = w / w.mean()           # normalise so the average weight is 1
        weights.append(torch.from_numpy(w.astype("float32")))
    return weights
