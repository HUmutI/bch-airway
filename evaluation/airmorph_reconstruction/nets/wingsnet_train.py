# -*- coding: utf-8 -*-
# Derived from AirMorph's networks/airway_network_wingsnet.py (Zhang et al.,
# https://github.com/EndoluminalSurgicalVision-IMR/AirMorph), used with the authors' permission.
"""Training-mode WingsNet: same weights, more outputs.

`networks.airway_network_wingsnet.WingsNet_v3.forward` returns only the decoder-side
aggregation `sigmoid(dc0_1(...))`. Two things it hides matter for training:

1. The eighteen side outputs s0..s17 that WingsNet's group supervision is defined over.
2. The *encoder-side* aggregation head `dc0_0` (plus its `dropout1`). Both modules are
   constructed in `__init__` but never called in the released forward, so in a pure
   inference build their parameters are dead weight. The paper's Fig. 6b3 draws
   WingsNet with **two** aggregation heads, which is what the original design has - so
   the authors' training run must have supervised both. If we did not, `dc0_0` would
   stay at its random initialisation and the checkpoint we ship would differ from the
   authors' in exactly those tensors.

This subclass therefore re-implements `forward` to return both aggregation heads and all
side outputs. It adds no parameters and renames nothing, so `state_dict()` is
key-for-key compatible with the provided `WingsNet_v3` and loads into the released
inference wrappers unchanged.

The released forward contains a fallback branch for the case `d2.shape[2] != e1.shape[2]`
that concatenates hardcoded `torch.zeros((1, 32, 1, 128, 128))` tensors. Those shapes are
only valid for a 128x128 in-plane patch, not the paper's 128x224x304, so instead of
copying it we raise. Keeping every patch dimension divisible by 8 makes the branch
unreachable anyway.
"""

from typing import List, Tuple

import torch

from networks.airway_network_wingsnet import WingsNet_v3


class WingsNetTrain(WingsNet_v3):
    """WingsNet_v3 exposing both aggregation heads and every side output."""

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            main: sigmoid of the decoder aggregation - identical to the released
                forward's return value, i.e. what inference consumes.
            aux: sigmoid of the encoder aggregation (`dc0_0`).
            sides: the eighteen raw side outputs s0..s17, each [B, 2, D, H, W].
        """

        e0, s0 = self.ec1(x)
        e1, s1 = self.ec2(e0)
        e1, s2 = self.ec3(e1)

        e2 = self.pool0(e1)
        e2, s3 = self.ec4(e2)
        e3, s4 = self.ec5(e2)
        e3, s5 = self.ec6(e3)

        e4 = self.pool1(e3)
        e4, s6 = self.ec7(e4)
        e5, s7 = self.ec8(e4)
        e5, s8 = self.ec9(e5)

        e6 = self.pool2(e5)
        e6, s9 = self.ec10(e6)
        e7, s10 = self.ec11(e6)
        e7, s11 = self.ec12(e7)

        e8 = self.up_sample0(e7)
        d0, s12 = self.dc1(torch.cat((e8, e5), 1))
        d0, s13 = self.dc2(d0)

        d1 = self.up_sample1(d0)
        d1, s14 = self.dc3(torch.cat((d1, e3), 1))
        d1, s15 = self.dc4(d1)

        d2 = self.up_sample2(d1)
        if d2.shape[2] != e1.shape[2]:
            raise RuntimeError(
                f"WingsNet depth mismatch: decoder {tuple(d2.shape[2:])} vs encoder "
                f"{tuple(e1.shape[2:])}. Use a patch size whose dimensions are all "
                f"divisible by 8.")

        d2, s16 = self.dc5(torch.cat((d2, e1), 1))
        d2, s17 = self.dc6(d2)

        # Decoder-side aggregation: the head inference reads.
        main = self.dc0_1(self.dropout2(
            torch.cat((s12, s13, s14, s15, s16, s17), 1)))
        # Encoder-side aggregation: WingsNet's second wing.
        aux = self.dc0_0(self.dropout1(
            torch.cat((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11), 1)))

        sides = [s0, s1, s2, s3, s4, s5, s6, s7, s8,
                 s9, s10, s11, s12, s13, s14, s15, s16, s17]

        return torch.sigmoid(main), torch.sigmoid(aux), sides
