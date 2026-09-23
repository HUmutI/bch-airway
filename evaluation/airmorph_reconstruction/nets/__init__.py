# -*- coding: utf-8 -*-
"""Network construction for AirMorph segmentation training.

The two architectures behave differently enough that the trainer must not treat them
interchangeably:

* `UNet3D` returns **logits** while `self.training` is True and applies its final
  activation only in eval mode (`networks/airway_network.py:231`). `finalsigmoid=1` means
  that activation is a Sigmoid, not a Softmax, and inference thresholds channel 1
  (`models/airway_model.py:62`). So the deployed foreground probability is
  `sigmoid(logits[:, 1])` - training with a softmax would silently produce checkpoints
  that are wrong at inference.
* `WingsNetTrain` applies `torch.sigmoid` inside `forward` unconditionally, exactly as
  the released `WingsNet_v3` does, and inference reads channel 0.

`forward_probability` hides that difference behind one contract: given a network and an
input, return the airway probability as [B, 1, D, H, W] plus whatever auxiliary outputs
the deep-supervision loss needs.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from networks.airway_network import UNet3D
from training.nets.wingsnet_train import WingsNetTrain


def init_weights_kaiming(module: nn.Module) -> None:
    """Kaiming (fan_out, leaky_relu) init for 3x3x3 convolutions. OFF by default.

    Neither the paper nor the released networks specify an initialisation, so PyTorch's
    default is what the authors would have trained with, and it is what we use.

    This function is kept as an opt-in (`config["kaiming_init"] = True`) but measurement
    argues against it. Applying Kaiming fan_out to every conv made things strictly worse
    on all five models: initial loss rose, outputs saturated (mean probability 0.69-0.74
    instead of a healthy ~0.55), and gradient norms grew. On WingsNet the effect was
    catastrophic - gradient norm went from ~0.8 to ~1100, which `grad_clip=12` would have
    throttled to nothing.

    The cause is WingsNet's unnormalised 1x1 heads: `SSEConv.conv2` emits 2 channels and
    `conv_se` emits 1, so fan_out is 2 or 1 and Kaiming assigns them a std near 1.0 over
    a 32-channel input. Those layers are followed by a sigmoid or feed a side output, not
    by a norm + ReLU, so Kaiming's assumption does not hold. Hence 1x1 convolutions are
    skipped here even when the option is enabled.
    """

    for submodule in module.modules():
        if isinstance(submodule, (nn.Conv3d, nn.ConvTranspose3d)):
            if tuple(submodule.kernel_size) == (1, 1, 1):
                continue
            nn.init.kaiming_normal_(
                submodule.weight, mode="fan_out", nonlinearity="leaky_relu")
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)


def build_network(config: Dict, device: str) -> nn.Module:
    """Instantiate the network for one of the five ensemble members."""

    kind = config["network"]
    arch = config["arch"]

    if kind == "unet3d":
        # UNet3D moves its submodules internally using this list.
        net = UNet3D(
            in_channels=arch["in_channels"],
            out_channels=arch["out_channels"],
            finalsigmoid=arch["finalsigmoid"],
            fmaps_degree=arch["fmaps_degree"],
            fmaps_layer_number=arch["fmaps_layer_number"],
            layer_order=config["layer_order"],
            GroupNormNumber=arch["GroupNormNumber"],
            device=[device],
        )
    elif kind == "wingsnet":
        net = WingsNetTrain(
            in_channel=arch["in_channel"], n_classes=arch["n_classes"]).to(device)
    else:
        raise KeyError(f"unknown network '{kind}'; expected 'unet3d' or 'wingsnet'")

    if config.get("kaiming_init", False):
        init_weights_kaiming(net)

    return net


def forward_probability(
    net: nn.Module,
    images: torch.Tensor,
    config: Dict,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Run a forward pass and return (airway probability, aux head, side outputs)."""

    output = net(images)

    if config["network"] == "wingsnet":
        main, aux, sides = output
        return main[:, 0:1, ...], aux, sides

    channel = config["output_channel"]
    selected = output[:, channel:channel + 1, ...]
    # UNet3D returns logits in train mode and already-sigmoided values in eval mode.
    if net.training:
        return torch.sigmoid(selected), None, None
    return selected, None, None


class ProbabilityPredictor(nn.Module):
    """Single-channel probability wrapper for MONAI's `sliding_window_inference`.

    Sliding-window inference needs a predictor returning one tensor, but `WingsNetTrain`
    returns a tuple and `UNet3D` returns two channels. This adapter normalises both to
    [B, 1, D, H, W] probabilities so validation sees exactly what the deployed
    `segmentator/airway_segmentator.py` ensemble consumes.
    """

    def __init__(self, net: nn.Module, config: Dict):
        super().__init__()
        self.net = net
        self.config = config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        prob, _, _ = forward_probability(self.net, images, self.config)
        return prob


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a DistributedDataParallel `module.` prefix if present."""

    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key[len("module."):] if key.startswith("module.") else key: value
            for key, value in state_dict.items()}


def save_checkpoint(
    net: nn.Module,
    path: str,
    checkpoint_format: str,
    extra: Optional[Dict] = None,
) -> None:
    """Write a checkpoint in the exact layout the provided inference code loads.

    `models/airway_model.py` does `load_state_dict(torch.load(path))` - a bare state dict.
    `models/airway_model_wingsnet.py` and `models/airway_model_break.py` do
    `load_state_dict(torch.load(path)["state_dict"])`. Getting this wrong means the
    trained weights will not load into the released pipeline at all.
    """

    state_dict = strip_module_prefix(net.state_dict())
    state_dict = {key: value.detach().cpu() for key, value in state_dict.items()}

    if checkpoint_format == "raw":
        torch.save(state_dict, path)
    elif checkpoint_format == "state_dict":
        payload = {"state_dict": state_dict}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
    else:
        raise KeyError(
            f"unknown checkpoint_format '{checkpoint_format}'; expected 'raw' or "
            f"'state_dict'")


__all__ = [
    "ProbabilityPredictor",
    "WingsNetTrain",
    "build_network",
    "forward_probability",
    "init_weights_kaiming",
    "save_checkpoint",
    "strip_module_prefix",
]
