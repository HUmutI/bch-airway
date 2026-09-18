# -*- coding: utf-8 -*-
"""Dataset of cached airway graphs for anatomical-labeling training.

Each item is one whole case (one graph): the labeling transformer classifies every node,
so the natural unit is a case, not a patch. Batching is one graph at a time (variable N);
`our_net` adds the batch dimension internally. Tensors come straight from the `.npz` that
`prepare_labeling.py` wrote, so no pycuda / skeletonisation happens in the training loop.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LabelingGraphDataset(Dataset):

    def __init__(self, config: Dict, split: str, keys: Optional[List[str]] = None):
        self.config = config
        self.split = split
        self.cache_dir = config["label_cache_dir"]

        if keys is None:
            with open(config["label_split_path"], "r") as handle:
                keys = json.load(handle)[split]
        # Keep only cases whose cache actually exists (a case can be split-assigned but
        # fail preprocessing).
        self.keys = [k for k in keys if os.path.exists(self._graph_path(k))]
        missing = len(keys) - len(self.keys)
        if missing:
            print(f"[LabelingGraphDataset:{split}] {missing} cached graphs missing, "
                  f"using {len(self.keys)}")

    def _graph_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key, "graph.npz")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict:
        key = self.keys[index]
        with np.load(self._graph_path(key)) as data:
            item = {
                "x": torch.from_numpy(data["x"]).float(),
                "spd": torch.from_numpy(data["spd"]).long(),
                "mask_top": torch.from_numpy(data["mask_top"]).long(),
                "y_lob": torch.from_numpy(data["y_lob"]).long(),
                "y_seg": torch.from_numpy(data["y_seg"]).long(),
                "y_sub": torch.from_numpy(data["y_sub"]).long(),
                "node_idx": torch.from_numpy(data["node_idx"]).long(),
                "key": key,
            }
        return item


def identity_collate(batch: List[Dict]) -> List[Dict]:
    """No stacking: variable node count per graph, one graph per step."""
    return batch
