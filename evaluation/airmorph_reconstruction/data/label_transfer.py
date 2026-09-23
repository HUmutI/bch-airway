# -*- coding: utf-8 -*-
"""Turn the ATM26 Task-2 branch-label volumes into per-graph-node training targets.

VERIFIED against the actual data (not assumed): the ATM26 Task-2 branch labels are a
**segmental** scheme, NOT the subsegmental raw codes. Each airway voxel carries an integer
`v` where:

    v = 0          -> background / unlabeled airway
    v = 1 .. 20    -> segmental class, mapping to configs/class2anno.json["seg"] id (v - 1)

Confirmed empirically (per-label centroids over multiple cases): ids 1-8 are the eight LEFT
segments (LB1+2, LB3, LB4, LB5, LB6, LB8, LB9, LB10), ids 9-18 are the ten RIGHT segments
(RB1..RB10), id 19 is the trachea / proximal trunk (topmost, central, ~70% of airway
voxels), and id 20 (rare; appears in the superseded labels as LB7) -> seg 19. So the
mapping is exactly `class2anno seg id = v - 1`.

Consequence: the labeling transformer's **lobar (6)** and **segmental (20)** heads can be
supervised with real ground truth (lobar derived from segmental by anatomy). Its
**subsegmental (82)** head has NO ground truth in the public ATM26 data — the a/b/c subtype
annotations were never released. `subseg_policy` controls what target, if any, the subseg
head receives (see below).

The transfer assigns each graph node the **majority** label over the voxels of its parsed
branch (`airway_parse == node_idx[j] + 1`, the inverse of the painting done in
`data_process/data_save_cuda.py::transform_pred_data`).
"""

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: segmental class id -> lobar class id (configs/class2anno.json "lob":
#: 0 trachea, 1 LUL, 2 LLL, 3 RUL, 4 RML, 5 RLL). Standard bronchial anatomy.
SEG_TO_LOB = {
    0: 1, 1: 1, 2: 1, 3: 1,            # LB1+2, LB3, LB4, LB5      -> Left Upper Lobe
    4: 2, 5: 2, 6: 2, 7: 2, 19: 2,     # LB6, LB8, LB9, LB10, LB7  -> Left Lower Lobe
    8: 3, 9: 3, 10: 3,                 # RB1, RB2, RB3             -> Right Upper Lobe
    11: 4, 12: 4,                      # RB4, RB5                  -> Right Middle Lobe
    13: 5, 14: 5, 15: 5, 16: 5, 17: 5, # RB6..RB10                -> Right Lower Lobe
    18: 0,                             # trachea
}

#: segmental class id -> base subsegmental class id in configs/class2anno.json "subseg".
#: Used only when subseg_policy == "base_segment": with no subtype labels available, the
#: subseg head is (optionally) supervised toward the segment's own base subseg class, so it
#: learns segment-granularity subseg predictions rather than staying at random init.
SEG_TO_SUBSEG_BASE = {
    0: 1,    # LB1+2 -> subseg "LB1+2"
    1: 8,    # LB3
    2: 15,   # LB4
    3: 18,   # LB5
    4: 21,   # LB6
    5: 28,   # LB8
    6: 31,   # LB9
    7: 34,   # LB10
    8: 41,   # RB1
    9: 44,   # RB2
    10: 47,  # RB3
    11: 50,  # RB4
    12: 53,  # RB5
    13: 56,  # RB6
    14: 63,  # RB7
    15: 66,  # RB8
    16: 69,  # RB9
    17: 72,  # RB10
    18: 0,   # trachea -> subseg "trachea"
    19: 79,  # LB7
}

#: highest ATM26 label value we accept (v - 1 must be a valid seg id 0..19).
_MAX_VALUE = 20


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_segment_maps(subseg_policy: str = "base_segment",
                       ignore_index: int = -1) -> Dict[int, Tuple[int, int, int]]:
    """Precompute  ATM26 value v (1..20) -> (lob_id, seg_id, sub_id).

    `subseg_policy`:
      - "ignore":       sub_id = ignore_index for every node (subseg head unsupervised).
      - "base_segment": sub_id = the segment's base subseg class (segment-granularity).
    """

    if subseg_policy not in ("ignore", "base_segment"):
        raise ValueError(f"unknown subseg_policy {subseg_policy!r}")

    table: Dict[int, Tuple[int, int, int]] = {}
    for v in range(1, _MAX_VALUE + 1):
        seg = v - 1
        if seg not in SEG_TO_LOB:
            continue
        lob = SEG_TO_LOB[seg]
        sub = ignore_index if subseg_policy == "ignore" else SEG_TO_SUBSEG_BASE[seg]
        table[v] = (lob, seg, sub)
    return table


def transfer_node_labels(
    airway_parse: np.ndarray,
    node_idx: np.ndarray,
    branch_label: np.ndarray,
    segment_maps: Dict[int, Tuple[int, int, int]],
    ignore_index: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Assign lobar/segmental/subsegmental targets to each of the N graph nodes.

    Args:
        airway_parse: [D,H,W] int volume, per-voxel branch id (values 1..K).
        node_idx:     [N] int, node j -> branch id (node_idx[j] + 1 in airway_parse).
        branch_label: [D,H,W] int volume, per-voxel ATM26 segmental value (same frame).
        segment_maps: from build_segment_maps().

    Returns:
        (y_lob, y_seg, y_sub) each [N] int64 (ignore_index where unresolved) and a stats dict.
    """

    if airway_parse.shape != branch_label.shape:
        raise ValueError(
            f"airway_parse {airway_parse.shape} != branch_label {branch_label.shape}")

    n = int(node_idx.shape[0])
    y_lob = np.full(n, ignore_index, dtype=np.int64)
    y_seg = np.full(n, ignore_index, dtype=np.int64)
    y_sub = np.full(n, ignore_index, dtype=np.int64)

    branch_label = np.asarray(branch_label)
    stats = {"nodes": n, "empty": 0, "background": 0, "unknown": 0, "resolved": 0}

    for j in range(n):
        parse_id = int(node_idx[j]) + 1
        voxels = branch_label[airway_parse == parse_id]
        if voxels.size == 0:
            stats["empty"] += 1
            continue

        codes = voxels.astype(np.int64)
        codes = codes[codes >= 0]
        if codes.size == 0:
            stats["empty"] += 1
            continue
        v = int(np.bincount(codes).argmax())

        if v == 0:                         # background / unlabeled airway -> leave ignored
            stats["background"] += 1
            continue
        if v not in segment_maps:
            stats["unknown"] += 1
            continue

        lob, seg, sub = segment_maps[v]
        y_lob[j], y_seg[j], y_sub[j] = lob, seg, sub
        stats["resolved"] += 1

    return y_lob, y_seg, y_sub, stats


def _self_check() -> None:
    """Validate the segmental derivation against the authors' class2anno names."""

    class2anno = _load_json(os.path.join(_PROJECT_ROOT, "configs/class2anno.json"))
    seg_names = class2anno["seg"]
    sub_names = class2anno["subseg"]

    table = build_segment_maps("base_segment")
    for v, (lob, seg, sub) in table.items():
        seg_name = seg_names[str(seg)]
        sub_name = sub_names[str(sub)]
        assert seg_name == sub_name, f"v={v}: seg {seg_name!r} vs subseg base {sub_name!r}"
    print(f"[label_transfer] self-check passed for {len(table)} segmental values")
    print(f"[label_transfer] v=19 -> {table[19]} (expect lob0/seg18=trachea, subbase0)")
    print(f"[label_transfer] v=1  -> {table[1]}  (expect lob1/seg0=LB1+2, subbase1)")
    print(f"[label_transfer] v=9  -> {table[9]}  (expect lob3/seg8=RB1, subbase41)")

    ignore_tbl = build_segment_maps("ignore", ignore_index=-1)
    assert all(sub == -1 for _, _, sub in ignore_tbl.values())
    print("[label_transfer] 'ignore' policy leaves every subseg target at ignore_index")


if __name__ == "__main__":
    _self_check()
