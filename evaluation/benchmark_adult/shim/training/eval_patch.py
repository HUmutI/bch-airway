# -*- coding: utf-8 -*-
# Contains small adapted copies from AirMorph (Zhang et al.,
# https://github.com/EndoluminalSurgicalVision-IMR/AirMorph), redistributed with the authors'
# permission (see NOTICE.md at the repository root).
"""Local workarounds for two blockers in the provided evaluation code.

Neither provided file is modified. Both issues are reported in training/README.md and a
one-line upstream patch is proposed there.

1. `util/airwayatlas_eval.py:141` calls
       skeletonize(gt_mask, method="lee`")
   with a stray backtick, so scikit-image raises on **every** 3D `clDice` call. This
   module provides `cldice_fixed`, a faithful copy with the typo removed.

2. `data_process/tree_parse_skelbased.py:5` imports `pycuda.driver` at module scope, so
   the three pure-scipy parsing helpers cannot be imported without a working pycuda
   install. `branch_detected_calculation` needs two of their outputs. The helpers are
   reproduced here verbatim; the pycuda-dependent parts of that module are not needed
   for segmentation evaluation.
"""

import numpy as np
import skimage.measure as measure
from scipy import ndimage
from skimage.morphology import skeletonize

from util.airwayatlas_eval import cl_score, convert_multiclass_to_binary


# ------------------------------------------------------------------------- clDice

def cldice_fixed(*, v_p_pred: np.ndarray, v_l_gt: np.ndarray) -> float:
    """`util.airwayatlas_eval.clDice` with the `method="lee`"` typo corrected."""

    pred_mask = convert_multiclass_to_binary(v_p_pred).astype(np.uint8)
    gt_mask = convert_multiclass_to_binary(v_l_gt).astype(np.uint8)

    if pred_mask.ndim == 2:
        tprec = cl_score(s_skeleton=skeletonize(pred_mask), v_image=gt_mask)
        tsens = cl_score(s_skeleton=skeletonize(gt_mask), v_image=pred_mask)
    elif pred_mask.ndim == 3:
        tprec = cl_score(
            s_skeleton=skeletonize(pred_mask, method="lee"), v_image=gt_mask)
        tsens = cl_score(
            s_skeleton=skeletonize(gt_mask, method="lee"), v_image=pred_mask)
    else:
        raise ValueError(f"expected a 2D or 3D mask, got shape {pred_mask.shape}")

    if (tprec + tsens) == 0:
        return 0.0

    return round(2 * tprec * tsens * 100 / (tprec + tsens), 2)


# ------------------------------------------------------------------ tree parsing

def large_connected_domain(label: np.ndarray) -> np.ndarray:
    """Largest connected component, hole-filled.

    Verbatim from `data_process/tree_parse_skelbased.py::large_connected_domain`.
    """

    cd, num = measure.label(label, return_num=True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    label = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    label = ndimage.binary_fill_holes(label)
    return label.astype(np.uint8)


def skeleton_parsing(skeleton: np.ndarray):
    """Split a skeleton at its branch points.

    Verbatim from `data_process/tree_parse_skelbased.py::skeleton_parsing`.
    """

    neighbor_filter = ndimage.generate_binary_structure(3, 3)
    skeleton_filtered = ndimage.convolve(skeleton, neighbor_filter) * skeleton
    skeleton_parse = skeleton.copy()
    skeleton_parse[skeleton_filtered > 3] = 0

    nd_labels, num = ndimage.label(skeleton_parse, structure=neighbor_filter)
    sizes = np.bincount(nd_labels.ravel())
    mask = sizes > 1
    mask[0] = False
    skeleton_parse = mask[nd_labels]

    cd, num = ndimage.label(skeleton_parse, structure=neighbor_filter)
    return skeleton_parse, cd, num


def airway_parsing(skeleton_parse: np.ndarray, label: np.ndarray, cd: np.ndarray):
    """Assign every airway voxel to its nearest skeleton segment.

    Verbatim from `data_process/tree_parse_skelbased.py::airway_parsing`.
    """

    _, inds = ndimage.distance_transform_edt(1 - skeleton_parse, return_indices=True)
    return cd[inds[0, ...], inds[1, ...], inds[2, ...]] * label


def parse_ground_truth(label: np.ndarray, skeleton: np.ndarray):
    """Produce the (label_parsing, label_skeleton) pair `branch_detected_calculation` wants.

    Returns:
        label_parsing: int array, per-voxel branch id over the airway.
        label_skeleton: uint8 binary skeleton restricted to parsed branches.
    """

    label = (label > 0).astype(np.uint8)
    skeleton = (skeleton > 0).astype(np.uint8)

    skeleton_parse, cd, _ = skeleton_parsing(skeleton)
    label_parsing = airway_parsing(skeleton_parse, label, cd)

    return label_parsing.astype(np.int64), skeleton_parse.astype(np.uint8)
