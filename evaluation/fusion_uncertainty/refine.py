"""Branch-relative voxel consensus: trim each branch's boundary shell to the
agreement its own centreline enjoys, leaving the core and any protrusion alone.

The union keeps every voxel any model marks, so a branch all three models see
carries the fattest of the three boundaries, and that shell is where the
union's false positives live, not in false branches (pruning every false
branch with the ground truth itself raises DSC by one point). A voxel is kept
when the support behind it is at least a fraction kappa of the support behind
its branch's centreline. A branch three models agree on is trimmed back
towards a majority consensus; a distal branch one model found keeps all its
voxels, because nothing in it has more support than its centreline.

Two guards keep the tree intact. Within a branch, the core closer to the
centreline than `core_frac` of the branch radius is always kept; and voxels
further from the centreline than (1 + `protrusion`) times the radius are
always kept, because they are not boundary shell but a side branch or a tip
the skeleton did not resolve. The seeded-component rule is re-applied last.
"""
import numpy as np
from scipy import ndimage

import common
import tree


def geometry(M, branch_map, tab, spacing):
    """Distance of every mask voxel to the mask skeleton (mm) and the radius of
    its branch (mm), both on the full grid (0 outside the mask)."""
    M = M.astype(bool)
    sl = tree.bbox_slices(M)
    sub = M[sl]
    skel = tree.skeletonize(sub).astype(bool)
    dist = ndimage.distance_transform_edt(~skel, sampling=tuple(spacing)).astype(np.float32)
    d = np.zeros(M.shape, np.float32); d[sl] = dist * sub
    r = tab["r_mean_mm"].to_numpy(np.float32).copy(); r[0] = 0
    return d, r[branch_map]


def refine(M, conf, branch_map, tab, kappa=0.5, core_frac=0.5, protrusion=None,
           max_tau=0.6, spacing=None, geom=None, core_min_vox=1.0):
    M = M.astype(bool)
    cs = tab["conf_skel"].to_numpy(np.float64)
    tau_b = np.clip(kappa * cs, 0.0, max_tau); tau_b[0] = 0.0
    keep = conf >= tau_b.astype(np.float32)[branch_map]
    if core_frac > 0 or protrusion is not None:
        d, rmap = geom if geom is not None else geometry(M, branch_map, tab, spacing)
        vox = float(np.max(spacing))
        if core_frac > 0:
            keep |= d <= np.maximum(core_frac * rmap, core_min_vox * vox)
        if protrusion is not None:
            keep |= d > (1.0 + protrusion) * rmap + vox
    out = (keep & M).astype(np.uint8)
    seed = (M & (branch_map > 0) & (conf >= 0.99)).astype(np.uint8)
    return common.keep_seeded_component(out, seed)
