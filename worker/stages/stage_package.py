#!/usr/bin/env python
"""Collect what the user gets, in the orientation they uploaded.

The pipeline works in LPS. The user's scan may not be, and a mask that does not
overlay the scan it came from is useless to a radiologist, so every output
here is rotated back to the source orientation before it is offered. The
viewer loads the same geometry, so what is shown is what is downloaded.

    results/
      image.nii.gz              the uploaded volume, as uploaded
      airway_mask.nii.gz        fused airway, binary
      airway_lobar.nii.gz       lobar label per airway voxel (1-6)
      airway_segmental.nii.gz   segmental label per airway voxel
      airway_branches.nii.gz    branch instance id per airway voxel
      airway_centreline.nii.gz  the skeleton
      *_view.nii.gz             the same, resampled for the browser (see below)
      mesh_airway.mz3           airway surface, world mm
      mesh_lobe_<k>.mz3         one surface per lobar label k
      itksnap_*_labels.txt      ITK-SNAP label descriptions in the site's colours
      labels.json               class names and per-branch names
      summary.json              counts and volumes for the page
      airway_results.zip        all of the above

The browser keeps the CT on the GPU at 4 bytes a voxel, so a 512x512x950 scan
would need a gigabyte. The *_view files are the same data resampled to an
isotropic grid of at most config.VIEW_MAX_VOXELS voxels, never finer than the
source. Downloads are untouched.

Runs in the AirMorph environment, CPU only.
"""
import argparse, json, os, shutil, struct, sys, zipfile
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.measure import marching_cubes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as C  # noqa: E402


def to_source(img, code):
    if code == "LPS":
        return img
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation(code)
    return f.Execute(img)


# ------------------------------------------------------------------ display copy
def view_grid(img, max_voxels):
    """Isotropic spacing, coarsest of (source max spacing, what fits the budget)."""
    size = np.array(img.GetSize(), float)
    sp = np.array(img.GetSpacing(), float)
    extent = size * sp
    s = max(sp.max(), float(np.cbrt(np.prod(extent) / max_voxels)))
    s = float(np.ceil(s * 20) / 20)                       # 0.05 mm steps
    new_size = [int(max(1, round(e / s))) for e in extent]
    return s, new_size


def resample(img, spacing, size, nearest):
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([spacing] * 3)
    r.SetSize(size)
    r.SetOutputOrigin(img.GetOrigin())
    r.SetOutputDirection(img.GetDirection())
    r.SetInterpolator(sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear)
    r.SetDefaultPixelValue(0 if nearest else -1024)
    return r.Execute(img)


# ------------------------------------------------------------------------ meshes
def write_mz3(path, verts, faces):
    """MZ3: 16-byte header, int32 faces, float32 vertices. Read natively by NiiVue."""
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<HHIII", 0x5A4D, 3, len(faces), len(verts), 0))
        fh.write(faces.tobytes())
        fh.write(verts.tobytes())


def mesh_from_mask(arr_zyx, img, sigma_vox=0.7, min_faces=100):
    """Surface of a binary mask in world mm, smoothed just enough to lose the stairs."""
    if not arr_zyx.any():
        return None
    # work in a padded bounding box: marching cubes on the whole volume is slow
    idx = np.argwhere(arr_zyx)
    lo = np.maximum(idx.min(0) - 3, 0)
    hi = np.minimum(idx.max(0) + 4, arr_zyx.shape)
    sub = arr_zyx[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    if sigma_vox > 0:
        sub = ndimage.gaussian_filter(sub, sigma_vox)
    if sub.max() <= 0.5:
        sub = arr_zyx[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    verts, faces, _, _ = marching_cubes(sub, level=0.5, step_size=1)
    verts = verts + lo                                   # (z, y, x) voxel index
    # drop tiny islands (smoothing crumbs)
    n = len(verts)
    lab = np.zeros(n, int) - 1
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    a = sp.coo_matrix((np.ones(len(faces) * 3),
                       (np.r_[faces[:, 0], faces[:, 1], faces[:, 2]],
                        np.r_[faces[:, 1], faces[:, 2], faces[:, 0]])), shape=(n, n))
    ncomp, lab = connected_components(a, directed=False)
    fl = lab[faces[:, 0]]
    counts = np.bincount(fl, minlength=ncomp)
    keep_c = counts >= min_faces
    if keep_c.sum() == 0:
        keep_c[counts.argmax()] = True
    keep_f = keep_c[fl]
    faces = faces[keep_f]
    used = np.unique(faces)
    remap = np.zeros(n, int) - 1
    remap[used] = np.arange(len(used))
    faces = remap[faces]
    verts = verts[used]
    # voxel (z,y,x) -> ITK index (x,y,z) -> physical LPS
    D = np.array(img.GetDirection()).reshape(3, 3)
    S = np.diag(img.GetSpacing())
    O = np.array(img.GetOrigin())
    xyz = verts[:, ::-1] @ (D @ S).T + O
    # NiiVue's world space is RAS. LPS -> RAS flips x and y.
    xyz[:, 0] *= -1
    xyz[:, 1] *= -1
    return xyz, faces


# ------------------------------------------------------------- ITK-SNAP labels
def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def itksnap_labels(path, names, colours):
    """ITK-SNAP 'Label Descriptions' format, one line per label value."""
    lines = ["################################################",
             "# ITK-SnAP Label Description File",
             "# File format: IDX -R- -G- -B- -A-- VIS MSH LABEL",
             "################################################",
             '    0     0    0    0        0  0  0    "Clear Label"']
    for v in sorted(names, key=int):
        r, g, b = hex_rgb(colours[v])
        lines.append(f'{int(v):5d} {r:5d} {g:4d} {b:4d}        1  1  1    "{names[v]}"')
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--class2anno", default=None)
    a = ap.parse_args()
    case = os.path.join(a.job_dir, "case")
    cid = os.path.basename(case)
    res = os.path.join(a.job_dir, "results")
    os.makedirs(res, exist_ok=True)
    code = json.load(open(os.path.join(case, "orient.json")))["source_orientation"]
    pal = json.load(open(C.PALETTE))

    shutil.copy2(os.path.join(a.job_dir, "input.nii.gz"), os.path.join(res, "image.nii.gz"))

    outputs = {
        "airway_mask.nii.gz":       "union3_lcc.nii.gz",
        "airway_lobar.nii.gz":      f"{cid}_pred_lob.nii.gz",
        "airway_segmental.nii.gz":  f"{cid}_pred_seg.nii.gz",
        "airway_branches.nii.gz":   f"{cid}_parse.nii.gz",
        "airway_centreline.nii.gz": "union3_lcc_skeleton.nii.gz",
    }
    stats = {}
    src_imgs = {}
    for out, src in outputs.items():
        p = os.path.join(case, src)
        if not os.path.exists(p):
            print(f"missing {src}, skipping {out}", flush=True)
            continue
        img = to_source(sitk.ReadImage(p), code)
        src_imgs[out] = img
        if out == "airway_mask.nii.gz":
            arr = sitk.GetArrayViewFromImage(img)
            sp = img.GetSpacing()
            stats["airway_voxels"] = int((arr > 0).sum())
            stats["airway_volume_ml"] = round(float((arr > 0).sum()) * sp[0] * sp[1] * sp[2] / 1000, 2)
        sitk.WriteImage(img, os.path.join(res, out), True)
        print(f"wrote {out}", flush=True)

    # Names for the viewer. class2anno.json is keyed by 0-based class id under
    # "lob", "seg" and "subseg", while the label volumes store class + 1 so that
    # 0 can mean background. The map written here is keyed by the voxel value,
    # which is what the viewer has in hand.
    labels = {}
    anno_p = os.path.join(case, f"{cid}_anno.json")
    if a.class2anno and os.path.exists(a.class2anno) and os.path.exists(anno_p):
        c2a = json.load(open(a.class2anno))
        labels = {
            "value_is_class_plus_one": True,
            "lobar": {str(int(k) + 1): v for k, v in c2a["lob"].items()},
            "segmental": {str(int(k) + 1): v for k, v in c2a["seg"].items()},
            "branches": json.load(open(anno_p)),
        }
        labels["n_branches"] = len(labels["branches"])
        json.dump(labels, open(os.path.join(res, "labels.json"), "w"), indent=1)
        cat = pal["categorical"]
        lob_col = {v: pal["lobar"].get(n, cat[i % len(cat)])
                   for i, (v, n) in enumerate(labels["lobar"].items())}
        seg_col = {v: cat[i % len(cat)] for i, v in enumerate(labels["segmental"])}
        itksnap_labels(os.path.join(res, "itksnap_lobar_labels.txt"), labels["lobar"], lob_col)
        itksnap_labels(os.path.join(res, "itksnap_segmental_labels.txt"), labels["segmental"], seg_col)

    if "airway_lobar.nii.gz" in src_imgs:
        lob = sitk.GetArrayFromImage(src_imgs["airway_lobar.nii.gz"])
        names = labels.get("lobar", {})
        vals, counts = np.unique(lob[lob > 0], return_counts=True)
        stats["lobes"] = [{"label": int(v), "name": names.get(str(int(v)), str(int(v))),
                           "voxels": int(c)} for v, c in zip(vals, counts)]
    stats["branches"] = labels.get("n_branches")
    stats["source_orientation"] = code

    # ---- display copies -------------------------------------------------
    ct = sitk.ReadImage(os.path.join(res, "image.nii.gz"))
    s, size = view_grid(ct, C.VIEW_MAX_VOXELS)
    stats["view_spacing_mm"] = s
    stats["view_size"] = size
    print(f"display grid {size} at {s} mm (source {list(ct.GetSize())} at "
          f"{[round(x, 3) for x in ct.GetSpacing()]})", flush=True)
    sitk.WriteImage(sitk.Cast(resample(ct, s, size, nearest=False), sitk.sitkInt16),
                    os.path.join(res, "image_view.nii.gz"), True)
    for out, img in src_imgs.items():
        v = resample(img, s, size, nearest=True)
        sitk.WriteImage(v, os.path.join(res, out.replace(".nii.gz", "_view.nii.gz")), True)
    print("wrote display copies", flush=True)

    # ---- meshes -----------------------------------------------------------
    meshes = []
    if "airway_mask.nii.gz" in src_imgs:
        img = src_imgs["airway_mask.nii.gz"]
        m = mesh_from_mask(sitk.GetArrayFromImage(img) > 0, img)
        if m is not None:
            write_mz3(os.path.join(res, "mesh_airway.mz3"), *m)
            meshes.append({"file": "mesh_airway.mz3", "kind": "airway", "faces": int(len(m[1]))})
    if "airway_lobar.nii.gz" in src_imgs:
        img = src_imgs["airway_lobar.nii.gz"]
        lob = sitk.GetArrayFromImage(img)
        for v in np.unique(lob[lob > 0]):
            m = mesh_from_mask(lob == v, img)
            if m is None:
                continue
            fn = f"mesh_lobe_{int(v)}.mz3"
            write_mz3(os.path.join(res, fn), *m)
            meshes.append({"file": fn, "kind": "lobe", "label": int(v),
                           "name": labels.get("lobar", {}).get(str(int(v)), str(int(v))),
                           "faces": int(len(m[1]))})
    stats["meshes"] = meshes
    print(f"wrote {len(meshes)} meshes", flush=True)

    json.dump(stats, open(os.path.join(res, "summary.json"), "w"), indent=1)

    zp = os.path.join(res, "airway_results.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:   # nii.gz is already compressed
        for f in sorted(os.listdir(res)):
            if f != "airway_results.zip" and not f.endswith("_view.nii.gz"):
                z.write(os.path.join(res, f), f)
    print(f"packaged: {json.dumps({k: v for k, v in stats.items() if k != 'meshes'})}", flush=True)


if __name__ == "__main__":
    main()
