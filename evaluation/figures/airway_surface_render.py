"""Depth-shaded anterior surface rendering of a binary airway mask.

No VTK in this environment, so the isosurface is raycast directly: the mask is
resampled onto an isotropic grid, smoothed into a scalar field, and the first
crossing of the 0.5 level along the anterior-posterior axis gives a depth and a
surface normal per pixel. Lambert plus a specular term then shades it.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation
from scipy import ndimage

RENDER_MM = 0.4          # isotropic voxel size of the render grid
SMOOTH_VOX = 1.1         # gaussian sigma, in render voxels
LEVEL = 0.5


def load_lps(path):
    """Return the volume resampled to LPS axis order, with its LPS spacing."""
    image = nib.load(str(path))
    start = io_orientation(image.affine)
    target = axcodes2ornt(("L", "P", "S"))
    transform = ornt_transform(start, target)
    data = apply_orientation(np.asanyarray(image.dataobj), transform)
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
    spacing = zooms[transform[:, 0].astype(int)]
    return data, spacing


def prepare(mask_path, lung_path=None, margin_mm=4.0):
    """Crop the airway to its bounding box and resample to an isotropic grid.

    When a lung mask is given the airway is clipped at the most superior
    segmented lung voxel, the thoracic-inlet proxy this cohort already uses, so
    panels are not made to differ by how much cervical trachea each scan covers.
    """
    mask, spacing = load_lps(mask_path)
    mask = mask > 0
    if lung_path is not None:
        lung, lung_spacing = load_lps(lung_path)
        lung = lung > 0
        if lung.shape == mask.shape and lung.any():
            inlet = np.where(lung.any(axis=(0, 1)))[0].max()
            mask[:, :, inlet + 1:] = False
    if not mask.any():
        raise ValueError(f"empty mask: {mask_path}")

    pad = np.ceil(margin_mm / spacing).astype(int)
    slices = []
    for axis, extent in enumerate(mask.shape):
        present = np.where(mask.any(axis=tuple(i for i in range(3) if i != axis)))[0]
        low = max(int(present.min()) - pad[axis], 0)
        high = min(int(present.max()) + pad[axis] + 1, extent)
        slices.append(slice(low, high))
    cropped = mask[tuple(slices)].astype(np.float32)

    field = ndimage.zoom(cropped, spacing / RENDER_MM, order=1, prefilter=False)
    field = ndimage.gaussian_filter(field, SMOOTH_VOX, output=np.float32)
    return field


def shade(field, light=(-0.45, -1.0, 0.55), ambient=0.30, specular=0.22, shininess=22.0):
    """Raycast along +y (anterior camera) and return RGB and coverage images.

    Images are indexed (row, column) with superior at the top and the patient's
    left on the viewer's right, the orientation of a conventional frontal view.
    """
    inside = field > LEVEL
    hit = inside.any(axis=1)
    depth = inside.argmax(axis=1)

    index = depth[:, None, :]
    normal = np.empty(hit.shape + (3,), dtype=np.float32)
    for axis in range(3):
        gradient = np.gradient(field, RENDER_MM, axis=axis)
        normal[..., axis] = np.take_along_axis(gradient, index, axis=1)[:, 0, :]
    del gradient
    normal *= -1.0                                  # field decreases outward
    length = np.linalg.norm(normal, axis=-1)
    normal /= np.where(length > 1e-8, length, 1.0)[..., None]

    light = np.asarray(light, dtype=np.float32)
    light /= np.linalg.norm(light)
    view = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    half = light + view
    half /= np.linalg.norm(half)

    diffuse = np.clip(normal @ light, 0.0, 1.0)
    highlight = np.clip(normal @ half, 0.0, 1.0) ** shininess
    intensity = ambient + (1.0 - ambient - specular) * diffuse + specular * highlight

    # Aerial perspective: structures further from the camera fade back slightly.
    span = depth[hit].ptp() if hit.any() else 1
    distance = np.zeros_like(intensity)
    if span > 0:
        distance = np.clip((depth - depth[hit].min()) / span, 0.0, 1.0)
    intensity *= 1.0 - 0.28 * distance

    dark = np.array([0.055, 0.145, 0.305], dtype=np.float32)
    mid = np.array([0.204, 0.478, 0.761], dtype=np.float32)
    bright = np.array([0.839, 0.925, 0.996], dtype=np.float32)
    t = np.clip(intensity, 0.0, 1.0)[..., None]
    rgb = np.where(t < 0.55, dark + (mid - dark) * (t / 0.55),
                   mid + (bright - mid) * ((t - 0.55) / 0.45))
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    rgb = np.flip(rgb.transpose(1, 0, 2), axis=0)         # (x, z, rgb) -> rows superior-first
    alpha = np.flip(hit.T.astype(np.float32), axis=0)
    return rgb, alpha


def prepare_labeled(mask_path, parse_path, generation_of, lung_path=None, margin_mm=4.0):
    """Same geometry as `prepare`, plus a per-voxel generation on the render grid.

    `{case}_parse.nii.gz` labels every airway voxel with its branch instance and
    covers union3_lcc exactly, so generation can be painted straight onto the
    surface. Labels are carried to the render grid nearest-neighbour and then
    filled outwards, because the smoothed isosurface sits slightly proud of the
    original voxels and would otherwise expose unlabelled shell.

    Returns the scalar field, the generation grid, and each branch instance's
    centroid in render-grid coordinates, for placing callouts.
    """
    mask, spacing = load_lps(mask_path)
    mask = mask > 0
    parse, _ = load_lps(parse_path)
    if parse.shape != mask.shape:
        raise ValueError("parse and airway masks differ in shape")
    if lung_path is not None:
        lung, _ = load_lps(lung_path)
        lung = lung > 0
        if lung.shape == mask.shape and lung.any():
            inlet = np.where(lung.any(axis=(0, 1)))[0].max()
            mask[:, :, inlet + 1:] = False
    if not mask.any():
        raise ValueError(f"empty mask: {mask_path}")

    pad = np.ceil(margin_mm / spacing).astype(int)
    slices = []
    for axis, extent in enumerate(mask.shape):
        present = np.where(mask.any(axis=tuple(i for i in range(3) if i != axis)))[0]
        low = max(int(present.min()) - pad[axis], 0)
        high = min(int(present.max()) + pad[axis] + 1, extent)
        slices.append(slice(low, high))
    slices = tuple(slices)
    cropped = mask[slices].astype(np.float32)
    instances = np.where(mask[slices], parse[slices], 0)

    zoom = spacing / RENDER_MM
    field = ndimage.zoom(cropped, zoom, order=1, prefilter=False)
    field = ndimage.gaussian_filter(field, SMOOTH_VOX, output=np.float32)

    generation = np.zeros(instances.shape, dtype=np.int16)
    for instance, value in generation_of.items():
        generation[instances == instance] = value
    generation = ndimage.zoom(generation, zoom, order=0, prefilter=False)
    if generation.shape != field.shape:                      # rounding can differ by a voxel
        trimmed = tuple(slice(0, min(a, b)) for a, b in zip(generation.shape, field.shape))
        grown = np.zeros(field.shape, dtype=np.int16)
        grown[trimmed] = generation[trimmed]
        generation = grown
    _, nearest = ndimage.distance_transform_edt(generation == 0, return_indices=True)
    generation = generation[tuple(nearest)]

    present = np.unique(instances)
    present = present[present > 0]
    positions = ndimage.center_of_mass(instances > 0, labels=instances, index=present)
    centroids = {int(instance): np.asarray(position) * zoom
                 for instance, position in zip(present, positions)}
    return field, generation, centroids


def shade_labeled(field, generation, palette, ambient=0.34, specular=0.20, shininess=22.0,
                  light=(-0.45, -1.0, 0.55)):
    """Raycast as in `shade`, but tint each pixel by the generation it lands on."""
    inside = field > LEVEL
    hit = inside.any(axis=1)
    depth = inside.argmax(axis=1)
    index = depth[:, None, :]

    normal = np.empty(hit.shape + (3,), dtype=np.float32)
    for axis in range(3):
        component = np.gradient(field, RENDER_MM, axis=axis)
        normal[..., axis] = np.take_along_axis(component, index, axis=1)[:, 0, :]
    del component
    normal *= -1.0
    length = np.linalg.norm(normal, axis=-1)
    normal /= np.where(length > 1e-8, length, 1.0)[..., None]

    light = np.asarray(light, dtype=np.float32)
    light /= np.linalg.norm(light)
    view = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    half = light + view
    half /= np.linalg.norm(half)

    diffuse = np.clip(normal @ light, 0.0, 1.0)
    highlight = np.clip(normal @ half, 0.0, 1.0) ** shininess
    intensity = ambient + (1.0 - ambient) * diffuse

    surface_generation = np.take_along_axis(generation, index, axis=1)[:, 0, :]
    base = palette(surface_generation)
    rgb = np.clip(base * intensity[..., None] + specular * highlight[..., None], 0.0, 1.0)

    rgb = np.flip(rgb.transpose(1, 0, 2), axis=0).astype(np.float32)
    alpha = np.flip(hit.T.astype(np.float32), axis=0)
    labels = np.flip(surface_generation.T, axis=0)
    return rgb, alpha, labels
