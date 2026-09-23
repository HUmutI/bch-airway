#!/usr/bin/env python3
"""Accuracy checks for the cross-sectional measurement code.

The tests build elliptical cylinders with known axes, sample them exactly the
way a real case is sampled, and confirm that the recovered short and long axes
match the truth. A tilted cylinder is included because a plane taken along the
voxel grid rather than along the centerline would overestimate both axes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import axis_geometry as ag


def elliptical_cylinder(
    short_mm: float, long_mm: float, spacing_mm: float, tilt_degrees: float
) -> ag.CaseVolumes:
    """Build a synthetic case containing one tilted elliptical cylinder."""
    size = 160
    grid = (np.arange(size) - size / 2.0) * spacing_mm
    z, y, x = np.meshgrid(grid, grid, grid, indexing="ij")

    angle = np.deg2rad(tilt_degrees)
    # Rotate about the x axis so the cylinder axis leaves the z direction.
    y_rotated = y * np.cos(angle) - z * np.sin(angle)
    z_rotated = y * np.sin(angle) + z * np.cos(angle)
    mask = ((x / (0.5 * long_mm)) ** 2 + (y_rotated / (0.5 * short_mm)) ** 2) <= 1.0
    mask &= np.abs(z_rotated) <= 25.0

    identity = np.eye(3) * spacing_mm
    return ag.CaseVolumes(
        case_id="phantom",
        parse=np.zeros_like(mask, dtype=np.int16),
        mask=mask,
        spacing_zyx=np.full(3, spacing_mm),
        index_to_physical=identity,
        physical_to_index=np.linalg.inv(identity),
        origin=np.full(3, -size / 2.0 * spacing_mm),
        lung_apex_z_mm=0.0,
        total_lung_volume_mL=1.0,
    )


def check(short_mm: float, long_mm: float, spacing_mm: float, tilt_degrees: float) -> None:
    volumes = elliptical_cylinder(short_mm, long_mm, spacing_mm, tilt_degrees)
    angle = np.deg2rad(tilt_degrees)
    axis = np.array([0.0, np.sin(angle), np.cos(angle)])
    measured = ag.cross_section(
        volumes, np.zeros(3), axis, ag.TRACHEA_HALF_EXTENT_MM
    )
    assert measured is not None, "phantom cross-section could not be measured"

    short_error = measured["short_axis_mm"] - short_mm
    long_error = measured["long_axis_mm"] - long_mm
    mean_error = measured["mean_axis_diameter_mm"] - 0.5 * (short_mm + long_mm)
    print(
        f"short {short_mm:5.1f} long {long_mm:5.1f} spacing {spacing_mm:4.2f} "
        f"tilt {tilt_degrees:4.1f} deg -> "
        f"short {measured['short_axis_mm']:6.2f} ({short_error:+.2f}) "
        f"long {measured['long_axis_mm']:6.2f} ({long_error:+.2f}) "
        f"mean err {mean_error:+.2f} mm"
    )
    tolerance = max(0.6, 1.2 * spacing_mm)
    assert abs(short_error) < tolerance, f"short axis error {short_error:.2f} mm"
    assert abs(long_error) < tolerance, f"long axis error {long_error:.2f} mm"


def check_eccentricity_direction() -> None:
    """A flattened lumen must report a short axis below its long axis."""
    volumes = elliptical_cylinder(6.0, 12.0, 0.4, 0.0)
    measured = ag.cross_section(volumes, np.zeros(3), np.array([0.0, 0.0, 1.0]), 28.0)
    assert measured["short_axis_mm"] < measured["long_axis_mm"]
    assert 0.0 < measured["eccentricity_ratio"] < 1.0
    print(f"eccentricity ratio {measured['eccentricity_ratio']:.3f} for a 6 x 12 mm lumen")


def check_edt_underestimates_mean_axis() -> None:
    """Confirm the claim that twice the inscribed radius tracks the short axis."""
    short_mm, long_mm, spacing_mm = 8.0, 16.0, 0.4
    volumes = elliptical_cylinder(short_mm, long_mm, spacing_mm, 0.0)
    measured = ag.cross_section(volumes, np.zeros(3), np.array([0.0, 0.0, 1.0]), 28.0)

    from scipy import ndimage

    distance = ndimage.distance_transform_edt(volumes.mask, sampling=volumes.spacing_zyx)
    center = tuple(dimension // 2 for dimension in volumes.mask.shape)
    edt_diameter = 2.0 * float(distance[center])

    print(
        f"inscribed-sphere diameter {edt_diameter:.2f} mm vs short axis "
        f"{measured['short_axis_mm']:.2f} mm vs mean axis "
        f"{measured['mean_axis_diameter_mm']:.2f} mm"
    )
    assert abs(edt_diameter - short_mm) < 0.6, "EDT diameter should track the short axis"
    assert measured["mean_axis_diameter_mm"] > edt_diameter + 1.0


def main() -> None:
    for short_mm, long_mm in ((5.0, 5.0), (6.0, 9.0), (10.0, 14.0), (14.0, 18.0)):
        for spacing_mm in (0.3, 0.5):
            for tilt_degrees in (0.0, 20.0):
                check(short_mm, long_mm, spacing_mm, tilt_degrees)
    check_eccentricity_direction()
    check_edt_underestimates_mean_axis()
    print("all cross-section accuracy checks passed")


if __name__ == "__main__":
    main()
