#!/usr/bin/env python3
"""Remove the tube-contaminated superior mask from sub-0534_acc-0193."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


CASE = Path("/fileserver/external/quin/ege/airway-seg/age_01-02/sub-0534/sub-0534_acc-0193")
CASE_ID = CASE.name
CUTOFF_Z = 518  # first tube-affected axial region begins immediately above this plane
BACKUP = CASE / "tube_correction_backup"


def main() -> None:
    mask_path = CASE / "union3_lcc.nii.gz"
    backup_files = [
        "union3_lcc.nii.gz", "union3_lcc_srcorient.nii.gz",
        "union3_lcc_skeleton.nii.gz", f"{CASE_ID}_skel_parsing.nii.gz",
        f"{CASE_ID}_parse.nii.gz", f"{CASE_ID}_pred_lob.nii.gz",
        f"{CASE_ID}_pred_seg.nii.gz", f"{CASE_ID}_pred_sub.nii.gz",
        f"{CASE_ID}_airway_graph.npy", f"{CASE_ID}_airway_feature_cls.npy",
        f"{CASE_ID}_airway_graph_cls.npy", f"{CASE_ID}_anno.json",
    ]
    BACKUP.mkdir(exist_ok=True)
    for name in backup_files:
        source = CASE / name
        destination = BACKUP / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)

    image = sitk.ReadImage(str(mask_path))
    original = sitk.GetArrayFromImage(image) > 0
    if CUTOFF_Z >= original.shape[0]:
        raise ValueError(f"cutoff {CUTOFF_Z} outside z size {original.shape[0]}")
    corrected = original.copy()
    removed = int(np.count_nonzero(corrected[CUTOFF_Z:]))
    corrected[CUTOFF_Z:] = False
    labels, n_components = ndimage.label(corrected, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    corrected = labels == largest
    retained = int(np.count_nonzero(corrected))
    if retained == 0 or n_components < 1:
        raise RuntimeError("tube correction produced an empty mask")

    output = sitk.GetImageFromArray(corrected.astype(np.uint8))
    output.CopyInformation(image)
    temporary = CASE / ".tube_corrected.union3_lcc.nii.gz"
    sitk.WriteImage(output, str(temporary), True)
    os.replace(temporary, mask_path)

    orientation = json.loads((CASE / "orient.json").read_text())["source_orientation"]
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(orientation)
    source_frame = orienter.Execute(output)
    temporary_src = CASE / ".tube_corrected.union3_lcc_srcorient.nii.gz"
    sitk.WriteImage(source_frame, str(temporary_src), True)
    os.replace(temporary_src, CASE / "union3_lcc_srcorient.nii.gz")

    provenance = {
        "case_id": CASE_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "superior tracheal medical tube produced annular mask and false skeleton branches",
        "working_frame": "LPS",
        "cutoff_index_z": CUTOFF_Z,
        "retained_index_z": f"0-{CUTOFF_Z - 1}",
        "removed_index_z": f"{CUTOFF_Z}-{original.shape[0] - 1}",
        "cutoff_physical_z_mm": image.TransformIndexToPhysicalPoint((0, 0, CUTOFF_Z))[2],
        "original_voxels": int(original.sum()),
        "removed_superior_voxels": removed,
        "corrected_largest_component_voxels": retained,
        "components_before_largest_component_filter": int(n_components),
        "backup_directory": str(BACKUP),
        "review_status": "tube correction; radiologist confirmation requested",
    }
    (CASE / "tube_correction.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
