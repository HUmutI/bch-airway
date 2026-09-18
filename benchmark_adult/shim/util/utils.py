"""Reimplementation of the two util.utils helpers the scorer needs.

load_itk_image        : read an ITK volume as (array, origin, spacing, direction)
restore_image_via_box : paste a bbox-cropped array back into its full-size frame
"""
import numpy as np
import SimpleITK as sitk


def load_itk_image(path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return (array,
            np.array(image.GetOrigin())[::-1],
            np.array(image.GetSpacing())[::-1],
            np.array(image.GetDirection()))


def restore_image_via_box(original_shape, cropped, bbox):
    full = np.zeros(original_shape, dtype=cropped.dtype)
    (z0, z1), (y0, y1), (x0, x1) = bbox
    full[z0:z1, y0:y1, x0:x1] = cropped
    return full
