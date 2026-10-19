"""Turn an uploaded CT in any supported format into the job's input.nii.gz.

The worker reads one NIfTI (input.nii.gz). NIfTI uploads are stored as they are
(a plain .nii is gzipped); MetaImage (.mha), NRRD (.nrrd) and a zipped DICOM
series (.zip) are read with SimpleITK and written as NIfTI with the same
geometry, so results still overlay the scan the user uploaded. DICOM rescale
slope/intercept are applied by the reader, so the NIfTI holds Hounsfield units.
"""
import gzip, os, shutil, tempfile, zipfile

# extension -> kind; .nii.gz must be tested before .gz-less endings
FORMATS = ((".nii.gz", "nifti"), (".nii", "nifti"), (".mha", "image"), (".nrrd", "image"), (".zip", "dicom-zip"))
ACCEPTED = ", ".join(e for e, _ in FORMATS)


def kind_of(name):
    low = name.lower()
    for ext, kind in FORMATS:
        if low.endswith(ext):
            return ext, kind
    return None, None


def _largest_dicom_series(root):
    import SimpleITK as sitk
    best = None
    for d, _, files in os.walk(root):
        if not files:
            continue
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(d) or ():
            names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(d, sid)
            if best is None or len(names) > len(best):
                best = names
    return best


def to_nifti(src, kind, dst):
    """Write `dst` (input.nii.gz) from the uploaded file `src`. Raises ValueError with a
    message for the user when the file cannot be read as a single 3D CT volume."""
    if kind == "nifti":
        if src.lower().endswith(".nii"):
            with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=1) as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(src)
        elif src != dst:
            os.replace(src, dst)
        return
    import SimpleITK as sitk
    try:
        if kind == "image":
            img = sitk.ReadImage(src)
        else:
            tmp = tempfile.mkdtemp(prefix="dicom_", dir=os.path.dirname(dst))
            try:
                with zipfile.ZipFile(src) as z:
                    for m in z.infolist():                       # no absolute paths or ".."
                        if m.is_dir() or os.path.isabs(m.filename) or ".." in m.filename.split("/"):
                            continue
                        z.extract(m, tmp)
                names = _largest_dicom_series(tmp)
                if not names:
                    raise ValueError("no DICOM series found in the .zip")
                if len(names) < 10:
                    raise ValueError(f"the largest DICOM series has only {len(names)} slices")
                r = sitk.ImageSeriesReader(); r.SetFileNames(names)
                img = r.Execute()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    except ValueError:
        raise
    except Exception as e:                                         # noqa: BLE001
        raise ValueError(f"could not read the file as a CT volume ({str(e).splitlines()[0][:160]})")
    if img.GetDimension() != 3 or img.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("expected a single 3D grey-value volume")
    sitk.WriteImage(img, dst, True)
    os.remove(src)
