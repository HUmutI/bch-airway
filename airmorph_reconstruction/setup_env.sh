#!/usr/bin/env bash
# Additive dependency install for AirMorph binary-airway-segmentation training.
#
# Targets the existing `clip` conda env, which already provides a compatible base
# (torch 2.6.0+cu118, numpy 1.26.4, scipy 1.15.3, scikit-image 0.25.2).
#
# NOTE: we deliberately do NOT run `pip install -r requirements.txt`. Those pins would
# downgrade packages that other work in `clip` depends on. Only the packages actually
# needed by the segmentation-training scope are installed here.
#
# The anatomical-labeling stage IS now in scope, so pycuda + torch_geometric are also
# installed (in a separate, non-fatal step so a missing nvcc doesn't block seg setup).
# Still skipped (only used by the radiomics / max-flow signature stages, out of scope):
#   pyradiomics, PyMaxflow
set -euo pipefail

ENV_PYTHON="${ENV_PYTHON:-/path/to/envs/conda-envs/clip/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Using interpreter: ${ENV_PYTHON}"
"${ENV_PYTHON}" -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo
echo "==> Installing segmentation-training dependencies"
"${ENV_PYTHON}" -m pip install --no-input \
    "monai==1.4.0" \
    "nibabel==5.3.2" \
    "SimpleITK==2.4.1" \
    "kimimaro==5.0.0" \
    "connected-components-3d==3.23.0" \
    "edt==3.0.0" \
    "fill_voids==2.0.8" \
    "fastremap==1.15.1" \
    "dijkstra3d==1.15.1" \
    "osteoid==0.3.1" \
    "xs3d==1.7.2" \
    "lungmask==0.2.20"

echo
echo "==> Installing anatomical-labeling dependencies (pycuda + torch_geometric)"
# pycuda compiles the graph-build CUDA kernels at runtime, so it needs a working nvcc
# (CUDA toolkit) on PATH. If this step fails, segmentation training still works; only
# the labeling stage is blocked until pycuda builds. torch_geometric supplies only the
# lightweight `Data` container the released dataset/labeling code uses (the TNN itself is
# pure attention, no message-passing layers).
if ! "${ENV_PYTHON}" -m pip install --no-input "pycuda==2024.1.2" "torch_geometric==2.6.1"; then
    echo "    [warn] pycuda / torch_geometric install failed (nvcc missing?)."
    echo "    [warn] Segmentation training is unaffected; labeling needs these — fix nvcc and re-run."
fi

# nvcc host-compiler note: CUDA 12.3's nvcc rejects host gcc > 12. This box ships gcc 13
# with no g++-12, so pycuda kernel JIT needs `-allow-unsupported-compiler`.
# training/data/prepare_labeling.py sets NVCC_PREPEND_FLAGS itself, but the PROVIDED
# inference pipeline (airwayatlas_pipeline.py, classifier/) does not — so for any pycuda-using
# provided code, export it first:
#     export NVCC_PREPEND_FLAGS='-allow-unsupported-compiler'
# (or install g++-12 and use `-ccbin g++-12`, which avoids the version override entirely).

echo
echo "==> Fetching the public lungmask R231 weight"
# networks/lungmask_network.py:635 hardcodes this local path with NO url fallback,
# so lung masking fails unless the file is present.
CKPT_DIR="${PROJECT_ROOT}/checkpoints"
CKPT_PATH="${CKPT_DIR}/unet_r231-d5d2fc3d.pth"
mkdir -p "${CKPT_DIR}"
if [[ -f "${CKPT_PATH}" ]]; then
    echo "    already present: ${CKPT_PATH}"
else
    curl -fL --progress-bar \
        -o "${CKPT_PATH}" \
        "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth"
    echo "    saved: ${CKPT_PATH}"
fi

echo
echo "==> Verifying imports"
"${ENV_PYTHON}" - <<'PY'
import importlib

required = [
    "monai", "nibabel", "SimpleITK", "kimimaro",
    "cc3d", "edt", "fill_voids", "fastremap", "lungmask",
]
missing = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"    {name:16s} OK   {getattr(module, '__version__', '?')}")
    except Exception as exc:
        missing.append(name)
        print(f"    {name:16s} FAIL {type(exc).__name__}: {exc}")

if missing:
    raise SystemExit(f"missing packages: {missing}")
print("\nAll segmentation-training dependencies are importable.")

# Labeling deps are checked but not required (segmentation can proceed without them).
print("\n== labeling stage (optional) ==")
for name in ["pycuda", "torch_geometric"]:
    try:
        module = importlib.import_module(name)
        print(f"    {name:16s} OK   {getattr(module, '__version__', '?')}")
    except Exception as exc:
        print(f"    {name:16s} MISSING {type(exc).__name__}: {exc}")
        print(f"    -> anatomical labeling training needs {name}; segmentation is fine.")
PY

echo
echo "Setup complete."
