#!/bin/bash
# Stage 0 - one-time setup: extract the three pipelines, build three venvs,
# compile TfeNet's CUDA extensions, and verify all three import.
#
# Needs the GPU visible (nvidia-smi must work) because the TfeNet verification
# step actually initialises CUDA. Run OUTSIDE the bwrap sandbox.
#
#   ./00_setup.sh
#
# Safe to re-run; each step is skipped if it already succeeded.
. "$(dirname "$(readlink -f "$0")")/config.sh"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 0. preconditions -------------------------------------------------------
log "checking GPU"
if ! nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<'EOF'
ERROR: nvidia-smi cannot talk to the driver.

If you are inside claude-safe.sh, that sandbox uses `--dev /dev`, which creates
a minimal /dev with no /dev/nvidia* nodes. Either run this script outside the
sandbox, or change claude-safe.sh line 22 to:

    --dev-bind /dev /dev \

TfeNet's DAConv/DSConv raise NotImplementedError on non-CUDA input
(DAConv/DAConv_CUDA.py:27), so there is no CPU route through this pipeline.
EOF
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

log "checking nvcc"
nvcc --version | tail -2

# --- 1. verify + extract tarballs -------------------------------------------
cd "$OR3_TARBALLS"
log "verifying tarball checksums"
check_md5() {
    local f="$1" want="$2" got
    got=$(md5sum "$f" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        echo "ERROR: $f md5 $got != expected $want" >&2; exit 1
    fi
    echo "  ok  $f"
}
check_md5 airmorph_seg.tar.gz   8ed1d39a68233f25f93eb1690693bdf3
check_md5 medpseg_airway.tar.gz abd5fd0efb9c68a01d90ecb04c27fd16
check_md5 tfenet_bas.tar.gz     929f4ce36d1268ec9523d3ec8aa9ec3e

cd "$OR3_CODE"
[ -d AirMorph ] || { log "extracting AirMorph"; tar xzf "$OR3_TARBALLS/airmorph_seg.tar.gz"; }
[ -d medpseg  ] || { log "extracting MEDPSeg";  tar xzf "$OR3_TARBALLS/medpseg_airway.tar.gz"; }
[ -d TfeNet   ] || { log "extracting TfeNet";   tar xzf "$OR3_TARBALLS/tfenet_bas.tar.gz"; }
ls -d AirMorph medpseg TfeNet

# --- 2. AirMorph env --------------------------------------------------------
# Guard on a completion marker, NOT on bin/python: `venv` creates the
# interpreter before any package is installed, so an install interrupted
# part-way would otherwise be silently skipped on the next run, leaving a
# half-built env that fails much later.
if [ ! -f "$OR3_ENVS/airmorph/.complete" ]; then
    log "creating airmorph venv"
    python3 -m venv "$OR3_ENVS/airmorph"
    "$OR3_ENVS/airmorph/bin/pip" install -q --upgrade pip wheel
    # cu121 to match this requirements.txt's pinned nvidia-* 12.1 libs - see the
    # long note in config.sh.
    "$OR3_ENVS/airmorph/bin/pip" install -q "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_CU121"
    # Two packages are stripped because neither is on the import path of
    # segmentator/airway_segmentator.py. The full transitive third-party set
    # reachable from it is: SimpleITK, fill_voids, kimimaro, monai,
    # more_itertools, numpy, pydicom, scipy, skimage, torch, tqdm.
    #
    #   pycuda      - branch labelling only, not segmentation. LOCAL_COMMANDS.md:81
    #   pyradiomics - imported only by features/airway_radiomic_features.py.
    #                 It also cannot install here: 3.0.1 has no cp310 wheel, so
    #                 pip builds the sdist, and its setup.py imports numpy at
    #                 module level without declaring it in build-requires, which
    #                 fails under PEP 517 build isolation.
    grep -vE '^(pycuda|pyradiomics)([=<>!~[:space:]]|$)' \
        "$OR3_CODE/AirMorph/requirements.txt" > /tmp/req_airmorph.txt
    echo "  stripped: $(grep -cE '^(pycuda|pyradiomics)' "$OR3_CODE/AirMorph/requirements.txt") package(s)"

    # numpy and Cython up front: several of the older sdists pinned here import
    # numpy during their own build.
    "$OR3_ENVS/airmorph/bin/pip" install -q numpy==1.26.4 Cython setuptools wheel
    if ! "$OR3_ENVS/airmorph/bin/pip" install -q -r /tmp/req_airmorph.txt; then
        log "retrying AirMorph requirements without build isolation"
        "$OR3_ENVS/airmorph/bin/pip" install -q -r /tmp/req_airmorph.txt \
            --no-build-isolation
    fi
    # Nothing above may have disturbed torch's own CUDA stack.
    "$OR3_ENVS/airmorph/bin/pip" check || echo "  WARNING: dependency conflicts above"
    touch "$OR3_ENVS/airmorph/.complete"
fi
log "verifying AirMorph"
"$OR3_ENVS/airmorph/bin/python" -c "
import sys; sys.path.insert(0, '$OR3_CODE/AirMorph')
import torch
assert torch.cuda.is_available(), 'torch cannot see the GPU'
# Actually exercise cuBLAS: a mismatched nvidia-* stack imports fine and only
# fails here, with an undefined symbol.
torch.linalg.matmul(torch.randn(8, 8, device='cuda'), torch.randn(8, 8, device='cuda'))
from segmentator.airway_segmentator import AirwayAtlasBinaryAirwaySegmentator
import SimpleITK, scipy
print(f'  AirMorph OK - torch {torch.__version__}, cuBLAS live, SimpleITK + scipy present')"

# --- 3. MEDPSeg env ---------------------------------------------------------
is_lober_check() { grep -q '^check_weight(os.path.join("medpseg", "lober.ckpt"))' "$1"; }

# medpseg's setup.py verifies all three weights at INSTALL time, including
# lober.ckpt - the 523 MB lobes head deliberately excluded from the tarball
# because or_all3_lcc does not use it (LOCAL_COMMANDS.md:11-13). Comment out
# that one check.
#
# This defers an error rather than suppressing one: lober.ckpt is loaded only
# inside `if lobe_seg:` (medpseg/pipeline.py:134), we always pass
# --disable_lobe, and the runtime check_weight at pipeline.py:136 still guards
# the lobe path - so if anyone enables lobes later it fails loudly, exactly as
# upstream intended.
MP_SETUP="$OR3_CODE/medpseg/setup.py"
cp -n "$MP_SETUP" "$MP_SETUP.orig" 2>/dev/null || true
if is_lober_check "$MP_SETUP"; then
    sed -i 's|^check_weight(os.path.join("medpseg", "lober.ckpt"))|#\0  # absent by design: or_all3_lcc runs --disable_lobe|' "$MP_SETUP"
    log "patched medpseg/setup.py to skip the lober.ckpt install check (backup: setup.py.orig)"
fi

# SimpleITK 2.0.2 (2021) has no cp310 wheel - PyPI jumps 1.2.0 -> 2.1.0 - so the
# pin is unsatisfiable on this interpreter. MEDPSeg only touches the stable core
# API (ReadImage, WriteImage, GetArrayFromImage, GetImageFromArray, Image,
# LabelOverlapMeasures, HausdorffDistance), none of which changed across 2.x, so
# move it to the 2.4.1 the AirMorph env already uses.
if grep -q "'SimpleITK==2.0.2'" "$MP_SETUP"; then
    sed -i "s|'SimpleITK==2.0.2'|'SimpleITK==2.4.1'|" "$MP_SETUP"
    log "patched medpseg/setup.py: SimpleITK 2.0.2 -> 2.4.1 (no cp310 wheel for 2.0.2)"
fi

if [ ! -f "$OR3_ENVS/medpseg/.complete" ]; then
    log "creating medpseg venv"
    python3 -m venv "$OR3_ENVS/medpseg"
    "$OR3_ENVS/medpseg/bin/pip" install -q --upgrade pip wheel
    # torchvision from the SAME index as torch - see config.sh. Installing it
    # up front also stops medpseg's unpinned 'torchvision' requirement from
    # resolving to the PyPI (cu124) build over our cu121 torch.
    "$OR3_ENVS/medpseg/bin/pip" install -q \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        --index-url "$TORCH_INDEX_CU121"
    # MEDPSeg's setup.py leaves numpy unpinned, so pip takes numpy 2.x - but it
    # also pins connected-components-3d==3.12.6, whose wheels are compiled
    # against the NumPy 1.x C ABI. The result imports and then dies with
    # "numpy.core.multiarray failed to import" the first time cc3d is touched
    # (poly_pipeline.py:16). Constrain the whole resolution to numpy 1.x, the
    # same 1.26.4 the AirMorph env uses.
    echo "numpy<2" > /tmp/or3_medpseg_constraints.txt
    "$OR3_ENVS/medpseg/bin/pip" install -q "numpy==1.26.4"
    # NOT editable. medpseg/pipeline.py:111 hardcodes its weight directory as
    #     pkg_path = os.path.join(site.getsitepackages()[...], "medpseg")
    # so it only ever looks in site-packages. Under `pip install -e` the package
    # and its .ckpt files stay in code/medpseg/, site-packages/medpseg never
    # exists, and every case dies on a missing poly_medseg_25d_fix.ckpt.
    #
    # A regular install copies the weights in via setup.py's package_data, so
    # the hardcoded path resolves. (Symlinking a weights-only directory there
    # instead creates a competing namespace package that intermittently shadows
    # the editable finder - "cannot import name check_weight ... unknown
    # location". Don't.)
    "$OR3_ENVS/medpseg/bin/pip" install -q "$OR3_CODE/medpseg" \
        -c /tmp/or3_medpseg_constraints.txt
    "$OR3_ENVS/medpseg/bin/pip" check || echo "  WARNING: dependency conflicts above"
    touch "$OR3_ENVS/medpseg/.complete"
fi
# MEDPSeg's CLI entry point imports the GUI unconditionally (run.py:14 ->
# gui.py:20), so `medpseg` will not start without tkinter even in headless mode.
# tkinter ships with the OS python package, not with pip, so a venv cannot
# supply it.
#
# In CLI mode no widget is ever built: MainWindow sets self.cli when both
# folders are given (gui.py:114) and substitutes DummyTkIntVar for the GUI
# variables (gui.py:122-125). Only the module-level import needs satisfying.
if ! "$OR3_ENVS/medpseg/bin/python" -c "import tkinter" 2>/dev/null; then
    if [ "${OR3_TK_STUB:-0}" = "1" ]; then
        log "installing import-only tkinter stub into the medpseg venv"
        SP="$OR3_ENVS/medpseg/lib/python3.10/site-packages/tkinter"
        mkdir -p "$SP"
        cat > "$SP/__init__.py" <<'PY'
"""Import-only tkinter stub for headless MEDPSeg.

MEDPSeg imports tkinter at module scope (gui.py:20-27) but never constructs a
widget in CLI mode, where DummyTkIntVar stands in for the GUI variables
(gui.py:114-125). This satisfies the import without pulling in Tk.

Anything here that is actually CALLED warns loudly rather than pretending to
work, so if a GUI path is ever exercised it is obvious rather than silent.
"""
import warnings


class _Stub:
    def __init__(self, *a, **k):
        warnings.warn("tkinter stub instantiated - GUI code is running under a "
                      "headless stub; install python3-tk for real GUI support",
                      RuntimeWarning, stacklevel=2)

    def __call__(self, *a, **k):
        return _Stub()

    def __getattr__(self, name):
        return _Stub()


def __getattr__(name):          # PEP 562: satisfies `from tkinter import X`
    return _Stub()
PY
        for sub in ttk messagebox filedialog; do
            cp "$SP/__init__.py" "$SP/$sub.py"
        done
        echo "  stub installed (tkinter, tkinter.ttk, .messagebox, .filedialog)"
    else
        cat >&2 <<EOF

ERROR: python3 tkinter is missing, and MEDPSeg's CLI imports it at module
scope (run.py:14 -> gui.py:20) even with -i/-o given.

Preferred fix - install the OS package, then re-run this script:

    sudo apt install python3-tk

No root? Re-run with the import-only stub instead. CLI mode never builds a
widget, so this is safe for or_all3_lcc, and the stub warns loudly if any GUI
code path is ever hit:

    OR3_TK_STUB=1 ./00_setup.sh

EOF
        exit 1
    fi
fi

log "verifying MEDPSeg"
# Check the weights resolve where pipeline.py:111 will actually look. This is
# outside the venv guard on purpose: an env built by an older version of this
# script has an editable install and would otherwise stay silently broken.
"$OR3_ENVS/medpseg/bin/python" -c "
import os, site, sys
pkg = os.path.join(site.getsitepackages()[0], 'medpseg')
missing = [w for w in ('poly_medseg_25d_fix.ckpt', 'poly_lung.ckpt')
           if not os.path.exists(os.path.join(pkg, w))]
if missing:
    sys.exit('medpseg weights missing from ' + pkg + ': ' + ', '.join(missing) +
             '\n  Fix: rm -rf \$OR3_ENVS/medpseg and re-run 00_setup.sh '
             '(an editable install puts them out of reach of pipeline.py:111)')
print('  medpseg weights resolve at', pkg)"
"$OR3_ENVS/medpseg/bin/python" -c "
import numpy, torch, torchvision, SimpleITK
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} breaks cc3d'
# cc3d is compiled against the numpy 1.x C ABI and is what actually crashes on a
# mismatch, so import it here rather than discovering it mid-cohort.
import cc3d
assert torch.cuda.is_available(), 'torch cannot see the GPU'
torch.linalg.matmul(torch.randn(8, 8, device='cuda'), torch.randn(8, 8, device='cuda'))
# torchvision built against a different CUDA than torch imports fine and only
# fails when its compiled ops are touched.
from torchvision.transforms import Resize, InterpolationMode
Resize((4, 4), interpolation=InterpolationMode.BILINEAR)(torch.randn(1, 8, 8, device='cuda'))
print(f'  torch {torch.__version__} + torchvision {torchvision.__version__} agree on CUDA, '
      f'SimpleITK {SimpleITK.__version__}')"
# Bare command, NOT `cmd && echo ok`: under `set -e` the failing half of an
# AND-list is exempt from errexit, so the previous form printed the traceback
# and then carried on to the next env as though MEDPSeg were fine.
"$OR3_ENVS/medpseg/bin/medpseg" --help >/dev/null
echo "  MEDPSeg CLI OK"

# --- 4. TfeNet env + CUDA extensions ----------------------------------------
# TfeNet is the only pipeline that COMPILES anything, so its interpreter must
# ship the CPython development headers. The system python3.10 here does not
# (there is no Python.h anywhere under /usr/include) and python3-dev needs root.
#
# So when headers are missing, build THIS ONE env from a conda-forge python via
# micromamba - a single static binary that unpacks entirely into userspace.
# conda pythons include their headers. LOCAL_COMMANDS.md used conda for all
# three envs anyway, so this is closer to the original recipe, not further from
# it. AirMorph and MEDPSeg compile nothing and stay on plain venvs.
#
# Both venv and micromamba envs expose bin/python and bin/pip, so everything
# downstream - including config.sh's py_tfenet() - is unchanged either way.
SYS_PY_H="$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["include"], "Python.h"))')"
if [ ! -f "$OR3_ENVS/tfenet/.complete" ]; then
    if [ -f "$SYS_PY_H" ]; then
        log "creating tfenet venv (system python provides headers)"
        python3 -m venv "$OR3_ENVS/tfenet"
    else
        log "system python has no Python.h - building tfenet env via micromamba"
        MAMBA_DIR="$OR3_RUN/micromamba"
        MAMBA_BIN="$MAMBA_DIR/bin/micromamba"
        if [ ! -x "$MAMBA_BIN" ]; then
            mkdir -p "$MAMBA_DIR"
            curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
                | tar -xj -C "$MAMBA_DIR" bin/micromamba
            echo "  micromamba bootstrapped at $MAMBA_BIN"
        fi
        export MAMBA_ROOT_PREFIX="$MAMBA_DIR"
        "$MAMBA_BIN" create -y -q -p "$OR3_ENVS/tfenet" -c conda-forge \
            python=3.10 pip
    fi

    # Whatever produced it, this interpreter must have headers or the extension
    # build fails later with "Python.h: No such file or directory".
    TF_PY_H="$("$OR3_ENVS/tfenet/bin/python" -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["include"], "Python.h"))')"
    if [ ! -f "$TF_PY_H" ]; then
        echo "ERROR: tfenet interpreter still lacks Python.h at $TF_PY_H" >&2
        exit 1
    fi
    echo "  headers OK: $TF_PY_H"
    # setuptools explicitly: --no-build-isolation below means the extension
    # build uses THIS env's build tools rather than a provisioned isolated set.
    # ninja because torch's BuildExtension otherwise warns it "could not find
    # ninja" and falls back to the serial distutils backend, which compiles
    # these kernels far more slowly.
    "$OR3_ENVS/tfenet/bin/pip" install -q --upgrade pip wheel setuptools ninja
    # cu124 here so torch's CUDA matches the nvcc that compiles DAConv/DSConv.
    "$OR3_ENVS/tfenet/bin/pip" install -q "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_CU124"
    "$OR3_ENVS/tfenet/bin/pip" install -q -r "$OR3_CODE/TfeNet/requirements.txt"
    "$OR3_ENVS/tfenet/bin/pip" check || echo "  WARNING: dependency conflicts above"
    touch "$OR3_ENVS/tfenet/.complete"
fi

# `import torch` FIRST. The compiled ops link against torch's libc10.so, which
# is only on the loader's path once torch itself has been imported; importing
# them first fails with "libc10.so: cannot open shared object file" even on a
# perfectly good build. LOCAL_COMMANDS.md:117 has the same ordering bug, so
# this guard would otherwise never be satisfied and would recompile every run.
if ! "$OR3_ENVS/tfenet/bin/python" -c "import torch, DACONV_CUDA, DSCONV_CUDA" 2>/dev/null; then
    log "building TfeNet CUDA extensions (this takes a few minutes)"
    # Ensure build tools here rather than in the venv block above, so this still
    # holds when that block is skipped by its .complete marker.
    "$OR3_ENVS/tfenet/bin/pip" install -q --upgrade setuptools wheel ninja

    # The shipped prebuilt .so files were deliberately excluded from the tarball
    # and must be compiled against the local torch/nvcc. LOCAL_COMMANDS.md:13
    #
    # --no-build-isolation is REQUIRED, not an optimisation. Both setup.py files
    # do `from torch.utils.cpp_extension import BuildExtension, CUDAExtension`
    # at line 3, but PEP 517 builds in a clean isolated env that has no torch,
    # so the import dies before any compiling starts. Disabling isolation lets
    # the build see the venv's torch - which is also the torch we want the
    # extension linked against.
    ( cd "$OR3_CODE/TfeNet/DAConv" \
        && "$OR3_ENVS/tfenet/bin/pip" install -e . --no-build-isolation )
    ( cd "$OR3_CODE/TfeNet/DSConv" \
        && "$OR3_ENVS/tfenet/bin/pip" install -e . --no-build-isolation )
fi
log "verifying TfeNet CUDA ops"
( cd "$OR3_CODE/TfeNet" && "$OR3_ENVS/tfenet/bin/python" -c "
import torch                      # must precede the ops - loads libc10.so
import DACONV_CUDA, DSCONV_CUDA
print('  TfeNet CUDA ops OK, torch', torch.__version__, 'cuda', torch.version.cuda,
      'device', torch.cuda.get_device_name(0))" )

# --- 5. checkpoints present -------------------------------------------------
log "checking checkpoints"
for f in "$OR3_CODE/TfeNet/checkpoint/BAS/TfeNet_checkpoint.ckpt" \
         "$OR3_CODE/TfeNet/checkpoint/BAS/TfeNetSmall_checkpoint.ckpt" \
         "$OR3_CODE/AirMorph/checkpoints/airway_model1.pth" \
         "$OR3_CODE/AirMorph/checkpoints/wingsnet.ckpt" \
         "$OR3_CODE/AirMorph/checkpoints/break1.ckpt" \
         "$OR3_CODE/AirMorph/checkpoints/unet_r231-d5d2fc3d.pth"; do
    [ -f "$f" ] && echo "  ok  ${f#$OR3_CODE/}" || { echo "  MISSING $f" >&2; exit 1; }
done

log "setup complete"
