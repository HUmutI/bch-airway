# Shared configuration for the or_all3_lcc cohort run.
# Source this from every stage script:  . "$(dirname "$0")/config.sh"

set -euo pipefail

# --- layout -----------------------------------------------------------------
export OR3_BASE="${OR3_BASE:-/fileserver/external/quin/ege}"
export OR3_RUN="$OR3_BASE/or_all3_run"       # code, envs, logs, manifest
export OR3_CODE="$OR3_RUN/code"              # extracted AirMorph/ medpseg/ TfeNet/
export OR3_ENVS="$OR3_RUN/envs"              # three venvs
export OR3_LOGS="$OR3_RUN/logs"
export OR3_TARBALLS="$OR3_BASE/all_3_lcc"    # source tarballs + fuse_union3.py
export OR3_DATASET="$OR3_BASE/airway-dataset"
export OR3_OUT="$OR3_BASE/airway-seg"        # results, mirrors dataset layout

# --- case lists -------------------------------------------------------------
export OR3_MANIFEST="$OR3_RUN/manifest.csv"
export OR3_CASES_ALL="$OR3_RUN/cases_all.txt"
export OR3_CASES_PILOT="$OR3_RUN/cases_pilot.txt"

# Which list the stage scripts iterate over. Override per invocation:
#   OR3_CASES=$OR3_CASES_ALL ./02_airmorph.sh
export OR3_CASES="${OR3_CASES:-$OR3_CASES_PILOT}"

# --- CUDA -------------------------------------------------------------------
# nvcc 12.4 is installed at /usr/local/cuda-12.4; torch is installed cu124 to
# match. LOCAL_COMMANDS.md:121 warns that TfeNet build failures are almost
# always a torch/nvcc CUDA mismatch, so keep these two in step.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_VERSION=2.5.1
# MEDPSeg imports torchvision.transforms at runtime (image_read.py:19,
# poly_seg_2d_module.py:26) but leaves it unpinned. 0.20.1 is the build that
# pairs with torch 2.5.1; it must come from the SAME CUDA index as torch, or
# pip pulls the PyPI default (cu124) on top of a cu121 torch.
export TORCHVISION_VERSION=0.20.1

# Torch CUDA build is chosen PER ENV, deliberately:
#
#   AirMorph / MEDPSeg -> cu121. AirMorph's requirements.txt is a pip freeze of a
#     working cu121 install and pins the runtime libs hard
#     (nvidia-cublas-cu12==12.1.3.1, nvidia-cuda-runtime-cu12==12.1.105,
#     nvidia-nccl-cu12==2.21.5, and "# torch==2.5.1+cu121"). Installing torch
#     cu124 first and then those requirements DOWNGRADES cuBLAS et al to the
#     12.1 set underneath a cu124 torch binary, which dies at CUDA init with an
#     undefined symbol. Neither pipeline compiles anything, so nvcc's version is
#     irrelevant to them - matching the vendor's tested pin set is what matters.
#
#   TfeNet -> cu124, to match the installed nvcc 12.4 that builds DAConv/DSConv.
#     Its requirements pin no CUDA libs, so there is nothing to conflict.
export TORCH_INDEX_CU121=https://download.pytorch.org/whl/cu121
export TORCH_INDEX_CU124=https://download.pytorch.org/whl/cu124
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- fusion -----------------------------------------------------------------
# 6  = fuse_union3.py default; the rule the published or_all3_lcc numbers were
#      measured with (DSC 93.06, clDice 90.24, TLD 95.22, BND 92.31).
# 26 = MONAI KeepLargestConnectedComponent(connectivity=3), the BCH_LIAM
#      clinical runs.
# Must stay FIXED across the whole cohort. See LOCAL_COMMANDS.md:311-314.
export OR3_CONNECTIVITY="${OR3_CONNECTIVITY:-6}"

# --- parallelism ------------------------------------------------------------
# Staging is CPU/IO bound (gzip); the GPU stages are strictly serial.
export OR3_STAGE_WORKERS="${OR3_STAGE_WORKERS:-10}"

mkdir -p "$OR3_RUN" "$OR3_CODE" "$OR3_ENVS" "$OR3_LOGS" "$OR3_OUT"

py_airmorph() { "$OR3_ENVS/airmorph/bin/python" "$@"; }
py_medpseg()  { "$OR3_ENVS/medpseg/bin/python"  "$@"; }
py_tfenet()   { "$OR3_ENVS/tfenet/bin/python"   "$@"; }
