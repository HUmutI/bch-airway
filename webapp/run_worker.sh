#!/bin/bash
# The GPU worker. Run OUTSIDE the sandbox, on the machine with the card.
#   ./run_worker.sh          real pipeline
#   ./run_worker.sh --mock   no GPU, fakes the stages with a finished cohort case
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p jobs
export AIRWAY_MOCK_CASE="${AIRWAY_MOCK_CASE:-/fileserver/external/quin/ege/airway-seg/age_00-01/sub-0012/sub-0012_acc-0295}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="$CUDA_HOME/bin:$PATH"
exec ./.venv/bin/python worker/worker.py "$@"
