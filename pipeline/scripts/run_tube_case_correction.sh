#!/bin/bash
set -euo pipefail
RUN="${OR3_RUN:-/fileserver/external/quin/ege/or_all3_run}"
PY="$RUN/envs/airmorph/bin/python"
cd "$RUN"
"$PY" scripts/19_finish_tube_case_correction.py
