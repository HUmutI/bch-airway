#!/bin/bash
# The web API. Runs anywhere; needs no GPU.
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p jobs
exec ./.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
