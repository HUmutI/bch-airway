#!/bin/bash
# Collect what the worker image needs from this file server into deploy/vendor/.
# Run once before `docker build`. Nothing here is generated; it is the same
# tarballs and scripts the cohort run used.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
BASE="${AIRWAY_BASE:-/fileserver/external/quin/ege}"
V=vendor
mkdir -p "$V/scripts"
for f in airmorph_seg.tar.gz medpseg_airway.tar.gz tfenet_bas.tar.gz fuse_union3.py; do
    cp -n "$BASE/all_3_lcc/$f" "$V/" 2>/dev/null || true
done
cp "$BASE/or_all3_run/scripts/09_airmorph_morphometry.py" "$V/scripts/"
cp "$BASE/cohort90_results/reproducibility_scripts/axis_geometry.py" "$V/scripts/"
cp "$BASE/or_all3_run/code/AirMorph/requirements.txt" "$V/requirements_airmorph.txt"
cp "$BASE/or_all3_run/code/TfeNet/requirements.txt" "$V/requirements_tfenet.txt"
md5sum -c - <<'SUM'
8ed1d39a68233f25f93eb1690693bdf3  vendor/airmorph_seg.tar.gz
abd5fd0efb9c68a01d90ecb04c27fd16  vendor/medpseg_airway.tar.gz
929f4ce36d1268ec9523d3ec8aa9ec3e  vendor/tfenet_bas.tar.gz
SUM
du -sh "$V"
echo "ready: docker build -f deploy/Dockerfile.worker -t airway-worker .   (run from airway_web/)"
