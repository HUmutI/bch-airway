#!/bin/sh
# Run only after the 83-case AirMorph classification job has completed.
set -eu

RUN="${OR3_RUN:-/fileserver/external/quin/ege/or_all3_run}"
PY="$RUN/envs/airmorph/bin/python"
STATUS="$RUN/normal_airway_morphometry_status.csv"
EXPECTED=83

cd "$RUN"

actual=$(( $(wc -l < "$STATUS") - 1 ))
if [ "$actual" -ne "$EXPECTED" ]; then
    echo "Refusing to finalize: status contains $actual/$EXPECTED cases." >&2
    echo "Wait for the classifier job to finish, then run this script again." >&2
    exit 2
fi

echo "[1/4] Re-exporting corrected morphometry from saved AirMorph products"
"$PY" scripts/09_airmorph_morphometry.py --phase export

echo "[2/4] Building conservative analysis-ready tables"
"$PY" scripts/13_build_analysis_ready_tables.py

echo "[3/4] Generating radiologist QC videos and review workbook"
"$PY" scripts/10_radiologist_qc.py

echo "[4/4] Running fail-closed delivery validation"
"$PY" scripts/11_validate_delivery.py --expected-cases "$EXPECTED"

mkdir -p radiologist_qc/tables
cp normal_airway_morphometry_status.csv radiologist_qc/tables/
cp delivery_validation.json radiologist_qc/

echo "Radiologist delivery passed validation: $RUN/radiologist_qc"
