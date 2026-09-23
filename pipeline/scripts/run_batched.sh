#!/bin/bash
# Case-major runner: push N cases all the way through AirMorph -> MEDPSeg ->
# TfeNet -> fuse -> QC -> verify before starting the next N.
#
#   ./run_batched.sh            # 20 cases per batch, full cohort
#   ./run_batched.sh 20 pilot   # 20 per batch, pilot list only
#   ./run_batched.sh 40         # 40 per batch
#
# WHY, versus run_pipeline.sh's stage-major order. run_pipeline.sh runs all 849
# through AirMorph, then all 849 through MEDPSeg, and so on. That means the
# first finished union3_lcc mask only appears after every GPU stage has run -
# about six days in - and a defect specific to a later stage is not visible
# until then. Batching by case gives finished masks from the first batch onward
# and surfaces cross-stage problems immediately.
#
# The cost is reloading model weights once per batch instead of once per run:
# roughly 30 s of AirMorph and TfeNet startup per batch, so ~20 min across the
# whole cohort. Negligible against days of inference. (MEDPSeg pays nothing
# extra - 03_medpseg.py already invokes its CLI once per case.)
#
# Every stage still skips cases whose output exists, so this is safe to stop and
# restart, and safe to run after a partial stage-major run: the 140 AirMorph
# masks already on disk are simply skipped.
#
# For the real run, detach it:
#   nohup ./run_batched.sh 20 > ../logs/batched_$(date +%F).log 2>&1 &
. "$(dirname "$(readlink -f "$0")")/config.sh"
S="$(dirname "$(readlink -f "$0")")"

BATCH="${1:-20}"
WHICH="${2:-all}"
case "$WHICH" in
    all)   SRC="$OR3_CASES_ALL" ;;
    pilot) SRC="$OR3_CASES_PILOT" ;;
    *)     echo "usage: $0 [cases-per-batch] [all|pilot]" >&2; exit 1 ;;
esac

[ -s "$SRC" ] || { echo "case list $SRC missing or empty - run 01_stage.py" >&2; exit 1; }
TOTAL=$(wc -l < "$SRC")
BATCHES=$(( (TOTAL + BATCH - 1) / BATCH ))
WORK="$OR3_RUN/.batch_cases.txt"
QCDIR="$OR3_RUN/qc_batches"
mkdir -p "$QCDIR"

echo "############################################################"
echo "batched run: $TOTAL cases, $BATCH per batch, $BATCHES batches"
echo "connectivity=$OR3_CONNECTIVITY  tfenet batch=${OR3_TFENET_BATCH:-1}"
echo "started $(date)"
echo "############################################################"

RUN_START=$(date +%s)
PROBLEMS=()

for ((b = 1; b <= BATCHES; b++)); do
    from=$(( (b - 1) * BATCH + 1 ))
    to=$(( b * BATCH ))
    sed -n "${from},${to}p" "$SRC" > "$WORK"
    n=$(wc -l < "$WORK")
    [ "$n" -gt 0 ] || continue

    tag=$(printf "%03d" "$b")
    echo
    echo "=========================================================="
    echo "BATCH $b/$BATCHES  (cases $from-$to, $n cases)   $(date '+%F %T')"
    echo "=========================================================="

    # A stage that loses cases must not abort the cohort; record and continue.
    step() {
        local name="$1"; shift
        if ! "$@"; then
            echo "  !! batch $tag: $name reported failures"
            PROBLEMS+=("batch $tag $name")
        fi
    }

    step airmorph py_airmorph "$S/02_airmorph.py" --cases "$WORK"
    step medpseg  py_medpseg  "$S/03_medpseg.py"  --cases "$WORK"
    step tfenet   py_tfenet   "$S/04_tfenet.py"   --cases "$WORK"
    step fuse     py_airmorph "$S/05_fuse.py"     --cases "$WORK" \
                      --connectivity "$OR3_CONNECTIVITY"
    step qc       py_airmorph "$S/06_qc.py"       --cases "$WORK" \
                      --out "$QCDIR/qc_batch_$tag.csv"
    step verify   py_airmorph "$S/07_verify_orientation.py" --cases "$WORK"

    done_n=$(find "$OR3_OUT" -name union3_lcc.nii.gz | wc -l)
    el=$(( $(date +%s) - RUN_START ))
    echo "---- batch $b/$BATCHES done. $done_n/$TOTAL cases fused. "\
"elapsed $((el/3600))h$(( (el%3600)/60 ))m ----"
    if [ "$b" -gt 0 ] && [ "$done_n" -gt 0 ]; then
        eta=$(( el * (TOTAL - done_n) / done_n ))
        echo "---- projected remaining: $((eta/3600))h$(( (eta%3600)/60 ))m ----"
    fi
done

rm -f "$WORK"

echo
echo "############################################################"
echo "ALL BATCHES COMPLETE  $(date)"
echo "fused: $(find "$OR3_OUT" -name union3_lcc.nii.gz | wc -l)/$TOTAL"
echo "############################################################"

# Cohort-wide QC over everything, now that all batches are in. Per-batch QC
# scores each case only against its own batch, which is too small a peer group
# for the age-bucket outlier test to mean much.
echo
echo "==== cohort-wide QC ===="
py_airmorph "$S/06_qc.py" --cases "$SRC" || PROBLEMS+=("final qc")
echo
echo "==== cohort-wide orientation verification ===="
py_airmorph "$S/07_verify_orientation.py" --cases "$SRC" || PROBLEMS+=("final verify")

if [ ${#PROBLEMS[@]} -gt 0 ]; then
    echo
    echo "stages that reported failures:"
    printf '    %s\n' "${PROBLEMS[@]}"
    echo "Re-run the same command; finished cases are skipped."
    exit 1
fi
echo
echo "all batches clean"
