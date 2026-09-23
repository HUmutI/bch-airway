#!/bin/bash
# Orchestrator: run every stage in order over a case list.
#
#   ./run_pipeline.sh pilot        # 13 representative cases (DO THIS FIRST)
#   ./run_pipeline.sh all          # the full 849
#   ./run_pipeline.sh all 04       # resume the full run from stage 4 onward
#
# Every stage is resumable - a case whose output is already on disk is skipped -
# so re-running after a crash, a reboot or an OOM picks up where it stopped.
#
# A stage that loses individual cases does NOT abort the run: one bad scan must
# not cost you the other 848. Failures are logged per case, counted per stage,
# and summarised at the end, and the script exits non-zero if any occurred.
#
# For the full cohort use nohup, since it runs for days:
#   nohup ./run_pipeline.sh all > ../logs/full_$(date +%F).log 2>&1 &
. "$(dirname "$(readlink -f "$0")")/config.sh"
S="$(dirname "$(readlink -f "$0")")"

MODE="${1:-pilot}"
FROM="${2:-01}"

case "$MODE" in
    pilot) export OR3_CASES="$OR3_CASES_PILOT"; SUBSET=pilot ;;
    all)   export OR3_CASES="$OR3_CASES_ALL";   SUBSET=all ;;
    *)     echo "usage: $0 {pilot|all} [start-stage]" >&2; exit 1 ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
PROBLEMS=()

banner() { echo; echo "############ $* ############"; date; echo; }
want()   { [ "$1" \> "$FROM" ] || [ "$1" = "$FROM" ]; }

# Run a stage, tee its log, and record (but do not propagate) per-case failures.
run_stage() {
    local num="$1" name="$2"; shift 2
    banner "STAGE $num  $name"
    if "$@" 2>&1 | tee "$OR3_LOGS/${num}_${name%% *}_$STAMP.log"; then
        echo "stage $num ok"
    else
        echo "stage $num reported failures - see $OR3_LOGS/${num}_${name%% *}_$STAMP.log"
        PROBLEMS+=("$num $name")
    fi
}

# Stage 1 always runs: it builds the case lists every other stage reads. If it
# cannot produce a usable list there is nothing to do, so this one is fatal.
if want 01; then
    banner "STAGE 01  staging -> LPS ($SUBSET)"
    if ! py_airmorph "$S/01_stage.py" --subset "$SUBSET" 2>&1 \
            | tee "$OR3_LOGS/01_stage_$STAMP.log"; then
        PROBLEMS+=("01 staging")
    fi
fi

[ -s "$OR3_CASES" ] || { echo "case list $OR3_CASES is empty or missing" >&2; exit 1; }
echo "case list: $OR3_CASES ($(wc -l < "$OR3_CASES") cases)"

# NB: `want NN && run_stage ...` would abort the script under `set -e` whenever
# the guard is false (a failed AND-list is itself a failure), which is exactly
# what happens when resuming from a later stage. Use explicit if/fi.
if want 02; then
    run_stage 02 "airmorph" \
        py_airmorph "$S/02_airmorph.py" --cases "$OR3_CASES"
fi

if want 03; then
    run_stage 03 "medpseg" \
        py_medpseg "$S/03_medpseg.py" --cases "$OR3_CASES"
fi

if want 04; then
    run_stage 04 "tfenet" \
        py_tfenet "$S/04_tfenet.py" --cases "$OR3_CASES"
fi

if want 05; then
    run_stage 05 "fuse" \
        py_airmorph "$S/05_fuse.py" --cases "$OR3_CASES" \
            --connectivity "$OR3_CONNECTIVITY"
fi

if want 06; then
    run_stage 06 "qc" \
        py_airmorph "$S/06_qc.py" --cases "$OR3_CASES"
fi

if want 07; then
    run_stage 07 "verify" \
        py_airmorph "$S/07_verify_orientation.py" --cases "$OR3_CASES"
fi

banner "PIPELINE COMPLETE ($MODE)"
echo "results:  $OR3_OUT"
echo "qc:       $OR3_RUN/qc.csv"
echo "logs:     $OR3_LOGS"
if [ ${#PROBLEMS[@]} -gt 0 ]; then
    echo
    echo "stages that reported failures:"
    printf '    %s\n' "${PROBLEMS[@]}"
    echo "Re-run the same command once fixed; finished cases are skipped."
    exit 1
fi
echo "all stages clean"
