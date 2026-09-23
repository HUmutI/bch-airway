#!/bin/bash
# Age-group-major runner: take one age group all the way through the pipeline -
# AirMorph -> MEDPSeg -> TfeNet -> fuse -> QC -> orientation check - and only
# then start the next group. Youngest first.
#
#   ./run_by_agegroup.sh                     # every group, 00-01 upward
#   ./run_by_agegroup.sh age_07-08           # just one group
#   ./run_by_agegroup.sh age_00-01 age_01-02 # a chosen subset, in that order
#
# Detached, for the real run:
#   nohup ./run_by_agegroup.sh > ../logs/bygroup_$(date +%F).log 2>&1 &
#
# Each group's results are complete and saved before the next begins, so the
# cohort is delivered as finished age cohorts rather than as one all-or-nothing
# batch at the end.
#
# Every stage still skips cases whose output already exists, so this is safe to
# stop and restart, and safe to run on top of the earlier stage-major run - the
# AirMorph masks already on disk are reused, not recomputed.
. "$(dirname "$(readlink -f "$0")")/config.sh"
S="$(dirname "$(readlink -f "$0")")"

# Which cohort to work through. Defaults to every staged scan; set
# OR3_MASTER_LIST to a file of case directories to restrict the run to a subset
# (e.g. cases_normal.txt, the 83 normal scans). Age grouping, the preflight and
# the final cohort QC all follow whichever list is selected.
MASTER="${OR3_MASTER_LIST:-$OR3_CASES_ALL}"
[ -s "$MASTER" ] || { echo "case list $MASTER missing or empty" >&2; exit 1; }

# NB: never name a shell array GROUPS - bash reserves it for the current
# user's group IDs, so `mapfile -t GROUPS` fails and `set -e` kills the script
# silently. Hence AGE_GROUPS.
BUCKET_DIR="$OR3_RUN/buckets"
# Per-run QC directory. Override with OR3_QCDIR so a second cohort (e.g. the
# abnormals) writes its own tables instead of overwriting the first run's.
QCDIR="${OR3_QCDIR:-$OR3_RUN/qc_agegroups}"
mkdir -p "$BUCKET_DIR" "$QCDIR"

# Split the cohort list into one file per age group. Names are zero-padded, so
# a plain sort is already youngest-first.
awk -F/ '{for(i=1;i<=NF;i++) if($i ~ /^age_/) print $i}' "$MASTER" \
    | sort -u > "$BUCKET_DIR/.all_groups"
while read -r g; do
    grep "/$g/" "$MASTER" > "$BUCKET_DIR/$g.txt"
done < "$BUCKET_DIR/.all_groups"

if [ $# -gt 0 ]; then
    AGE_GROUPS=("$@")
else
    mapfile -t AGE_GROUPS < "$BUCKET_DIR/.all_groups"
fi

echo "############################################################"
echo "age-group-major run"
echo "groups: ${AGE_GROUPS[*]}"
echo "cohort: $MASTER ($(wc -l < "$MASTER") cases)"
echo "connectivity=$OR3_CONNECTIVITY   started $(date)"
echo "############################################################"

RUN_START=$(date +%s)
PROBLEMS=()

# One-off preflight: AirMorph does not write atomically, so a previous run that
# was killed mid-write can have left a truncated mask behind. Drop anything
# unreadable now and the owning stage will simply recompute it.
echo
echo "==== preflight: verifying existing outputs ===="
py_airmorph "$S/08_check_integrity.py" --cases "$MASTER" \
    || PROBLEMS+=("integrity preflight")

for g in "${AGE_GROUPS[@]}"; do
    LIST="$BUCKET_DIR/$g.txt"
    if [ ! -s "$LIST" ]; then
        echo "!! no cases for $g - skipping"; continue
    fi
    n=$(wc -l < "$LIST")

    echo
    echo "=========================================================="
    echo "AGE GROUP $g   ($n cases)   $(date '+%F %T')"
    echo "=========================================================="

    step() {
        local name="$1"; shift
        if ! "$@"; then
            echo "  !! $g: $name reported failures"
            PROBLEMS+=("$g $name")
        fi
    }

    step airmorph py_airmorph "$S/02_airmorph.py" --cases "$LIST"
    step medpseg  py_medpseg  "$S/03_medpseg.py"  --cases "$LIST"
    step tfenet   py_tfenet   "$S/04_tfenet.py"   --cases "$LIST"
    step fuse     py_airmorph "$S/05_fuse.py"     --cases "$LIST" \
                      --connectivity "$OR3_CONNECTIVITY"
    step qc       py_airmorph "$S/06_qc.py"       --cases "$LIST" \
                      --out "$QCDIR/qc_$g.csv"
    step verify   py_airmorph "$S/07_verify_orientation.py" --cases "$LIST"

    fused=$(grep -c . "$LIST")
    have=$(while read -r d; do
             [ -f "$d/union3_lcc.nii.gz" ] && echo x; done < "$LIST" | wc -l)
    total=$(find "$OR3_OUT" -name union3_lcc.nii.gz | wc -l)
    el=$(( $(date +%s) - RUN_START ))
    echo
    echo "---- $g COMPLETE: $have/$fused fused in this group "\
"| cohort $total fused | elapsed $((el/3600))h$(( (el%3600)/60 ))m ----"
    echo "---- results: $OR3_OUT/$g/   qc: $QCDIR/qc_$g.csv ----"
done

echo
echo "############################################################"
echo "ALL REQUESTED GROUPS DONE  $(date)"
echo "fused in this cohort: $(while read -r d; do [ -f "$d/union3_lcc.nii.gz" ] && echo x; done < "$MASTER" | wc -l)/$(wc -l < "$MASTER")"
echo "############################################################"

# Cohort-wide QC last: the per-group files score each case only against its own
# age group, which is what you want clinically, but the cohort file is the one
# to read for cross-group comparisons.
echo
echo "==== cohort-wide QC ===="
py_airmorph "$S/06_qc.py" --cases "$MASTER" --out "$QCDIR/qc_cohort.csv" \
    || PROBLEMS+=("final qc")

if [ ${#PROBLEMS[@]} -gt 0 ]; then
    echo
    echo "reported failures:"
    printf '    %s\n' "${PROBLEMS[@]}"
    echo "Re-run the same command; finished cases are skipped."
    exit 1
fi
echo
echo "all groups clean"
