#!/usr/bin/env bash
# Fetch the three model tarballs (code + trained weights) from this repo's
# GitHub Release and verify them. They are too large for git itself.
#
#   ./models/download_models.sh                # -> models/tarballs/
#   ./models/download_models.sh /some/dir      # -> /some/dir
#
# Needs `gh` logged in to an account that can read the repo (the repo is
# private), or set MODELS_URL to a mirror that serves the same files.
set -euo pipefail

REPO="${MODELS_REPO:-quin-med-harvard-edu/airway-seg-and-labelling}"   # mirror: HUmutI/bch-airway
TAG="${MODELS_TAG:-models-v1}"
DEST="${1:-$(cd "$(dirname "$0")" && pwd)/tarballs}"
mkdir -p "$DEST"

declare -A MD5=(
  [airmorph_seg.tar.gz]=8ed1d39a68233f25f93eb1690693bdf3
  [medpseg_airway.tar.gz]=abd5fd0efb9c68a01d90ecb04c27fd16
  [tfenet_bas.tar.gz]=929f4ce36d1268ec9523d3ec8aa9ec3e
)

for f in "${!MD5[@]}"; do
  if [ -f "$DEST/$f" ] && echo "${MD5[$f]}  $DEST/$f" | md5sum -c --quiet - 2>/dev/null; then
    echo "have  $f"; continue
  fi
  echo "fetch $f"
  rm -f "$DEST/$f"
  gh release download "$TAG" -R "$REPO" -p "$f" -D "$DEST"
  echo "${MD5[$f]}  $DEST/$f" | md5sum -c --quiet - || { echo "checksum mismatch: $f" >&2; exit 1; }
done
echo "all three tarballs in $DEST"
