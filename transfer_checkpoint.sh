#!/bin/bash
set -euo pipefail

# Source (home3)
SRC="/home/xlan1/projects/verl-agent/checkpoints"

# Destination base on scratch1 (adjust if your path differs)
DEST_BASE="/wstor/scratch1/xlan1"
DEST="${DEST_BASE}/verl-agent/checkpoints"

echo "Syncing from:  ${SRC}"
echo "         to:    ${DEST}"
echo

mkdir -p "${DEST}"

# -a : archive (preserve perms, times, etc.)
# -v : verbose
# -h : human-readable sizes
# --progress : show copy progress
# --delete : make DEST mirror SRC (removes files not in SRC)
rsync -avh --progress --delete "${SRC}/" "${DEST}/"

echo
echo "Sync complete."