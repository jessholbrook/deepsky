#!/usr/bin/env bash
# Continuously sync checkpoints + latest sample grid off the pod.
#   bash cloud/sync_checkpoints.sh runs/cloud128 myremote:deepsky-ckpts
set -euo pipefail

RUN_DIR="${1:?usage: sync_checkpoints.sh <run_dir> <rclone-remote:path>}"
REMOTE="${2:?usage: sync_checkpoints.sh <run_dir> <rclone-remote:path>}"

while true; do
    latest_grid=$(ls -t "$RUN_DIR"/samples_*.png 2>/dev/null | head -1 || true)
    if [ -n "$latest_grid" ]; then
        rclone copyto "$latest_grid" "$REMOTE/samples_latest.png"
    fi
    rclone copy "$RUN_DIR" "$REMOTE" --include "ckpt_*.pt" --include "metrics.csv"
    echo "synced $(date)"
    sleep 1800
done
