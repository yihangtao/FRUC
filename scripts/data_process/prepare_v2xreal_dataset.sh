#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 RAW_ROOT TARGET_ROOT [DEVICE] [CAMERA_INDEX_BASE]" >&2
    exit 2
fi
RAW_ROOT="$1"
TARGET_ROOT="$2"
DEVICE="${3:-cuda:0}"
CAMERA_INDEX_BASE="${4:-0}"
for split in train val test; do
    RAW_SPLIT="$RAW_ROOT/$split"
    if [[ "$split" == val && ! -d "$RAW_SPLIT" && -d "$RAW_ROOT/validate" ]]; then
        RAW_SPLIT="$RAW_ROOT/validate"
    fi
    [[ -d "$RAW_SPLIT" ]] || continue
    python datasets/preprocess_v2xreal.py --data_root "$RAW_SPLIT" \
        --target_dir "$TARGET_ROOT/$split" --split "$split" --camera_index_base "$CAMERA_INDEX_BASE"
    python scripts/data_process/generate_masks_v2xreal_smp.py \
        --data_root "$TARGET_ROOT/$split" --device "$DEVICE"
done
echo "Preprocessing finished. Add or verify context.json view associations before cooperative training."
