#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Train the single-agent V2X-Real path.
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" scripts/training/fruc/train.py \
    --image_dir "${IMAGE_DIR:-data/v2xreal/train}" --log_dir "${LOG_DIR:-logs/fruc_ego}" \
    --ckpt_path "${CKPT_PATH:-pretrained/fruc/model_fruc_v2xreal.pth}" --dataset_type v2xreal \
    --sequence_length 4 --max_epoch 10 "$@"
