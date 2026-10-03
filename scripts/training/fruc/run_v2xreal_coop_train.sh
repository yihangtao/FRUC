#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Activate the FRUC environment before running this script.
# Override paths and GPU count through environment variables; extra CLI flags are forwarded.
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" scripts/training/fruc/train.py \
    --image_dir "${IMAGE_DIR:-data/v2xreal/train}" \
    --log_dir "${LOG_DIR:-logs/fruc_v2xreal}" \
    --ckpt_path "${CKPT_PATH:-pretrained/fruc/model_fruc_v2xreal.pth}" \
    --dataset_type v2xreal_coop \
    --batch_size 1 --max_epoch 10 --save_image 500 --save_ckpt 2000 \
    --fruc_loss_weight 0.2 "$@"
