#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

exec python scripts/visualization/fruc/object_edit_web_app.py \
    --image_dir "${IMAGE_DIR:-data/v2xreal/val}" \
    --ckpt_path "${CKPT_PATH:-pretrained/fruc/model_fruc_v2xreal.pth}" "$@"
