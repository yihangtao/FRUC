#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Remove foreground Gaussians and inspect the reconstructed background.
exec python scripts/visualization/fruc/demo_v2xreal_coop_novel_view.py \
    --image_dir "${IMAGE_DIR:-data/v2xreal/val}" \
    --ckpt_path "${CKPT_PATH:-pretrained/fruc/model_fruc_v2xreal.pth}" \
    --output_dir "${OUTPUT_DIR:-output/fruc_editing}" --remove_all_dynamic --offset_scales 0 "$@"
