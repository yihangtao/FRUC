#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Public inference entry point; uses the cooperative FRUC pipeline.
exec bash scripts/visualization/fruc/run_v2xreal_coop_inference.sh "$@"
