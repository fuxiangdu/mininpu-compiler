#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/run_v8.sh"
bash "${SCRIPT_DIR}/13_test_attention_pipeline.sh"

echo "[PASS] MiniNPU compiler v9 completed"
