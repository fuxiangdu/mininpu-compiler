#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/run_v7.sh"
bash "${SCRIPT_DIR}/10_test_batch_matmul_lowering.sh"
bash "${SCRIPT_DIR}/11_test_softmax_lowering.sh"
bash "${SCRIPT_DIR}/12_test_rms_norm_lowering.sh"

echo "[PASS] MiniNPU compiler v8 completed"
