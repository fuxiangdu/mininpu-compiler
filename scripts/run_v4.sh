#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/00_preflight.sh"
bash "${SCRIPT_DIR}/01_build.sh"
bash "${SCRIPT_DIR}/02_test.sh"
bash "${SCRIPT_DIR}/03_test_dialect.sh"
bash "${SCRIPT_DIR}/04_test_fusion.sh"
bash "${SCRIPT_DIR}/05_test_tiling.sh"
bash "${SCRIPT_DIR}/06_test_lowering.sh"

echo "[PASS] MiniNPU compiler v4 completed"
