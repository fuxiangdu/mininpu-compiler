#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/00_preflight.sh"
bash "${SCRIPT_DIR}/01_build.sh"
bash "${SCRIPT_DIR}/02_test.sh"
bash "${SCRIPT_DIR}/03_test_dialect.sh"

echo "[PASS] MiniNPU compiler v1 completed"
