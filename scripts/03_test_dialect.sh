#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPT="${PROJECT_ROOT}/build/bin/mininpu-opt"
RESULT_DIR="${PROJECT_ROOT}/results/v1"

test -x "${OPT}" || {
  echo "[ERROR] mininpu-opt is missing; run scripts/01_build.sh first"
  exit 1
}

mkdir -p "${RESULT_DIR}"

"${OPT}" "${PROJECT_ROOT}/test/dialect.mlir" \
  -o "${RESULT_DIR}/dialect-roundtrip.mlir"
"${OPT}" "${RESULT_DIR}/dialect-roundtrip.mlir" -o /dev/null

for op in mininpu.matmul mininpu.bias_add mininpu.relu; do
  grep -q "${op}" "${RESULT_DIR}/dialect-roundtrip.mlir" || {
    echo "[ERROR] expected operation is missing: ${op}"
    exit 1
  }
done

if "${OPT}" "${PROJECT_ROOT}/test/invalid_matmul.mlir" \
    -o /dev/null >"${RESULT_DIR}/invalid_matmul.log" 2>&1; then
  echo "[ERROR] invalid matmul unexpectedly passed verification"
  exit 1
fi

grep -q "incompatible contracting dimensions" \
  "${RESULT_DIR}/invalid_matmul.log" || {
    echo "[ERROR] expected verifier diagnostic was not found"
    sed -n '1,160p' "${RESULT_DIR}/invalid_matmul.log"
    exit 1
  }

echo "===== registered MiniNPU IR ====="
sed -n '1,160p' "${RESULT_DIR}/dialect-roundtrip.mlir"
echo "===== expected verifier rejection ====="
sed -n '1,80p' "${RESULT_DIR}/invalid_matmul.log"
echo "[PASS] MiniNPU dialect parse/print round trip passed"
echo "[PASS] MatMul, BiasAdd and Relu operations are registered"
echo "[PASS] MiniNPU MatMul verifier rejected an invalid shape"
