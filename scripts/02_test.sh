#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPT="${PROJECT_ROOT}/build/bin/mininpu-opt"
RESULT_DIR="${PROJECT_ROOT}/results/v0"

test -x "${OPT}" || {
  echo "[ERROR] mininpu-opt is missing; run scripts/01_build.sh first"
  exit 1
}

mkdir -p "${RESULT_DIR}"

"${OPT}" "${PROJECT_ROOT}/test/smoke.mlir" \
  -o "${RESULT_DIR}/smoke-roundtrip.mlir"
"${OPT}" "${RESULT_DIR}/smoke-roundtrip.mlir" \
  -o /dev/null

"${OPT}" "${PROJECT_ROOT}/test/canonicalize.mlir" \
  --canonicalize \
  -o "${RESULT_DIR}/canonicalized.mlir"

grep -Eq 'arith.constant 15( : i32)?' "${RESULT_DIR}/canonicalized.mlir" || {
  echo "[ERROR] expected folded constant 15 was not found"
  sed -n '1,160p' "${RESULT_DIR}/canonicalized.mlir"
  exit 1
}

if grep -q 'arith.addi' "${RESULT_DIR}/canonicalized.mlir"; then
  echo "[ERROR] arith.addi remained after canonicalization"
  sed -n '1,160p' "${RESULT_DIR}/canonicalized.mlir"
  exit 1
fi

echo "===== round-trip IR ====="
sed -n '1,120p' "${RESULT_DIR}/smoke-roundtrip.mlir"
echo "===== canonicalized IR ====="
sed -n '1,120p' "${RESULT_DIR}/canonicalized.mlir"
echo "[PASS] MLIR parse/print round trip passed"
echo "[PASS] Built-in canonicalization and constant folding passed"

