#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPT="${PROJECT_ROOT}/build/bin/mininpu-opt"
RESULT_DIR="${PROJECT_ROOT}/results/v4"
PIPELINE='builtin.module(mininpu-fuse-linear-relu,mininpu-plan-tiles,mininpu-lower-to-linalg)'

test -x "${OPT}" || {
  echo "[ERROR] mininpu-opt is missing; run scripts/01_build.sh first"
  exit 1
}

mkdir -p "${RESULT_DIR}"

"${OPT}" "${PROJECT_ROOT}/test/lowering.mlir" \
  "--pass-pipeline=${PIPELINE}" \
  --verify-each \
  -o "${RESULT_DIR}/lowered.mlir"

for expected in \
  "tensor.empty" \
  "linalg.fill" \
  "linalg.matmul" \
  "linalg.generic" \
  "arith.addf" \
  "arith.maximumf" \
  "linalg.yield"
do
  grep -Fq "${expected}" "${RESULT_DIR}/lowered.mlir" || {
    echo "[ERROR] lowered IR is missing ${expected}"
    exit 1
  }
done

if grep -Eq '"mininpu\.(matmul|bias_add|relu|fused_matmul_bias_relu)"' \
    "${RESULT_DIR}/lowered.mlir"; then
  echo "[ERROR] a MiniNPU operation survived dialect conversion"
  exit 1
fi

grep -Fq "mininpu.tile_m = 112" "${RESULT_DIR}/lowered.mlir" || {
  echo "[ERROR] tile-planning metadata was not preserved on linalg.matmul"
  exit 1
}

"${OPT}" "${RESULT_DIR}/lowered.mlir" \
  --mininpu-lower-to-linalg \
  --verify-each \
  -o "${RESULT_DIR}/lowered_twice.mlir"

if ! cmp -s "${RESULT_DIR}/lowered.mlir" \
    "${RESULT_DIR}/lowered_twice.mlir"; then
  echo "[ERROR] lowering pass is not idempotent"
  diff -u "${RESULT_DIR}/lowered.mlir" \
    "${RESULT_DIR}/lowered_twice.mlir" || true
  exit 1
fi

if "${OPT}" "${PROJECT_ROOT}/test/lowering.mlir" \
    --mininpu-lower-to-linalg -o /dev/null \
    >"${RESULT_DIR}/wrong_order.log" 2>&1; then
  echo "[ERROR] unfused MiniNPU operations were unexpectedly accepted"
  exit 1
fi

grep -Fq "failed to legalize operation" "${RESULT_DIR}/wrong_order.log" || {
  echo "[ERROR] expected conversion failure diagnostic was not found"
  sed -n '1,160p' "${RESULT_DIR}/wrong_order.log"
  exit 1
}

echo "===== lowered Tensor/Linalg/Arith IR ====="
sed -n '1,220p' "${RESULT_DIR}/lowered.mlir"
echo "===== expected wrong-pipeline rejection ====="
sed -n '1,100p' "${RESULT_DIR}/wrong_order.log"
echo "[PASS] MiniNPU fused operation lowered to standard MLIR dialects"
echo "[PASS] UB tile-planning metadata preserved on linalg.matmul"
echo "[PASS] Lowering pass idempotence validated"
echo "[PASS] Illegal unfused MiniNPU operations rejected"
