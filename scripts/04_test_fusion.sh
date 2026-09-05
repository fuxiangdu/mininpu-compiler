#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPT="${PROJECT_ROOT}/build/bin/mininpu-opt"
RESULT_DIR="${PROJECT_ROOT}/results/v2"
PASS_NAME="--mininpu-fuse-linear-relu"

test -x "${OPT}" || {
  echo "[ERROR] mininpu-opt is missing; run scripts/01_build.sh first"
  exit 1
}

mkdir -p "${RESULT_DIR}"

"${OPT}" "${PROJECT_ROOT}/test/fusion.mlir" "${PASS_NAME}" \
  -o "${RESULT_DIR}/fused-once.mlir"

grep -Fq 'mininpu.fused_matmul_bias_relu' \
  "${RESULT_DIR}/fused-once.mlir" || {
    echo "[ERROR] fused operation was not generated"
    exit 1
  }

for old_op in mininpu.matmul mininpu.bias_add '"mininpu.relu"'; do
  if grep -Fq "${old_op}" "${RESULT_DIR}/fused-once.mlir"; then
    echo "[ERROR] unfused operation remained: ${old_op}"
    exit 1
  fi
done

"${OPT}" "${RESULT_DIR}/fused-once.mlir" "${PASS_NAME}" \
  -o "${RESULT_DIR}/fused-twice.mlir"

diff -u "${RESULT_DIR}/fused-once.mlir" \
  "${RESULT_DIR}/fused-twice.mlir" >"${RESULT_DIR}/idempotence.diff" || {
    echo "[ERROR] fusion pass is not idempotent"
    sed -n '1,160p' "${RESULT_DIR}/idempotence.diff"
    exit 1
  }

"${OPT}" "${PROJECT_ROOT}/test/no_fusion_shared_use.mlir" "${PASS_NAME}" \
  -o "${RESULT_DIR}/shared-use.mlir"

if grep -Fq 'mininpu.fused_matmul_bias_relu' \
    "${RESULT_DIR}/shared-use.mlir"; then
  echo "[ERROR] graph with a shared intermediate was incorrectly fused"
  exit 1
fi

for old_op in mininpu.matmul mininpu.bias_add '"mininpu.relu"'; do
  grep -Fq "${old_op}" "${RESULT_DIR}/shared-use.mlir" || {
    echo "[ERROR] expected operation disappeared from shared-use graph: ${old_op}"
    exit 1
  }
done

echo "===== before fusion ====="
sed -n '1,160p' "${PROJECT_ROOT}/test/fusion.mlir"
echo "===== after fusion ====="
sed -n '1,160p' "${RESULT_DIR}/fused-once.mlir"
echo "===== shared-use graph left unchanged ====="
sed -n '1,180p' "${RESULT_DIR}/shared-use.mlir"
echo "[PASS] MatMul-BiasAdd-ReLU chain fused into one operation"
echo "[PASS] Fusion pass is idempotent"
echo "[PASS] Shared-intermediate safety guard passed"
