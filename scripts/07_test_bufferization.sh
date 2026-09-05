#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPT="${PROJECT_ROOT}/build/bin/mininpu-opt"
RESULT_DIR="${PROJECT_ROOT}/results/v5"

# v5 deliberately keeps linalg operations in buffer form.  Loop/LLVM lowering
# and execution are separate stages so that this regression can inspect the
# tensor-to-memref ownership boundary directly.
PIPELINE='builtin.module(mininpu-fuse-linear-relu,mininpu-plan-tiles,mininpu-lower-to-linalg,empty-tensor-to-alloc-tensor,one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline,canonicalize,cse)'

test -x "${OPT}" || {
  echo "[ERROR] mininpu-opt is missing; run scripts/01_build.sh first"
  exit 1
}

mkdir -p "${RESULT_DIR}"

"${OPT}" "${PROJECT_ROOT}/test/bufferization_local.mlir" \
  "--pass-pipeline=${PIPELINE}" \
  --verify-each \
  -o "${RESULT_DIR}/bufferized.mlir"

for expected in \
  "memref<" \
  "memref.alloc" \
  "memref.dealloc" \
  "memref.load" \
  "linalg.matmul" \
  "linalg.generic" \
  "call @consume_f32" \
  "mininpu.tile_m = 112"
do
  grep -Fq "${expected}" "${RESULT_DIR}/bufferized.mlir" || {
    echo "[ERROR] bufferized IR is missing ${expected}"
    exit 1
  }
done

if grep -Eq 'tensor<|tensor\.' "${RESULT_DIR}/bufferized.mlir"; then
  echo "[ERROR] tensor types or tensor operations survived v5 bufferization"
  exit 1
fi

if grep -Eq '"mininpu\.(matmul|bias_add|relu|fused_matmul_bias_relu)"' \
    "${RESULT_DIR}/bufferized.mlir"; then
  echo "[ERROR] a MiniNPU operation survived the v5 pipeline"
  exit 1
fi

# One-Shot Bufferize must reject the custom tensor operations when it is run
# before MiniNPU lowering.  A successful command here would hide a bad pass
# order behind an unbufferized boundary.
if "${OPT}" "${PROJECT_ROOT}/test/bufferization_local.mlir" \
    '--pass-pipeline=builtin.module(empty-tensor-to-alloc-tensor,one-shot-bufferize{bufferize-function-boundaries})' \
    -o /dev/null >"${RESULT_DIR}/wrong_order.log" 2>&1; then
  echo "[ERROR] bufferization unexpectedly accepted unlowered MiniNPU ops"
  exit 1
fi

ALLOC_COUNT=$(grep -Fc "memref.alloc" "${RESULT_DIR}/bufferized.mlir")
DEALLOC_COUNT=$(grep -Fc "memref.dealloc" "${RESULT_DIR}/bufferized.mlir")

echo "===== bufferized MemRef/Linalg IR ====="
sed -n '1,260p' "${RESULT_DIR}/bufferized.mlir"
echo "===== expected wrong-pipeline rejection ====="
sed -n '1,100p' "${RESULT_DIR}/wrong_order.log"
echo "[PASS] Tensor values and tensor operations were eliminated"
echo "[PASS] Linalg operations now use MemRef buffer semantics"
echo "[PASS] Tile-planning metadata survived bufferization"
echo "[PASS] Local buffer ownership was closed by memref.dealloc"
echo "[PASS] Incorrect bufferization order was rejected"
echo "[INFO] memref.alloc count=${ALLOC_COUNT}, memref.dealloc count=${DEALLOC_COUNT}"
