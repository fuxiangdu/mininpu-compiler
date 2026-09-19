#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPT="${PROJECT_ROOT}/build/bin/mininpu-opt"
RESULT_DIR="${PROJECT_ROOT}/results/v8"
TRANSLATE="${MLIR_TRANSLATE:-$(command -v mlir-translate-18 || command -v mlir-translate || true)}"
CLANG="${CLANG:-$(command -v clang-18 || command -v clang || true)}"
LLC="${LLC:-$(command -v llc-18 || command -v llc || true)}"
CC_BIN="${CC_BIN:-$(command -v cc || command -v gcc || true)}"

FRONTEND_PIPELINE='builtin.module(mininpu-lower-to-linalg,empty-tensor-to-alloc-tensor,one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline,canonicalize,cse)'
LOOP_PIPELINE='builtin.module(func.func(convert-linalg-to-loops,lower-affine,canonicalize,cse))'
LLVM_PIPELINE='builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),convert-cf-to-llvm,expand-strided-metadata,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)'

test -x "${OPT}" || {
  echo "[ERROR] mininpu-opt is missing; run scripts/01_build.sh first"
  exit 1
}

mkdir -p "${RESULT_DIR}"

"${OPT}" "${PROJECT_ROOT}/test/batch_matmul_lowering.mlir" \
  --mininpu-lower-to-linalg \
  --verify-each \
  -o "${RESULT_DIR}/batch_matmul_lowered.mlir"

grep -Fq "linalg.batch_matmul" \
    "${RESULT_DIR}/batch_matmul_lowered.mlir" || {
  echo "[ERROR] equal-batch lowering did not produce linalg.batch_matmul"
  exit 1
}

GENERIC_COUNT=$(grep -Fc "linalg.generic" \
  "${RESULT_DIR}/batch_matmul_lowered.mlir")
if [ "${GENERIC_COUNT}" -ne 2 ]; then
  echo "[ERROR] expected two broadcast linalg.generic ops, found ${GENERIC_COUNT}"
  exit 1
fi

for map_result in "(0, d1, d3)" "(0, d3, d2)"; do
  grep -Fq -- "-> ${map_result}" \
      "${RESULT_DIR}/batch_matmul_lowered.mlir" || {
    echo "[ERROR] broadcast indexing map is missing: ${map_result}"
    exit 1
  }
done

if grep -Fq '"mininpu.batch_matmul"' \
    "${RESULT_DIR}/batch_matmul_lowered.mlir"; then
  echo "[ERROR] a MiniNPU BatchMatMul survived conversion"
  exit 1
fi

if "${OPT}" "${PROJECT_ROOT}/test/batch_matmul_dynamic.mlir" \
    --mininpu-lower-to-linalg -o /dev/null \
    >"${RESULT_DIR}/batch_matmul_dynamic.log" 2>&1; then
  echo "[ERROR] dynamic BatchMatMul lowering unexpectedly succeeded"
  exit 1
fi

grep -Fq "failed to legalize operation" \
    "${RESULT_DIR}/batch_matmul_dynamic.log" || {
  echo "[ERROR] expected dynamic BatchMatMul rejection was not found"
  sed -n '1,120p' "${RESULT_DIR}/batch_matmul_dynamic.log"
  exit 1
}

test -x "${TRANSLATE}" || {
  echo "[ERROR] mlir-translate is missing"
  exit 1
}

"${OPT}" "${PROJECT_ROOT}/test/batch_matmul_execution.mlir" \
  "--pass-pipeline=${FRONTEND_PIPELINE}" --verify-each \
  -o "${RESULT_DIR}/batch_matmul_bufferized.mlir"
"${OPT}" "${RESULT_DIR}/batch_matmul_bufferized.mlir" \
  "--pass-pipeline=${LOOP_PIPELINE}" --verify-each \
  -o "${RESULT_DIR}/batch_matmul_loops.mlir"
"${OPT}" "${RESULT_DIR}/batch_matmul_loops.mlir" \
  "--pass-pipeline=${LLVM_PIPELINE}" --verify-each \
  -o "${RESULT_DIR}/batch_matmul_llvm.mlir"
"${TRANSLATE}" --mlir-to-llvmir \
  "${RESULT_DIR}/batch_matmul_llvm.mlir" \
  -o "${RESULT_DIR}/batch_matmul.ll"

if test -x "${CLANG}"; then
  "${CLANG}" -O2 \
    "${RESULT_DIR}/batch_matmul.ll" \
    "${PROJECT_ROOT}/runtime/check_f32.c" -lm \
    -o "${RESULT_DIR}/batch_matmul_runner"
elif test -x "${LLC}" && test -x "${CC_BIN}"; then
  "${LLC}" -filetype=obj -relocation-model=pic \
    "${RESULT_DIR}/batch_matmul.ll" \
    -o "${RESULT_DIR}/batch_matmul.o"
  "${CC_BIN}" -O2 -c "${PROJECT_ROOT}/runtime/check_f32.c" \
    -o "${RESULT_DIR}/check_f32.o"
  "${CC_BIN}" "${RESULT_DIR}/batch_matmul.o" \
    "${RESULT_DIR}/check_f32.o" -lm \
    -o "${RESULT_DIR}/batch_matmul_runner"
else
  echo "[ERROR] native linking needs Clang, or both llc and a C compiler"
  exit 1
fi

set +e
"${RESULT_DIR}/batch_matmul_runner" 2>&1 | \
  tee "${RESULT_DIR}/batch_matmul_runtime.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [ "${RUN_STATUS}" -ne 0 ]; then
  echo "[ERROR] BatchMatMul runner reported ${RUN_STATUS} mismatches"
  exit 1
fi

PASS_COUNT=$(grep -Fc '[PASS] output[' \
  "${RESULT_DIR}/batch_matmul_runtime.log")
if [ "${PASS_COUNT}" -ne 8 ]; then
  echo "[ERROR] expected eight checked outputs, found ${PASS_COUNT}"
  exit 1
fi

echo "===== BatchMatMul structured lowering ====="
grep -n -E 'linalg\.(batch_matmul|generic)|indexing_maps' \
  "${RESULT_DIR}/batch_matmul_lowered.mlir"
echo "[PASS] Equal batches lower to linalg.batch_matmul"
echo "[PASS] Unit batch broadcasting lowers to affine linalg.generic maps"
echo "[PASS] Dynamic BatchMatMul lowering is rejected explicitly"
echo "[PASS] Broadcast BatchMatMul matched all eight native reference outputs"
