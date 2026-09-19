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
LLVM_PIPELINE='builtin.module(func.func(convert-scf-to-cf,convert-math-to-llvm,convert-arith-to-llvm),convert-cf-to-llvm,expand-strided-metadata,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)'

test -x "${OPT}" || {
  echo "[ERROR] mininpu-opt is missing; run scripts/01_build.sh first"
  exit 1
}

mkdir -p "${RESULT_DIR}"

"${OPT}" "${PROJECT_ROOT}/test/rms_norm_lowering.mlir" \
  --mininpu-lower-to-linalg --verify-each \
  -o "${RESULT_DIR}/rms_norm_lowered.mlir"

GENERIC_COUNT=$(grep -Fc "linalg.generic" \
  "${RESULT_DIR}/rms_norm_lowered.mlir")
if [ "${GENERIC_COUNT}" -ne 9 ]; then
  echo "[ERROR] expected nine linalg.generic stages, found ${GENERIC_COUNT}"
  exit 1
fi

for operation in "arith.mulf" "arith.addf" "arith.divf" "math.rsqrt"; do
  grep -Fq "${operation}" "${RESULT_DIR}/rms_norm_lowered.mlir" || {
    echo "[ERROR] RMSNorm stage is missing: ${operation}"
    exit 1
  }
done

for cast_operation in "arith.extf" "arith.truncf"; do
  grep -Fq "${cast_operation}" "${RESULT_DIR}/rms_norm_lowered.mlir" || {
    echo "[ERROR] FP16/FP32 accumulation cast is missing: ${cast_operation}"
    exit 1
  }
done

grep -Fq 'iterator_types = ["parallel", "parallel", "reduction"]' \
    "${RESULT_DIR}/rms_norm_lowered.mlir" || {
  echo "[ERROR] final-axis reduction iterator was not materialized"
  exit 1
}

if grep -Fq '"mininpu.rms_norm"' \
    "${RESULT_DIR}/rms_norm_lowered.mlir"; then
  echo "[ERROR] a MiniNPU RMSNorm survived conversion"
  exit 1
fi

if "${OPT}" "${PROJECT_ROOT}/test/rms_norm_dynamic.mlir" \
    --mininpu-lower-to-linalg -o /dev/null \
    >"${RESULT_DIR}/rms_norm_dynamic.log" 2>&1; then
  echo "[ERROR] dynamic RMSNorm lowering unexpectedly succeeded"
  exit 1
fi

grep -Fq "failed to legalize operation" \
    "${RESULT_DIR}/rms_norm_dynamic.log" || {
  echo "[ERROR] expected dynamic RMSNorm rejection was not found"
  sed -n '1,120p' "${RESULT_DIR}/rms_norm_dynamic.log"
  exit 1
}

test -x "${TRANSLATE}" || {
  echo "[ERROR] mlir-translate is missing"
  exit 1
}

"${OPT}" "${PROJECT_ROOT}/test/rms_norm_execution.mlir" \
  "--pass-pipeline=${FRONTEND_PIPELINE}" --verify-each \
  -o "${RESULT_DIR}/rms_norm_bufferized.mlir"
"${OPT}" "${RESULT_DIR}/rms_norm_bufferized.mlir" \
  "--pass-pipeline=${LOOP_PIPELINE}" --verify-each \
  -o "${RESULT_DIR}/rms_norm_loops.mlir"
"${OPT}" "${RESULT_DIR}/rms_norm_loops.mlir" \
  "--pass-pipeline=${LLVM_PIPELINE}" --verify-each \
  -o "${RESULT_DIR}/rms_norm_llvm.mlir"
"${TRANSLATE}" --mlir-to-llvmir \
  "${RESULT_DIR}/rms_norm_llvm.mlir" \
  -o "${RESULT_DIR}/rms_norm.ll"

if test -x "${CLANG}"; then
  "${CLANG}" -O2 "${RESULT_DIR}/rms_norm.ll" \
    "${PROJECT_ROOT}/runtime/check_f32.c" -lm \
    -o "${RESULT_DIR}/rms_norm_runner"
elif test -x "${LLC}" && test -x "${CC_BIN}"; then
  "${LLC}" -filetype=obj -relocation-model=pic \
    "${RESULT_DIR}/rms_norm.ll" -o "${RESULT_DIR}/rms_norm.o"
  "${CC_BIN}" -O2 -c "${PROJECT_ROOT}/runtime/check_f32.c" \
    -o "${RESULT_DIR}/check_f32.o"
  "${CC_BIN}" "${RESULT_DIR}/rms_norm.o" \
    "${RESULT_DIR}/check_f32.o" -lm \
    -o "${RESULT_DIR}/rms_norm_runner"
else
  echo "[ERROR] native linking needs Clang, or both llc and a C compiler"
  exit 1
fi

set +e
"${RESULT_DIR}/rms_norm_runner" 2>&1 | \
  tee "${RESULT_DIR}/rms_norm_runtime.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [ "${RUN_STATUS}" -ne 0 ]; then
  echo "[ERROR] RMSNorm runner reported ${RUN_STATUS} mismatches"
  exit 1
fi

PASS_COUNT=$(grep -Fc '[PASS] output[' \
  "${RESULT_DIR}/rms_norm_runtime.log")
if [ "${PASS_COUNT}" -ne 8 ]; then
  echo "[ERROR] expected eight checked outputs, found ${PASS_COUNT}"
  exit 1
fi

echo "===== FP32-accumulating RMSNorm lowering ====="
grep -n -E 'linalg\.generic|arith\.(extf|truncf|mulf|addf|divf)|math\.rsqrt' \
  "${RESULT_DIR}/rms_norm_lowered.mlir"
echo "[PASS] RMSNorm lowers to square-sum, reciprocal-RMS and scale stages"
echo "[PASS] FP16 inputs use explicit FP32 accumulation and result casts"
echo "[PASS] Rank-1 weights broadcast along the final dimension"
echo "[PASS] Dynamic RMSNorm lowering is rejected explicitly"
echo "[PASS] RMSNorm matched all eight native reference outputs"
