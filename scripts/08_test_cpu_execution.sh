#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPT="${PROJECT_ROOT}/build/bin/mininpu-opt"
TRANSLATE=/usr/bin/mlir-translate-18
CLANG=/usr/bin/clang-18
RESULT_DIR="${PROJECT_ROOT}/results/v6"

FRONTEND_PIPELINE='builtin.module(mininpu-fuse-linear-relu,mininpu-plan-tiles,mininpu-lower-to-linalg,empty-tensor-to-alloc-tensor,one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline,canonicalize,cse)'
LOOP_PIPELINE='builtin.module(func.func(convert-linalg-to-loops,lower-affine,canonicalize,cse))'
LLVM_PIPELINE='builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),convert-cf-to-llvm,expand-strided-metadata,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)'

for tool in "${OPT}" "${TRANSLATE}" "${CLANG}"; do
  test -x "${tool}" || {
    echo "[ERROR] required executable is missing: ${tool}"
    exit 1
  }
done

mkdir -p "${RESULT_DIR}"

"${OPT}" "${PROJECT_ROOT}/test/cpu_execution.mlir" \
  "--pass-pipeline=${FRONTEND_PIPELINE}" \
  --verify-each \
  -o "${RESULT_DIR}/bufferized.mlir"

grep -Fq 'memref<' "${RESULT_DIR}/bufferized.mlir" || {
  echo "[ERROR] v6 frontend did not produce MemRef IR"
  exit 1
}
grep -Fq 'mininpu.tile_m = 2' "${RESULT_DIR}/bufferized.mlir" || {
  echo "[ERROR] tile-planning metadata is missing before loop lowering"
  exit 1
}

"${OPT}" "${RESULT_DIR}/bufferized.mlir" \
  "--pass-pipeline=${LOOP_PIPELINE}" \
  --verify-each \
  -o "${RESULT_DIR}/loops.mlir"

grep -Fq 'scf.for' "${RESULT_DIR}/loops.mlir" || {
  echo "[ERROR] Linalg operations were not materialized as SCF loops"
  exit 1
}
if grep -Eq 'linalg\.|"mininpu\.' "${RESULT_DIR}/loops.mlir"; then
  echo "[ERROR] structured or MiniNPU operations survived loop lowering"
  exit 1
fi

"${OPT}" "${RESULT_DIR}/loops.mlir" \
  "--pass-pipeline=${LLVM_PIPELINE}" \
  --verify-each \
  -o "${RESULT_DIR}/llvm_dialect.mlir"

grep -Fq 'llvm.func @main' "${RESULT_DIR}/llvm_dialect.mlir" || {
  echo "[ERROR] LLVM dialect main function is missing"
  exit 1
}
grep -Fq 'llvm.call @check_f32' "${RESULT_DIR}/llvm_dialect.mlir" || {
  echo "[ERROR] runtime check call was not lowered to the LLVM dialect"
  exit 1
}
if grep -Eq 'scf\.|func\.|memref\.|arith\.|linalg\.|"mininpu\.' \
    "${RESULT_DIR}/llvm_dialect.mlir"; then
  echo "[ERROR] a non-LLVM executable operation survived final lowering"
  exit 1
fi

"${TRANSLATE}" --mlir-to-llvmir \
  "${RESULT_DIR}/llvm_dialect.mlir" \
  -o "${RESULT_DIR}/program.ll"

grep -Fq 'define i32 @main()' "${RESULT_DIR}/program.ll" || {
  echo "[ERROR] translated LLVM IR does not define a native main"
  exit 1
}

"${CLANG}" -O2 \
  "${RESULT_DIR}/program.ll" \
  "${PROJECT_ROOT}/runtime/check_f32.c" \
  -lm \
  -o "${RESULT_DIR}/mininpu_v6_runner"

set +e
"${RESULT_DIR}/mininpu_v6_runner" \
  2>&1 | tee "${RESULT_DIR}/runtime.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [ "${RUN_STATUS}" -ne 0 ]; then
  echo "[ERROR] native runner reported ${RUN_STATUS} mismatched outputs"
  exit 1
fi

PASS_COUNT=$(grep -Fc '[PASS] output[' "${RESULT_DIR}/runtime.log")
if [ "${PASS_COUNT}" -ne 4 ]; then
  echo "[ERROR] expected four checked output elements, found ${PASS_COUNT}"
  exit 1
fi

echo "===== SCF loop evidence ====="
grep -n -m 12 -E 'scf\.for|memref\.(load|store)' "${RESULT_DIR}/loops.mlir"
echo "===== LLVM dialect evidence ====="
grep -n -m 12 -E 'llvm\.func @main|llvm\.call @check_f32|llvm\.(load|store)' \
  "${RESULT_DIR}/llvm_dialect.mlir"
echo "[PASS] Linalg operations lowered to explicit SCF loops"
echo "[PASS] SCF, MemRef, Arith and Func lowered to the LLVM dialect"
echo "[PASS] LLVM dialect translated to LLVM IR and linked by Clang"
echo "[PASS] Native execution matched all four reference outputs"
