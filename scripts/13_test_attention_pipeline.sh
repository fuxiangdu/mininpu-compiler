#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPT="${PROJECT_ROOT}/build/bin/mininpu-opt"
RESULT_DIR="${PROJECT_ROOT}/results/v9"
TRANSLATE="${MLIR_TRANSLATE:-$(command -v mlir-translate-18 || command -v mlir-translate || true)}"
CLANG="${CLANG:-$(command -v clang-18 || command -v clang || true)}"
LLC="${LLC:-$(command -v llc-18 || command -v llc || true)}"
CC_BIN="${CC_BIN:-$(command -v cc || command -v gcc || true)}"

FRONTEND_PIPELINE='builtin.module(mininpu-plan-attention,mininpu-lower-to-linalg,empty-tensor-to-alloc-tensor,one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline,canonicalize,cse)'
LOOP_PIPELINE='builtin.module(func.func(convert-linalg-to-loops,lower-affine,canonicalize,cse))'
LLVM_PIPELINE='builtin.module(func.func(convert-scf-to-cf,convert-math-to-llvm,convert-arith-to-llvm),convert-cf-to-llvm,expand-strided-metadata,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)'

test -x "${OPT}" || {
  echo "[ERROR] mininpu-opt is missing; run scripts/01_build.sh first"
  exit 1
}

mkdir -p "${RESULT_DIR}"

"${OPT}" "${PROJECT_ROOT}/test/attention_plan.mlir" \
  --mininpu-plan-attention --verify-each \
  -o "${RESULT_DIR}/attention_planned.mlir"

for expected in \
    'mininpu.attention_graph_count = 2 : i64' \
    'mininpu.attention_plan = "streaming-fusion-candidate"' \
    'mininpu.attention_plan = "materialize-shared-boundaries"' \
    'mininpu.estimated_traffic_saved_bytes = 96 : i64' \
    'mininpu.estimated_traffic_saved_bytes = 64 : i64' \
    'mininpu.fuse_qk_softmax = false'; do
  grep -Fq "${expected}" "${RESULT_DIR}/attention_planned.mlir" || {
    echo "[ERROR] attention plan is missing: ${expected}"
    exit 1
  }
done

for role in qk softmax pv output_norm; do
  grep -Fq "mininpu.attention_role = \"${role}\"" \
      "${RESULT_DIR}/attention_planned.mlir" || {
    echo "[ERROR] attention graph role is missing: ${role}"
    exit 1
  }
done

if sed -n '/func.func @non_last_axis/,/^  }/p' \
    "${RESULT_DIR}/attention_planned.mlir" | \
    grep -Fq 'mininpu.attention_role'; then
  echo "[ERROR] non-final-axis Softmax was misidentified as attention"
  exit 1
fi

"${OPT}" "${RESULT_DIR}/attention_planned.mlir" \
  --mininpu-plan-attention --verify-each \
  -o "${RESULT_DIR}/attention_replanned.mlir"
diff -u "${RESULT_DIR}/attention_planned.mlir" \
  "${RESULT_DIR}/attention_replanned.mlir"

test -x "${TRANSLATE}" || {
  echo "[ERROR] mlir-translate is missing"
  exit 1
}

"${OPT}" "${PROJECT_ROOT}/test/attention_execution.mlir" \
  "--pass-pipeline=${FRONTEND_PIPELINE}" --verify-each \
  -o "${RESULT_DIR}/attention_bufferized.mlir"
"${OPT}" "${RESULT_DIR}/attention_bufferized.mlir" \
  "--pass-pipeline=${LOOP_PIPELINE}" --verify-each \
  -o "${RESULT_DIR}/attention_loops.mlir"
"${OPT}" "${RESULT_DIR}/attention_loops.mlir" \
  "--pass-pipeline=${LLVM_PIPELINE}" --verify-each \
  -o "${RESULT_DIR}/attention_llvm.mlir"
"${TRANSLATE}" --mlir-to-llvmir \
  "${RESULT_DIR}/attention_llvm.mlir" \
  -o "${RESULT_DIR}/attention.ll"

if test -x "${CLANG}"; then
  "${CLANG}" -O2 "${RESULT_DIR}/attention.ll" \
    "${PROJECT_ROOT}/runtime/check_f32.c" -lm \
    -o "${RESULT_DIR}/attention_runner"
elif test -x "${LLC}" && test -x "${CC_BIN}"; then
  "${LLC}" -filetype=obj -relocation-model=pic \
    "${RESULT_DIR}/attention.ll" -o "${RESULT_DIR}/attention.o"
  "${CC_BIN}" -O2 -c "${PROJECT_ROOT}/runtime/check_f32.c" \
    -o "${RESULT_DIR}/check_f32.o"
  "${CC_BIN}" "${RESULT_DIR}/attention.o" \
    "${RESULT_DIR}/check_f32.o" -lm \
    -o "${RESULT_DIR}/attention_runner"
else
  echo "[ERROR] native linking needs Clang, or both llc and a C compiler"
  exit 1
fi

set +e
"${RESULT_DIR}/attention_runner" 2>&1 | \
  tee "${RESULT_DIR}/attention_runtime.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [ "${RUN_STATUS}" -ne 0 ]; then
  echo "[ERROR] attention runner reported ${RUN_STATUS} mismatches"
  exit 1
fi

PASS_COUNT=$(grep -Fc '[PASS] output[' \
  "${RESULT_DIR}/attention_runtime.log")
if [ "${PASS_COUNT}" -ne 4 ]; then
  echo "[ERROR] expected four checked outputs, found ${PASS_COUNT}"
  exit 1
fi

echo "===== Cross-operation attention plan ====="
grep -n -E 'attention_(graph|plan|role)|fuse_|traffic_saved' \
  "${RESULT_DIR}/attention_planned.mlir"
echo "[PASS] QK-Softmax-PV-RMSNorm graph roles were identified"
echo "[PASS] Single-use graph selected a streaming fusion candidate"
echo "[PASS] Shared score values forced a safe materialization boundary"
echo "[PASS] Non-final-axis Softmax graphs were excluded"
echo "[PASS] Static intermediate traffic savings were estimated"
echo "[PASS] Attention planning is idempotent"
echo "[PASS] End-to-end attention graph matched four native outputs"
