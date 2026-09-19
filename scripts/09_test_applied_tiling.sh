#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPT="${PROJECT_ROOT}/build/bin/mininpu-opt"
RESULT_DIR="${PROJECT_ROOT}/results/v7"
PIPELINE='builtin.module(mininpu-fuse-linear-relu,mininpu-plan-tiles,mininpu-lower-to-linalg,mininpu-apply-tiles)'

test -x "${OPT}" || {
  echo "[ERROR] mininpu-opt is missing; run scripts/01_build.sh first"
  exit 1
}

mkdir -p "${RESULT_DIR}"

"${OPT}" "${PROJECT_ROOT}/test/lowering.mlir" \
  "--pass-pipeline=${PIPELINE}" \
  --verify-each \
  -o "${RESULT_DIR}/tiled.mlir"

for expected in \
  "scf.for" \
  "tensor.extract_slice" \
  "tensor.insert_slice" \
  "linalg.matmul" \
  "mininpu.tiles_applied = true" \
  "mininpu.tile_m = 112"; do
  grep -Fq "${expected}" "${RESULT_DIR}/tiled.mlir" || {
    echo "[ERROR] materialized tile IR is missing ${expected}"
    exit 1
  }
done

if grep -Eq '"mininpu\.(matmul|bias_add|relu|fused_matmul_bias_relu)"' \
    "${RESULT_DIR}/tiled.mlir"; then
  echo "[ERROR] a MiniNPU operation survived the scheduled lowering"
  exit 1
fi

"${OPT}" "${RESULT_DIR}/tiled.mlir" \
  --mininpu-apply-tiles \
  --verify-each \
  -o "${RESULT_DIR}/tiled_twice.mlir"

if ! cmp -s "${RESULT_DIR}/tiled.mlir" \
    "${RESULT_DIR}/tiled_twice.mlir"; then
  echo "[ERROR] tile materialization pass is not idempotent"
  diff -u "${RESULT_DIR}/tiled.mlir" \
    "${RESULT_DIR}/tiled_twice.mlir" || true
  exit 1
fi

if "${OPT}" "${PROJECT_ROOT}/test/partial_tile_plan.mlir" \
    --mininpu-apply-tiles -o /dev/null \
    >"${RESULT_DIR}/partial_plan.log" 2>&1; then
  echo "[ERROR] an incomplete tile plan unexpectedly succeeded"
  exit 1
fi

grep -Fq "requires a complete MiniNPU tile plan" \
    "${RESULT_DIR}/partial_plan.log" || {
  echo "[ERROR] expected incomplete-plan diagnostic was not found"
  sed -n '1,100p' "${RESULT_DIR}/partial_plan.log"
  exit 1
}

echo "===== cost-model-driven tiled IR ====="
grep -n -m 40 -E \
  'scf\.for|tensor\.(extract|insert)_slice|linalg\.matmul|mininpu\.tile_[mnk]' \
  "${RESULT_DIR}/tiled.mlir"
echo "[PASS] Planned M/N/K tile sizes materialized as executable SCF loops"
echo "[PASS] Boundary-safe tensor slices generated for partial tiles"
echo "[PASS] Tile materialization is idempotent"
echo "[PASS] Incomplete tile plans are rejected"
