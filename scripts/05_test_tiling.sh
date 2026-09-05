#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPT="${PROJECT_ROOT}/build/bin/mininpu-opt"
RESULT_DIR="${PROJECT_ROOT}/results/v3"
PIPELINE='builtin.module(mininpu-fuse-linear-relu,mininpu-plan-tiles)'

test -x "${OPT}" || {
  echo "[ERROR] mininpu-opt is missing; run scripts/01_build.sh first"
  exit 1
}

mkdir -p "${RESULT_DIR}"

for capacity in 64k 256k; do
  "${OPT}" "${PROJECT_ROOT}/test/tiling_${capacity}.mlir" \
    "--pass-pipeline=${PIPELINE}" \
    -o "${RESULT_DIR}/planned_${capacity}.mlir"
done

python3 "${SCRIPT_DIR}/check_tile_plan.py" \
  "${RESULT_DIR}/planned_64k.mlir" \
  --m 128 --n 512 --k 256 --element-bytes 4 --ub-bytes 65536

python3 "${SCRIPT_DIR}/check_tile_plan.py" \
  "${RESULT_DIR}/planned_256k.mlir" \
  --m 128 --n 512 --k 256 --element-bytes 4 --ub-bytes 262144

PLAN_64K=$(grep -oE 'mininpu.tile_[mnk] = [0-9]+' \
  "${RESULT_DIR}/planned_64k.mlir" | tr '\n' ';')
PLAN_256K=$(grep -oE 'mininpu.tile_[mnk] = [0-9]+' \
  "${RESULT_DIR}/planned_256k.mlir" | tr '\n' ';')
if [ "${PLAN_64K}" = "${PLAN_256K}" ]; then
  echo "[ERROR] 64 KiB and 256 KiB unexpectedly produced identical tiles"
  exit 1
fi

if "${OPT}" "${PROJECT_ROOT}/test/tiling_too_small.mlir" \
    "--pass-pipeline=${PIPELINE}" -o /dev/null \
    >"${RESULT_DIR}/too_small.log" 2>&1; then
  echo "[ERROR] impossible UB configuration unexpectedly succeeded"
  exit 1
fi
grep -Fq "no legal tile fits" "${RESULT_DIR}/too_small.log" || {
  echo "[ERROR] expected no-legal-tile diagnostic was not found"
  sed -n '1,160p' "${RESULT_DIR}/too_small.log"
  exit 1
}

"${OPT}" "${PROJECT_ROOT}/test/tiling_dynamic.mlir" \
  --mininpu-plan-tiles -o "${RESULT_DIR}/dynamic.mlir" \
  2>"${RESULT_DIR}/dynamic.log"
if grep -Fq "mininpu.tile_m" "${RESULT_DIR}/dynamic.mlir"; then
  echo "[ERROR] dynamic shape unexpectedly received a static tile plan"
  exit 1
fi
grep -Fq "skipping compile-time tile planning for dynamic shapes" \
  "${RESULT_DIR}/dynamic.log" || {
  echo "[ERROR] expected dynamic-shape remark was not found"
  exit 1
}

echo "===== 64 KiB plan ====="
sed -n '1,120p' "${RESULT_DIR}/planned_64k.mlir"
echo "===== 256 KiB plan ====="
sed -n '1,120p' "${RESULT_DIR}/planned_256k.mlir"
echo "===== expected impossible-UB rejection ====="
sed -n '1,80p' "${RESULT_DIR}/too_small.log"
echo "[PASS] UB capacity constraint validated"
echo "[PASS] Capacity-sensitive tile selection validated"
echo "[PASS] Dynamic-shape fallback validated"
