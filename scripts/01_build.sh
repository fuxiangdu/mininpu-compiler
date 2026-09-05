#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${PROJECT_ROOT}/build"

/usr/bin/cmake --fresh \
  -S "${PROJECT_ROOT}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm

/usr/bin/cmake --build "${BUILD_DIR}" --parallel 2

test -x "${BUILD_DIR}/bin/mininpu-opt" || {
  echo "[ERROR] mininpu-opt was not generated"
  exit 1
}

echo "[PASS] Built ${BUILD_DIR}/bin/mininpu-opt"

