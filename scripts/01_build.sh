#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${PROJECT_ROOT}/build"

CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
NINJA_BIN="${NINJA_BIN:-$(command -v ninja || true)}"
C_COMPILER="${C_COMPILER:-$(command -v clang-18 || command -v clang || command -v gcc || true)}"
CXX_COMPILER="${CXX_COMPILER:-$(command -v clang++-18 || command -v clang++ || command -v g++ || true)}"
MLIR_DIR="${MLIR_DIR:-/usr/lib/llvm-18/lib/cmake/mlir}"
LLVM_DIR="${LLVM_DIR:-/usr/lib/llvm-18/lib/cmake/llvm}"

for tool in "${CMAKE_BIN}" "${NINJA_BIN}" "${C_COMPILER}" "${CXX_COMPILER}"; do
  test -x "${tool}" || {
    echo "[ERROR] required build tool is missing: ${tool:-<not found>}"
    exit 1
  }
done

for package_dir in "${MLIR_DIR}" "${LLVM_DIR}"; do
  test -d "${package_dir}" || {
    echo "[ERROR] CMake package directory is missing: ${package_dir}"
    echo "        Set MLIR_DIR and LLVM_DIR to your LLVM build or installation."
    exit 1
  }
done

"${CMAKE_BIN}" --fresh \
  -S "${PROJECT_ROOT}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_MAKE_PROGRAM="${NINJA_BIN}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}"

"${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel "${BUILD_JOBS:-2}"

test -x "${BUILD_DIR}/bin/mininpu-opt" || {
  echo "[ERROR] mininpu-opt was not generated"
  exit 1
}

echo "[PASS] Built ${BUILD_DIR}/bin/mininpu-opt"
