#!/bin/bash
set -euo pipefail

LLVM_ROOT="${LLVM_ROOT:-/usr/lib/llvm-18}"
CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
NINJA_BIN="${NINJA_BIN:-$(command -v ninja || true)}"
CXX_COMPILER="${CXX_COMPILER:-$(command -v clang++-18 || command -v clang++ || command -v g++ || true)}"
MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt-18 || command -v mlir-opt || true)}"
MLIR_TBLGEN="${MLIR_TBLGEN:-$(command -v mlir-tblgen-18 || command -v mlir-tblgen || true)}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-$(command -v mlir-translate-18 || command -v mlir-translate || true)}"
MLIR_DIR="${MLIR_DIR:-${LLVM_ROOT}/lib/cmake/mlir}"
LLVM_DIR="${LLVM_DIR:-${LLVM_ROOT}/lib/cmake/llvm}"

echo "[INFO] OS: $(. /etc/os-release && echo "${PRETTY_NAME}")"
echo "[INFO] architecture: $(uname -m)"
echo "[INFO] memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "[INFO] available disk: $(df -h / | awk 'NR == 2 {print $4}')"

for path in \
  "${CMAKE_BIN}" \
  "${NINJA_BIN}" \
  "${CXX_COMPILER}" \
  "${MLIR_OPT}" \
  "${MLIR_TBLGEN}" \
  "${MLIR_TRANSLATE}" \
  "${MLIR_DIR}/MLIRConfig.cmake" \
  "${LLVM_DIR}/LLVMConfig.cmake"
do
  if [ -z "${path}" ] || [ ! -e "${path}" ]; then
    echo "[ERROR] required path is missing: ${path:-<not found>}"
    exit 1
  fi
done

echo "[INFO] C++ compiler: $("${CXX_COMPILER}" --version | head -n 1)"
echo "[INFO] MLIR: $("${MLIR_OPT}" --version | head -n 1)"
echo "[PASS] MiniNPU v7 preflight checks passed"
