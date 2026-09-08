#!/bin/bash
set -euo pipefail

LLVM_ROOT=/usr/lib/llvm-18

echo "[INFO] OS: $(. /etc/os-release && echo "${PRETTY_NAME}")"
echo "[INFO] architecture: $(uname -m)"
echo "[INFO] memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "[INFO] available disk: $(df -h / | awk 'NR == 2 {print $4}')"

for path in \
  /usr/bin/clang-18 \
  /usr/bin/clang++-18 \
  /usr/bin/cmake \
  /usr/bin/ninja \
  /usr/bin/mlir-opt-18 \
  /usr/bin/mlir-tblgen-18 \
  /usr/bin/mlir-translate-18 \
  "${LLVM_ROOT}/lib/cmake/mlir/MLIRConfig.cmake" \
  "${LLVM_ROOT}/lib/cmake/llvm/LLVMConfig.cmake"
do
  if [ ! -e "${path}" ]; then
    echo "[ERROR] required path is missing: ${path}"
    exit 1
  fi
done

echo "[INFO] clang: $(/usr/bin/clang++-18 --version | head -n 1)"
echo "[INFO] MLIR: $(/usr/bin/mlir-opt-18 --version | head -n 1)"
echo "[PASS] MiniNPU v6 preflight checks passed"
