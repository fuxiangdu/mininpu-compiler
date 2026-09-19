#!/bin/bash
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "[ERROR] sudo is required to install system packages"
  exit 1
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  build-essential \
  clang-18 \
  cmake \
  git \
  libmlir-18-dev \
  llvm-18-dev \
  mlir-18-tools \
  ninja-build \
  python3

echo "[PASS] Installed the Ubuntu 24.04 MiniNPU build dependencies"
echo "[INFO] Run: bash scripts/run_v7.sh"
