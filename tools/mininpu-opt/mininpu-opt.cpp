//===- mininpu-opt.cpp - MiniNPU optimizer driver ----------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Registers the out-of-tree MiniNPU dialect, the upstream dialects used by the
// lowering pipeline, and all MiniNPU optimization and lowering passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "MiniNPU/Dialect/MiniNPU/MiniNPUDialect.h"
#include "MiniNPU/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mininpu::registerMiniNPUPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mininpu::MiniNPUDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MiniNPU optimizer driver\n", registry));
}
