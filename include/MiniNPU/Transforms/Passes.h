//===- Passes.h - MiniNPU transformation passes ----------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef MININPU_TRANSFORMS_PASSES_H
#define MININPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <memory>

namespace mininpu {

std::unique_ptr<mlir::Pass> createFuseMatMulBiasReluPass();
std::unique_ptr<mlir::Pass> createPlanTilesPass();
std::unique_ptr<mlir::Pass> createLowerToLinalgPass();
std::unique_ptr<mlir::Pass> createApplyTilesPass();
std::unique_ptr<mlir::Pass> createPlanAttentionPass();
void registerPlanTilesPass();
void registerLowerToLinalgPass();
void registerApplyTilesPass();
void registerPlanAttentionPass();
void registerMiniNPUPasses();

} // namespace mininpu

#endif // MININPU_TRANSFORMS_PASSES_H
