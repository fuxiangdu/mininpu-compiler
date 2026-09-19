//===- ApplyTiles.cpp - Materialize MiniNPU tile plans ----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "MiniNPU/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>

using namespace mlir;

namespace {

FailureOr<int64_t> readTileSize(linalg::MatmulOp operation, StringRef name) {
  auto attribute = operation->getAttrOfType<IntegerAttr>(name);
  if (!attribute)
    return failure();
  int64_t value = attribute.getInt();
  if (value <= 0) {
    operation.emitError() << name << " must be a positive integer";
    return failure();
  }
  return value;
}

class ApplyTilesPass final
    : public PassWrapper<ApplyTilesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyTilesPass)

  StringRef getArgument() const final { return "mininpu-apply-tiles"; }
  StringRef getDescription() const final {
    return "Materialize MiniNPU tile plans as tiled Linalg/SCF IR";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    SmallVector<linalg::MatmulOp> plannedOperations;
    getOperation().walk([&](linalg::MatmulOp operation) {
      if (!operation->hasAttr("mininpu.tiles_applied") &&
          (operation->hasAttr("mininpu.tile_m") ||
           operation->hasAttr("mininpu.tile_n") ||
           operation->hasAttr("mininpu.tile_k")))
        plannedOperations.push_back(operation);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::MatmulOp operation : plannedOperations) {
      FailureOr<int64_t> tileM = readTileSize(operation, "mininpu.tile_m");
      FailureOr<int64_t> tileN = readTileSize(operation, "mininpu.tile_n");
      FailureOr<int64_t> tileK = readTileSize(operation, "mininpu.tile_k");
      if (failed(tileM) || failed(tileN) || failed(tileK)) {
        operation.emitError("requires a complete MiniNPU tile plan");
        signalPassFailure();
        return;
      }

      linalg::LinalgTilingOptions options;
      options.setTileSizes({*tileM, *tileN, *tileK});
      rewriter.setInsertionPoint(operation);
      FailureOr<linalg::TiledLinalgOp> tiled =
          linalg::tileLinalgOp(rewriter, operation, options);
      if (failed(tiled)) {
        operation.emitError("failed to materialize the planned tile sizes");
        signalPassFailure();
        return;
      }

      tiled->op->setAttr("mininpu.tiles_applied",
                         rewriter.getBoolAttr(true));
      rewriter.replaceOp(operation, tiled->tensorResults);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mininpu::createApplyTilesPass() {
  return std::make_unique<ApplyTilesPass>();
}

void mininpu::registerApplyTilesPass() {
  PassRegistration<ApplyTilesPass>();
}
