//===- FuseMatMulBiasRelu.cpp - Fuse a MiniNPU linear block ----*- C++ -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "MiniNPU/Transforms/Passes.h"

#include "MiniNPU/Dialect/MiniNPU/MiniNPUOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

class FuseMatMulBiasReluPattern final
    : public OpRewritePattern<mininpu::ReluOp> {
public:
  using OpRewritePattern<mininpu::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mininpu::ReluOp relu,
                                PatternRewriter &rewriter) const override {
    auto biasAdd = relu.getInput().getDefiningOp<mininpu::BiasAddOp>();
    if (!biasAdd)
      return rewriter.notifyMatchFailure(relu, "input is not mininpu.bias_add");

    auto matmul = biasAdd.getInput().getDefiningOp<mininpu::MatMulOp>();
    if (!matmul)
      return rewriter.notifyMatchFailure(biasAdd,
                                         "input is not mininpu.matmul");

    if (!biasAdd.getOutput().hasOneUse())
      return rewriter.notifyMatchFailure(
          biasAdd, "bias_add result has more than one user");
    if (!matmul.getOutput().hasOneUse())
      return rewriter.notifyMatchFailure(matmul,
                                         "matmul result has more than one user");

    rewriter.replaceOpWithNewOp<mininpu::FusedMatMulBiasReluOp>(
        relu, relu.getOutput().getType(), matmul.getLhs(), matmul.getRhs(),
        biasAdd.getBias());
    rewriter.eraseOp(biasAdd);
    rewriter.eraseOp(matmul);
    return success();
  }
};

class FuseMatMulBiasReluPass final
    : public PassWrapper<FuseMatMulBiasReluPass,
                         OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMatMulBiasReluPass)

  StringRef getArgument() const final { return "mininpu-fuse-linear-relu"; }
  StringRef getDescription() const final {
    return "Fuse MiniNPU MatMul-BiasAdd-ReLU chains with single-use values";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatMulBiasReluPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mininpu::createFuseMatMulBiasReluPass() {
  return std::make_unique<FuseMatMulBiasReluPass>();
}

void mininpu::registerMiniNPUPasses() {
  PassRegistration<FuseMatMulBiasReluPass>();
  registerPlanTilesPass();
  registerLowerToLinalgPass();
}
