//===- LowerToLinalg.cpp - Lower MiniNPU ops to standard MLIR --*- C++ -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "MiniNPU/Transforms/Passes.h"

#include "MiniNPU/Dialect/MiniNPU/MiniNPUDialect.h"
#include "MiniNPU/Dialect/MiniNPU/MiniNPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

#include <memory>

using namespace mlir;

namespace {

class LowerFusedLinearPattern final
    : public OpConversionPattern<mininpu::FusedMatMulBiasReluOp> {
public:
  using OpConversionPattern<
      mininpu::FusedMatMulBiasReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mininpu::FusedMatMulBiasReluOp operation,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(operation.getOutput().getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          operation, "v4 lowering requires a statically shaped ranked tensor");

    Type elementType = resultType.getElementType();
    if (!isa<FloatType>(elementType))
      return rewriter.notifyMatchFailure(
          operation, "v4 lowering currently supports floating-point tensors");

    Location location = operation.getLoc();

    auto empty = rewriter.create<tensor::EmptyOp>(
        location, resultType.getShape(), elementType);
    auto zero = rewriter.create<arith::ConstantOp>(
        location, rewriter.getFloatAttr(elementType, 0.0));
    auto filled = rewriter.create<linalg::FillOp>(
        location, TypeRange{resultType}, ValueRange{zero.getResult()},
        ValueRange{empty.getResult()});

    auto matmul = rewriter.create<linalg::MatmulOp>(
        location, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{filled->getResult(0)});

    // Tile-planning attributes describe the matrix multiplication and remain
    // visible after the custom operation has been eliminated.
    for (NamedAttribute attribute : operation->getAttrs()) {
      StringRef name = attribute.getName().getValue();
      if (name.take_front(8) == "mininpu.")
        matmul->setAttr(attribute.getName(), attribute.getValue());
    }

    MLIRContext *context = rewriter.getContext();
    AffineExpr row = rewriter.getAffineDimExpr(0);
    AffineExpr column = rewriter.getAffineDimExpr(1);
    AffineMap biasMap = AffineMap::get(2, 0, {column}, context);
    AffineMap outputMap = AffineMap::get(2, 0, {row, column}, context);
    llvm::SmallVector<AffineMap> indexingMaps{biasMap, outputMap};
    llvm::SmallVector<utils::IteratorType> iteratorTypes(
        2, utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        location, TypeRange{resultType}, ValueRange{adaptor.getBias()},
        ValueRange{matmul->getResult(0)}, indexingMaps, iteratorTypes,
        [elementType](OpBuilder &builder, Location nestedLocation,
                      ValueRange arguments) {
          // The first argument is the broadcast bias input; the second is the
          // current matrix output supplied through the DPS output operand.
          auto biased = builder.create<arith::AddFOp>(
              nestedLocation, arguments[1], arguments[0]);
          auto nestedZero = builder.create<arith::ConstantOp>(
              nestedLocation, builder.getFloatAttr(elementType, 0.0));
          auto activated = builder.create<arith::MaximumFOp>(
              nestedLocation, biased.getResult(), nestedZero.getResult());
          builder.create<linalg::YieldOp>(
              nestedLocation, ValueRange{activated.getResult()});
        });

    rewriter.replaceOp(operation, generic.getResults());
    return success();
  }
};

class LowerToLinalgPass final
    : public PassWrapper<LowerToLinalgPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLinalgPass)

  StringRef getArgument() const final { return "mininpu-lower-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower planned MiniNPU fused operations to Tensor/Linalg/Arith";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, tensor::TensorDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<mininpu::MiniNPUDialect>();

    RewritePatternSet patterns(&context);
    patterns.add<LowerFusedLinearPattern>(&context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mininpu::createLowerToLinalgPass() {
  return std::make_unique<LowerToLinalgPass>();
}

void mininpu::registerLowerToLinalgPass() {
  PassRegistration<LowerToLinalgPass>();
}
