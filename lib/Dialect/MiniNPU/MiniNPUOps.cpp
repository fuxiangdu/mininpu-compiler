//===- MiniNPUOps.cpp - MiniNPU operation definitions ----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "MiniNPU/Dialect/MiniNPU/MiniNPUOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include <cmath>

using namespace mlir;
using namespace mininpu;

namespace {

bool dimensionsCompatible(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

bool broadcastDimensionsCompatible(int64_t lhs, int64_t rhs) {
  return dimensionsCompatible(lhs, rhs) || lhs == 1 || rhs == 1;
}

} // namespace

LogicalResult MatMulOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!lhsType || !rhsType || !outputType)
    return emitOpError("requires ranked tensor operands and result");
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      outputType.getRank() != 2)
    return emitOpError("requires rank-2 lhs, rhs and output tensors");
  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != outputType.getElementType())
    return emitOpError("requires identical element types");
  if (!dimensionsCompatible(lhsType.getDimSize(1), rhsType.getDimSize(0)))
    return emitOpError("has incompatible contracting dimensions");
  if (!dimensionsCompatible(lhsType.getDimSize(0), outputType.getDimSize(0)) ||
      !dimensionsCompatible(rhsType.getDimSize(1), outputType.getDimSize(1)))
    return emitOpError("has an output shape inconsistent with lhs and rhs");
  return success();
}

LogicalResult BiasAddOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !biasType || !outputType)
    return emitOpError("requires ranked tensor operands and result");
  if (inputType.getRank() != 2 || biasType.getRank() != 1 ||
      outputType.getRank() != 2)
    return emitOpError("requires rank-2 input/output and rank-1 bias");
  if (inputType != outputType)
    return emitOpError("requires output type to equal input type");
  if (inputType.getElementType() != biasType.getElementType())
    return emitOpError("requires input and bias to have identical element types");
  if (!dimensionsCompatible(inputType.getDimSize(1), biasType.getDimSize(0)))
    return emitOpError("has a bias length inconsistent with the final dimension");
  return success();
}

LogicalResult ReluOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !outputType)
    return emitOpError("requires ranked tensor input and result");
  if (inputType != outputType)
    return emitOpError("requires output type to equal input type");
  return success();
}

LogicalResult BatchMatMulOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!lhsType || !rhsType || !outputType)
    return emitOpError("requires ranked tensor operands and result");
  if (lhsType.getRank() != 3 || rhsType.getRank() != 3 ||
      outputType.getRank() != 3)
    return emitOpError("requires rank-3 lhs, rhs and output tensors");
  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != outputType.getElementType())
    return emitOpError("requires identical element types");
  if (!dimensionsCompatible(lhsType.getDimSize(2), rhsType.getDimSize(1)))
    return emitOpError("has incompatible contracting dimensions");
  if (!broadcastDimensionsCompatible(lhsType.getDimSize(0),
                                     rhsType.getDimSize(0)))
    return emitOpError("has incompatible batch dimensions");
  if (!dimensionsCompatible(lhsType.getDimSize(1),
                            outputType.getDimSize(1)) ||
      !dimensionsCompatible(rhsType.getDimSize(2),
                            outputType.getDimSize(2)))
    return emitOpError("has an output shape inconsistent with lhs and rhs");

  int64_t lhsBatch = lhsType.getDimSize(0);
  int64_t rhsBatch = rhsType.getDimSize(0);
  int64_t outputBatch = outputType.getDimSize(0);
  if (!ShapedType::isDynamic(lhsBatch) &&
      !ShapedType::isDynamic(rhsBatch)) {
    int64_t expectedBatch = lhsBatch == 1 ? rhsBatch : lhsBatch;
    if (!dimensionsCompatible(expectedBatch, outputBatch))
      return emitOpError("has an output batch inconsistent with broadcasting");
  }
  return success();
}

LogicalResult SoftmaxOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !outputType)
    return emitOpError("requires ranked tensor input and result");
  if (inputType.getRank() < 1)
    return emitOpError("requires an input with rank at least one");
  if (inputType != outputType)
    return emitOpError("requires output type to equal input type");
  if (!isa<FloatType>(inputType.getElementType()))
    return emitOpError("requires a floating-point element type");

  int64_t rank = inputType.getRank();
  int64_t axis = getAxisAttr().getInt();
  if (axis < -rank || axis >= rank)
    return emitOpError() << "axis " << axis << " is outside [-" << rank
                         << ", " << rank - 1 << "]";
  return success();
}

LogicalResult RMSNormOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !weightType || !outputType)
    return emitOpError("requires ranked tensor operands and result");
  if (inputType.getRank() < 1 || weightType.getRank() != 1)
    return emitOpError(
        "requires a non-scalar input and a rank-1 weight tensor");
  if (inputType != outputType)
    return emitOpError("requires output type to equal input type");
  if (!isa<FloatType>(inputType.getElementType()) ||
      inputType.getElementType() != weightType.getElementType())
    return emitOpError(
        "requires identical floating-point input and weight element types");
  if (!dimensionsCompatible(inputType.getDimSize(inputType.getRank() - 1),
                            weightType.getDimSize(0)))
    return emitOpError(
        "has a weight length inconsistent with the final input dimension");

  double epsilon = getEpsilon().convertToDouble();
  if (!std::isfinite(epsilon) || epsilon <= 0.0)
    return emitOpError("requires a finite, positive epsilon");
  return success();
}

LogicalResult FusedMatMulBiasReluOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!lhsType || !rhsType || !biasType || !outputType)
    return emitOpError("requires ranked tensor operands and result");
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      biasType.getRank() != 1 || outputType.getRank() != 2)
    return emitOpError(
        "requires rank-2 lhs/rhs/output tensors and a rank-1 bias");
  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != biasType.getElementType() ||
      lhsType.getElementType() != outputType.getElementType())
    return emitOpError("requires identical element types");
  if (!dimensionsCompatible(lhsType.getDimSize(1), rhsType.getDimSize(0)))
    return emitOpError("has incompatible contracting dimensions");
  if (!dimensionsCompatible(lhsType.getDimSize(0), outputType.getDimSize(0)) ||
      !dimensionsCompatible(rhsType.getDimSize(1), outputType.getDimSize(1)))
    return emitOpError("has an output shape inconsistent with lhs and rhs");
  if (!dimensionsCompatible(rhsType.getDimSize(1), biasType.getDimSize(0)))
    return emitOpError("has a bias length inconsistent with the N dimension");
  return success();
}

#define GET_OP_CLASSES
#include "MiniNPU/Dialect/MiniNPU/MiniNPUOps.cpp.inc"
