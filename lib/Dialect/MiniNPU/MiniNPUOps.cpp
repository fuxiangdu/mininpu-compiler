//===- MiniNPUOps.cpp - MiniNPU operation definitions ----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "MiniNPU/Dialect/MiniNPU/MiniNPUOps.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mininpu;

namespace {

bool dimensionsCompatible(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
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
