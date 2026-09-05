//===- MiniNPUOps.h - MiniNPU operation declarations -----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef MININPU_DIALECT_MININPU_MININPUOPS_H
#define MININPU_DIALECT_MININPU_MININPUOPS_H

#include "MiniNPU/Dialect/MiniNPU/MiniNPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "MiniNPU/Dialect/MiniNPU/MiniNPUOps.h.inc"

#endif // MININPU_DIALECT_MININPU_MININPUOPS_H
