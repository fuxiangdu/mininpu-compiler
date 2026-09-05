//===- MiniNPUDialect.cpp - MiniNPU dialect definition ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "MiniNPU/Dialect/MiniNPU/MiniNPUDialect.h"
#include "MiniNPU/Dialect/MiniNPU/MiniNPUOps.h"

using namespace mininpu;

#include "MiniNPU/Dialect/MiniNPU/MiniNPUDialect.cpp.inc"

void MiniNPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MiniNPU/Dialect/MiniNPU/MiniNPUOps.cpp.inc"
      >();
}
