//===- MiniNPUDialect.td - MiniNPU dialect definition ------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef MININPU_DIALECT_MININPU_MININPUDIALECT
#define MININPU_DIALECT_MININPU_MININPUDIALECT

include "mlir/IR/DialectBase.td"

def MiniNPU_Dialect : Dialect {
  let name = "mininpu";
  let cppNamespace = "::mininpu";
  let summary = "An educational tensor dialect for NPU compiler practice";
  let description = [{
    MiniNPU models a small tensor graph before target-specific NPU tiling and
    lowering.  Version 1 defines graph-level operations only; later versions
    add fusion, UB-aware planning and lowering passes.
  }];
}

#endif // MININPU_DIALECT_MININPU_MININPUDIALECT
