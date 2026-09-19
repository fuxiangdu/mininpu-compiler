//===- MiniNPUOps.td - MiniNPU operation definitions -------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef MININPU_DIALECT_MININPU_MININPUOPS
#define MININPU_DIALECT_MININPU_MININPUOPS

include "MiniNPUDialect.td"
include "mlir/IR/OpBase.td"

class MiniNPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<MiniNPU_Dialect, mnemonic, traits>;

def MiniNPU_MatMulOp : MiniNPU_Op<"matmul"> {
  let summary = "Rank-2 tensor matrix multiplication";
  let description = [{
    Multiplies MxK and KxN tensors and produces an MxN tensor.  The verifier
    checks ranks, element types and all statically known dimensions.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def MiniNPU_BiasAddOp : MiniNPU_Op<"bias_add"> {
  let summary = "Add a rank-1 bias to the final dimension of a rank-2 tensor";
  let arguments = (ins AnyTensor:$input, AnyTensor:$bias);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def MiniNPU_ReluOp : MiniNPU_Op<"relu"> {
  let summary = "Elementwise rectified linear unit";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def MiniNPU_BatchMatMulOp : MiniNPU_Op<"batch_matmul"> {
  let summary = "Batched rank-3 tensor matrix multiplication";
  let description = [{
    Multiplies BxMxK and BxKxN tensors and produces BxMxN. A unit batch
    dimension may be broadcast. The verifier checks ranks, element types,
    contraction dimensions, batch compatibility and the output shape.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def MiniNPU_SoftmaxOp : MiniNPU_Op<"softmax"> {
  let summary = "Numerically stable softmax along one tensor dimension";
  let description = [{
    Computes softmax along the selected axis. Negative axes follow the usual
    Python convention and are normalized against the input rank.
  }];
  let arguments = (ins AnyTensor:$input, I64Attr:$axis);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def MiniNPU_RMSNormOp : MiniNPU_Op<"rms_norm"> {
  let summary = "RMS normalization over the final tensor dimension";
  let description = [{
    Normalizes the final input dimension using a rank-1 weight and a positive
    epsilon. Lowering accumulates the sum of squares in fp32.
  }];
  let arguments = (ins AnyTensor:$input, AnyTensor:$weight,
                       F64Attr:$epsilon);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def MiniNPU_FusedMatMulBiasReluOp
    : MiniNPU_Op<"fused_matmul_bias_relu"> {
  let summary = "Fused matrix multiplication, bias addition and ReLU";
  let description = [{
    Represents one target-level fusion candidate with MxK lhs, KxN rhs and
    N-element bias operands. The result is an MxN tensor.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, AnyTensor:$bias);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

#endif // MININPU_DIALECT_MININPU_MININPUOPS
