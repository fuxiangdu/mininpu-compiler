# v4-v5 lowering and bufferization pipeline

## Input and output

The v4 pass converts the planned custom operation into upstream MLIR
dialects:

```text
mininpu.fused_matmul_bias_relu
  -> tensor.empty
  -> linalg.fill(0)
  -> linalg.matmul
  -> linalg.generic(bias broadcast + ReLU)
```

`linalg.matmul` receives the v3 tile-plan attributes. Keeping target metadata
next to the matrix operation makes the plan visible to later scheduling or
code-generation work even though the custom operation no longer exists.

## Why dialect conversion is used

The pass declares the entire MiniNPU dialect illegal and the Tensor, Linalg,
Arith and Func dialects legal. Conversion succeeds only when every custom
operation has been eliminated. Running the lowering before fusion therefore
fails instead of silently leaving a mixed-dialect module.

## Destination-passing style

The output tensor is created by `tensor.empty`, initialized to zero by
`linalg.fill`, and passed as the destination of `linalg.matmul`. The result of
the matrix multiplication is then the destination of `linalg.generic`. Inside
the generic region, the bias is broadcast over the final dimension, added to
the matrix element, and clamped with zero using `arith.maximumf`.

## Current boundary

v4 supports statically shaped floating-point tensors. v5 then runs
`empty-tensor-to-alloc-tensor`, One-Shot Bufferize with function-boundary
bufferization, and the ownership-based deallocation pipeline:

```text
Tensor/Linalg/Arith
  -> bufferization.alloc_tensor
  -> one-shot-bufferize
  -> MemRef/Linalg/Arith
  -> ownership-based local deallocation
```

The v5 regression uses a scalar external sink so that the result is observable
but its allocation remains local. It verifies `memref.alloc`, `memref.load` and
`memref.dealloc`, complete elimination of tensor types/operations, preservation
of tile attributes, and rejection of bufferization before custom-op lowering.

v5 still does not produce executable LLVM IR or proprietary NPU machine code.
Loop/vector lowering, runtime ABI integration and target code generation are
separate later stages.
