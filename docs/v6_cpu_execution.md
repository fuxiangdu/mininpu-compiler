# v6 CPU execution pipeline

v6 turns the small MiniNPU MatMul-BiasAdd-ReLU program into a native x86-64
executable. It is an execution proof for the compiler pipeline, not an NPU
instruction generator.

```text
MiniNPU graph
  -> fusion and UB-aware planning
  -> Tensor/Linalg/Arith
  -> One-Shot Bufferization
  -> MemRef/Linalg/Arith
  -> explicit SCF loops
  -> LLVM dialect
  -> LLVM IR
  -> Clang native executable
```

The executable evaluates a 2x2 f32 case. For

```text
lhs  = [[1, 2], [3, 4]]
rhs  = [[1, -1], [2, 3]]
bias = [-6, 1]
```

the reference result after bias addition and ReLU is
`[[0, 6], [5, 10]]`. A small C runtime checks every output element and returns
the mismatch count as the process exit code.

Tile-planning metadata is inspected before Linalg is materialized as loops.
The metadata is intentionally no longer present after loop lowering because
the annotated `linalg.matmul` operation has been consumed.

The final executable targets the host CPU. Proprietary NPU instruction
selection, device runtime integration and hardware performance measurements
remain outside the scope of v6.
