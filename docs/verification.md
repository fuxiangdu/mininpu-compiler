# Verification report

## Environment

The uploaded verification run was completed on:

- Ubuntu 24.04.4 LTS, x86-64;
- Clang/LLVM/MLIR 18.1.3;
- CMake 3.28.3 and Ninja 1.11.1;
- four CPU cores and approximately 4 GiB RAM.

The source manifest matched every tracked source, script, test and document.
The compiler built successfully, including `FuseMatMulBiasRelu.cpp`,
`PlanTiles.cpp` and `LowerToLinalg.cpp`.

## Regression matrix

| Stage | Positive checks | Negative or safety checks | Result |
| --- | --- | --- | --- |
| v0 | parse/print; canonicalization; constant folding | — | PASS |
| v1 | custom operations registered | incompatible MatMul contraction rejected | PASS |
| v2 | 3-to-1 fusion; idempotence | shared intermediate left unchanged | PASS |
| v3 | 64/256 KiB planning | 128 B rejected; dynamic shape deferred | PASS |
| v4 | structured lowering; metadata; idempotence | unfused illegal operations rejected | PASS |

## Tile-plan evidence

Input problem: `M=128`, `K=256`, `N=512`, `f32`, granularity 16.

| Metric | 64 KiB | 256 KiB |
| --- | ---: | ---: |
| `tile_m` | 48 | 112 |
| `tile_n` | 48 | 96 |
| `tile_k` | 48 | 96 |
| estimated working set | 55,488 B | 246,144 B |
| estimated tile count | 198 | 36 |
| arithmetic intensity score | 1.9930796 | 4.1934477 |

The working set is computed as:

```text
2*b*(tm*tk + tk*tn) + 4*(tm*tn) + b*(tm*tn) + b*tn
```

This models double-buffered input tiles, an fp32 accumulator, an output tile
and one bias vector. The score is MACs per estimated resident byte, with fewer
total tiles and then more MACs per tile used as deterministic tie breakers.

## Lowering evidence

The custom fused operation is eliminated and replaced by:

```text
tensor.empty -> linalg.fill -> linalg.matmul -> linalg.generic
```

The generic operation uses affine maps `(d0,d1)->(d1)` for bias broadcasting
and `(d0,d1)->(d0,d1)` for the output. Its scalar body uses `arith.addf` and
`arith.maximumf`. The 256 KiB tile metadata remains attached to
`linalg.matmul`, and no MiniNPU operation remains in the output.

Evidence files:

- `evidence/planned_64k.mlir`;
- `evidence/planned_256k.mlir`;
- `evidence/lowered.mlir`.

## Interpretation limits

These results prove compiler transformations, verifier behavior and cost-model
consistency. They are not hardware latency measurements and do not prove that
the selected tiles are optimal on a commercial NPU.
