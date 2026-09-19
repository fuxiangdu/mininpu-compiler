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

The v7 scheduling extension was additionally built and fully regressed against
an LLVM/MLIR 18.1.8 source build with GCC 13.3.0. This run covered
`ApplyTiles.cpp`, the `llc` plus GCC native-link fallback and all v0-v7 scripts.

## Regression matrix

| Stage | Positive checks | Negative or safety checks | Result |
| --- | --- | --- | --- |
| v0 | parse/print; canonicalization; constant folding | — | PASS |
| v1 | custom operations registered | incompatible MatMul contraction rejected | PASS |
| v2 | 3-to-1 fusion; idempotence | shared intermediate left unchanged | PASS |
| v3 | 64/256 KiB planning | 128 B rejected; dynamic shape deferred | PASS |
| v4 | fused and standalone structured lowering; metadata; idempotence | shared-use safety; dynamic shapes rejected | PASS |
| v5 | tensor elimination; MemRef allocation/load; tile metadata | local deallocation; wrong order rejected | PASS |
| v6 | SCF loops; LLVM dialect/IR; native linking | four reference outputs and process status | PASS |
| v7 | planned M/N/K sizes become SCF loops and slices | partial boundaries; idempotence; incomplete plan rejected | PASS |
| v8 | BatchMatMul, stable Softmax and FP32 RMSNorm lowering | broadcasts, large logits, FP16 casts and dynamic rejection | PASS |
| v9 | attention graph planning and native execution | shared-value boundary, idempotence and four outputs | PASS |

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

MatMul, BiasAdd and ReLU also have independent conversion patterns. This makes
lowering compositional: an unfused graph, including one whose MatMul result is
shared by multiple users, can be fully legalized without requiring an unsafe
fusion first. Dynamic shapes remain an explicit unsupported case at this
stage and fail full dialect conversion.

Evidence files:

- `evidence/planned_64k.mlir`;
- `evidence/planned_256k.mlir`;
- `evidence/lowered.mlir`.
- `evidence/bufferized.mlir`.

## Bufferization evidence

The v5 regression completed successfully on LLVM/MLIR 18.1.3. The generated IR
contains MemRef-form Linalg, one `memref.alloc`, one matching `memref.dealloc`
and a `memref.load`; it contains no tensor types or tensor operations. The v3
tile-plan attributes remain attached to `linalg.matmul`. Running One-Shot
Bufferize before MiniNPU lowering is rejected as expected because the custom
tensor operations do not implement `BufferizableOpInterface`.

## v6 execution evidence

The verified v6 pipeline evaluates a 2x2 MatMul-BiasAdd-ReLU program whose expected
result is `[[0, 6], [5, 10]]`. The successful regression contains explicit `scf.for` loops,
an LLVM-dialect-only module, translated LLVM IR containing a native `main`,
successful Clang linking, four passing runtime checks and exit status zero.
The complete v0-v6 regression passed on Ubuntu 24.04 with LLVM/MLIR 18.1.3.
Generated v6 IR and runtime outputs are stored under `docs/evidence/v6/`.

## v7 scheduled-tiling evidence

The 256 KiB `(112,96,96)` plan is materialized as three nested `scf.for`
loops. Dynamic `tensor.extract_slice` and `tensor.insert_slice` operations use
clamped sizes at the M/N/K boundaries, including the non-divisible 128/512/256
tails. The tiled `linalg.matmul` carries both the original cost-model evidence
and `mininpu.tiles_applied = true`, which prevents accidental repeated tiling.
The regression verifies byte-for-byte idempotence and rejects a partial plan.

## v8 multi-operation evidence

Equal-batch BatchMatMul lowers to `linalg.batch_matmul`; unit-batch cases use
explicit affine broadcast maps. Softmax lowers to max, subtract, exponential,
sum and divide stages and remains finite for logits `[1000,1001,1002]`.
RMSNorm uses explicit FP16-to-FP32 casts for square-sum accumulation and casts
the scaled result back to the input type. All three paths reach native code and
match their reference outputs.

## v9 attention-plan evidence

The planner recognizes `BatchMatMul -> Softmax -> BatchMatMul -> RMSNorm`, tags
the four graph roles and checks every producer-consumer edge with SSA use
counts. A single-use graph becomes a `streaming-fusion-candidate`; an extra
consumer of the score tensor forces `materialize-shared-boundaries`. For the
checked `1x2x2xf32` graph, the pass estimates 96 versus 64 bytes of avoidable
write/read traffic. Replanning is byte-for-byte idempotent, and the complete
unfused semantic graph lowers to LLVM and matches four native outputs within
`5.96e-08` absolute error.

## Interpretation limits

These results prove compiler transformations, verifier behavior and cost-model
consistency. Fusion attributes are planning decisions, not evidence that a
monolithic fused attention kernel has been emitted. These are not hardware
latency measurements and do not prove that selected tiles are optimal on a
commercial NPU.
