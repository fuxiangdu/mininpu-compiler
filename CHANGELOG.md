# Changelog

## 0.7.0

- Added Linalg-to-SCF loop materialization and progressive LLVM dialect lowering.
- Added LLVM IR translation, native Clang linking and host CPU execution.
- Added four-element numerical checking through a minimal C runtime ABI.

## 0.6.0

- Added an end-to-end One-Shot Bufferization stage after MiniNPU lowering.
- Added ownership-based buffer deallocation for function-local allocations.
- Added tensor-elimination, metadata-preservation and wrong-order regressions.

## 0.5.0

- Added full MiniNPU-to-Tensor/Linalg/Arith dialect conversion.
- Preserved tile-planning metadata on the lowered `linalg.matmul` operation.
- Added lowering idempotence and wrong-pipeline negative tests.

## 0.4.0

- Added UB-capacity-aware M/N/K tile planning.
- Added deterministic cost-model tie breaking and dynamic-shape fallback.

## 0.3.0

- Added safe MatMul-BiasAdd-ReLU graph fusion.

## 0.2.0

- Added the MiniNPU dialect, ODS operations and semantic verifiers.

## 0.1.0

- Added the standalone `mininpu-opt` driver and upstream-pass smoke tests.
