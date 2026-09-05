# Changelog

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
