//===- PlanTiles.cpp - UB-aware MiniNPU tile planner -----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "MiniNPU/Transforms/Passes.h"

#include "MiniNPU/Dialect/MiniNPU/MiniNPUOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

using namespace mlir;

namespace {

constexpr int64_t kDefaultUbBytes = 256 * 1024;
constexpr int64_t kDefaultTileGranularity = 16;

struct TilePlan {
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  int64_t workingSetBytes = 0;
  int64_t tileCount = 0;
  int64_t macsPerTile = 0;
  long double arithmeticIntensity = 0.0;
};

int64_t ceilDiv(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

std::vector<int64_t> buildCandidates(int64_t dimension,
                                     int64_t granularity) {
  std::vector<int64_t> candidates;
  if (dimension <= 0 || granularity <= 0)
    return candidates;
  if (dimension < granularity) {
    candidates.push_back(dimension);
    return candidates;
  }
  for (int64_t value = granularity; value <= dimension;
       value += granularity)
    candidates.push_back(value);
  std::reverse(candidates.begin(), candidates.end());
  return candidates;
}

std::optional<int64_t> estimateWorkingSetBytes(int64_t tileM, int64_t tileN,
                                               int64_t tileK,
                                               int64_t elementBytes) {
  // Lhs and rhs are double-buffered. Accumulation is modeled in fp32. The
  // output tile and one bias vector are resident in UB as well.
  __int128 inputElements = static_cast<__int128>(tileM) * tileK +
                           static_cast<__int128>(tileK) * tileN;
  __int128 inputBytes = static_cast<__int128>(2) * inputElements *
                        elementBytes;
  __int128 accumulatorBytes =
      static_cast<__int128>(tileM) * tileN * sizeof(float);
  __int128 outputBytes =
      static_cast<__int128>(tileM) * tileN * elementBytes;
  __int128 biasBytes = static_cast<__int128>(tileN) * elementBytes;
  __int128 total = inputBytes + accumulatorBytes + outputBytes + biasBytes;
  if (total > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(total);
}

bool isBetterPlan(const TilePlan &candidate, const TilePlan &best) {
  constexpr long double kEpsilon = 1.0e-15L;
  if (candidate.arithmeticIntensity > best.arithmeticIntensity + kEpsilon)
    return true;
  if (std::abs(candidate.arithmeticIntensity - best.arithmeticIntensity) >
      kEpsilon)
    return false;
  if (candidate.tileCount != best.tileCount)
    return candidate.tileCount < best.tileCount;
  return candidate.macsPerTile > best.macsPerTile;
}

std::optional<TilePlan> chooseTilePlan(int64_t m, int64_t n, int64_t k,
                                       int64_t elementBytes, int64_t ubBytes,
                                       int64_t granularity) {
  std::optional<TilePlan> best;
  for (int64_t tileM : buildCandidates(m, granularity)) {
    for (int64_t tileN : buildCandidates(n, granularity)) {
      for (int64_t tileK : buildCandidates(k, granularity)) {
        auto workingSet = estimateWorkingSetBytes(tileM, tileN, tileK,
                                                  elementBytes);
        if (!workingSet || *workingSet > ubBytes)
          continue;

        TilePlan candidate;
        candidate.tileM = tileM;
        candidate.tileN = tileN;
        candidate.tileK = tileK;
        candidate.workingSetBytes = *workingSet;
        candidate.macsPerTile = tileM * tileN * tileK;
        candidate.tileCount =
            ceilDiv(m, tileM) * ceilDiv(n, tileN) * ceilDiv(k, tileK);
        candidate.arithmeticIntensity =
            static_cast<long double>(candidate.macsPerTile) /
            static_cast<long double>(candidate.workingSetBytes);

        if (!best || isBetterPlan(candidate, *best))
          best = candidate;
      }
    }
  }
  return best;
}

std::optional<int64_t> getElementBytes(Type type) {
  unsigned bitWidth = 0;
  if (auto integerType = dyn_cast<IntegerType>(type))
    bitWidth = integerType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(type))
    bitWidth = floatType.getWidth();
  else
    return std::nullopt;
  return (static_cast<int64_t>(bitWidth) + 7) / 8;
}

FailureOr<int64_t> readPositiveConfig(ModuleOp module, StringRef name,
                                      int64_t fallback) {
  auto attribute = module->getAttrOfType<IntegerAttr>(name);
  if (!attribute)
    return fallback;
  int64_t value = attribute.getInt();
  if (value <= 0) {
    module.emitError() << name << " must be a positive integer";
    return failure();
  }
  return value;
}

class PlanTilesPass final
    : public PassWrapper<PlanTilesPass, OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanTilesPass)

  StringRef getArgument() const final { return "mininpu-plan-tiles"; }
  StringRef getDescription() const final {
    return "Plan UB-constrained tiles for fused MiniNPU linear operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<int64_t> ubBytes = readPositiveConfig(
        module, "mininpu.ub_bytes", kDefaultUbBytes);
    FailureOr<int64_t> granularity = readPositiveConfig(
        module, "mininpu.tile_granularity", kDefaultTileGranularity);
    if (failed(ubBytes) || failed(granularity)) {
      signalPassFailure();
      return;
    }

    Builder builder(&getContext());
    WalkResult result = module.walk(
        [&](mininpu::FusedMatMulBiasReluOp operation) -> WalkResult {
          auto lhsType = dyn_cast<RankedTensorType>(operation.getLhs().getType());
          auto rhsType = dyn_cast<RankedTensorType>(operation.getRhs().getType());
          if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
              !rhsType.hasStaticShape()) {
            operation.emitRemark(
                "skipping compile-time tile planning for dynamic shapes");
            return WalkResult::advance();
          }

          int64_t m = lhsType.getDimSize(0);
          int64_t k = lhsType.getDimSize(1);
          int64_t n = rhsType.getDimSize(1);
          if (m <= 0 || n <= 0 || k <= 0) {
            operation.emitError("requires positive static M, N and K");
            return WalkResult::interrupt();
          }

          std::optional<int64_t> elementBytes =
              getElementBytes(lhsType.getElementType());
          if (!elementBytes) {
            operation.emitError(
                "tile planner supports integer and floating-point elements");
            return WalkResult::interrupt();
          }

          std::optional<TilePlan> plan = chooseTilePlan(
              m, n, k, *elementBytes, *ubBytes, *granularity);
          if (!plan) {
            operation.emitError()
                << "no legal tile fits in " << *ubBytes
                << " UB bytes with granularity " << *granularity;
            return WalkResult::interrupt();
          }

          operation->setAttr("mininpu.tile_m",
                             builder.getI64IntegerAttr(plan->tileM));
          operation->setAttr("mininpu.tile_n",
                             builder.getI64IntegerAttr(plan->tileN));
          operation->setAttr("mininpu.tile_k",
                             builder.getI64IntegerAttr(plan->tileK));
          operation->setAttr(
              "mininpu.estimated_working_set_bytes",
              builder.getI64IntegerAttr(plan->workingSetBytes));
          operation->setAttr("mininpu.estimated_tile_count",
                             builder.getI64IntegerAttr(plan->tileCount));
          operation->setAttr(
              "mininpu.arithmetic_intensity",
              builder.getF64FloatAttr(
                  static_cast<double>(plan->arithmeticIntensity)));
          operation->setAttr("mininpu.ub_bytes",
                             builder.getI64IntegerAttr(*ubBytes));
          operation->setAttr("mininpu.double_buffered",
                             builder.getBoolAttr(true));
          return WalkResult::advance();
        });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mininpu::createPlanTilesPass() {
  return std::make_unique<PlanTilesPass>();
}

void mininpu::registerPlanTilesPass() {
  PassRegistration<PlanTilesPass>();
}
