//===- PlanAttention.cpp - Plan MiniNPU attention subgraphs ----*- C++ -*-===//
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

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

using namespace mlir;

namespace {

constexpr StringLiteral kGraphId = "mininpu.attention_graph_id";
constexpr StringLiteral kRole = "mininpu.attention_role";

std::optional<int64_t> getTensorBytes(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  auto elementType = type ? dyn_cast<FloatType>(type.getElementType())
                          : FloatType();
  if (!type || !type.hasStaticShape() || !elementType)
    return std::nullopt;

  __int128 elements = 1;
  for (int64_t dimension : type.getShape()) {
    if (dimension < 0)
      return std::nullopt;
    elements *= dimension;
  }
  __int128 bytes = elements * ((elementType.getWidth() + 7) / 8);
  if (bytes > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(bytes);
}

void clearAttentionAttrs(Operation *operation) {
  operation->removeAttr(kGraphId);
  operation->removeAttr(kRole);
  operation->removeAttr("mininpu.attention_plan");
  operation->removeAttr("mininpu.fuse_qk_softmax");
  operation->removeAttr("mininpu.fuse_softmax_pv");
  operation->removeAttr("mininpu.fuse_pv_rms_norm");
  operation->removeAttr("mininpu.estimated_traffic_saved_bytes");
}

class PlanAttentionPass final
    : public PassWrapper<PlanAttentionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanAttentionPass)

  StringRef getArgument() const final { return "mininpu-plan-attention"; }
  StringRef getDescription() const final {
    return "Recognize attention-RMSNorm graphs and plan safe fusion boundaries";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->removeAttr("mininpu.attention_graph_count");
    module.walk([](Operation *operation) { clearAttentionAttrs(operation); });

    Builder builder(&getContext());
    int64_t graphCount = 0;
    module.walk([&](mininpu::RMSNormOp rmsNorm) {
      auto pv = rmsNorm.getInput().getDefiningOp<mininpu::BatchMatMulOp>();
      if (!pv)
        return;
      auto softmax = pv.getLhs().getDefiningOp<mininpu::SoftmaxOp>();
      if (!softmax)
        return;
      auto qk = softmax.getInput().getDefiningOp<mininpu::BatchMatMulOp>();
      if (!qk)
        return;

      auto scoreType = dyn_cast<RankedTensorType>(softmax.getInput().getType());
      int64_t axis = softmax.getAxisAttr().getInt();
      if (!scoreType)
        return;
      if (axis < 0)
        axis += scoreType.getRank();
      if (axis != scoreType.getRank() - 1)
        return;

      int64_t graphId = graphCount++;
      auto graphIdAttr = builder.getI64IntegerAttr(graphId);
      qk->setAttr(kGraphId, graphIdAttr);
      softmax->setAttr(kGraphId, graphIdAttr);
      pv->setAttr(kGraphId, graphIdAttr);
      rmsNorm->setAttr(kGraphId, graphIdAttr);
      qk->setAttr(kRole, builder.getStringAttr("qk"));
      softmax->setAttr(kRole, builder.getStringAttr("softmax"));
      pv->setAttr(kRole, builder.getStringAttr("pv"));
      rmsNorm->setAttr(kRole, builder.getStringAttr("output_norm"));

      bool fuseQkSoftmax = qk.getOutput().hasOneUse();
      bool fuseSoftmaxPv = softmax.getOutput().hasOneUse();
      bool fusePvRmsNorm = pv.getOutput().hasOneUse();
      rmsNorm->setAttr("mininpu.fuse_qk_softmax",
                       builder.getBoolAttr(fuseQkSoftmax));
      rmsNorm->setAttr("mininpu.fuse_softmax_pv",
                       builder.getBoolAttr(fuseSoftmaxPv));
      rmsNorm->setAttr("mininpu.fuse_pv_rms_norm",
                       builder.getBoolAttr(fusePvRmsNorm));
      bool fullyStreamable =
          fuseQkSoftmax && fuseSoftmaxPv && fusePvRmsNorm;
      rmsNorm->setAttr(
          "mininpu.attention_plan",
          builder.getStringAttr(fullyStreamable
                                    ? "streaming-fusion-candidate"
                                    : "materialize-shared-boundaries"));

      __int128 savedBytes = 0;
      bool trafficKnown = true;
      auto addSavedTraffic = [&](Value value, bool fused) {
        std::optional<int64_t> bytes = getTensorBytes(value);
        if (!fused)
          return;
        if (!bytes) {
          trafficKnown = false;
          return;
        }
        savedBytes += static_cast<__int128>(*bytes) * 2;
      };
      addSavedTraffic(qk.getOutput(), fuseQkSoftmax);
      addSavedTraffic(softmax.getOutput(), fuseSoftmaxPv);
      addSavedTraffic(pv.getOutput(), fusePvRmsNorm);
      if (trafficKnown &&
          savedBytes <= std::numeric_limits<int64_t>::max())
        rmsNorm->setAttr(
            "mininpu.estimated_traffic_saved_bytes",
            builder.getI64IntegerAttr(static_cast<int64_t>(savedBytes)));
    });

    module->setAttr("mininpu.attention_graph_count",
                    builder.getI64IntegerAttr(graphCount));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mininpu::createPlanAttentionPass() {
  return std::make_unique<PlanAttentionPass>();
}

void mininpu::registerPlanAttentionPass() {
  PassRegistration<PlanAttentionPass>();
}
