# 简历项目描述

## 推荐标题

**MiniNPU Compiler：基于 MLIR 的自定义 Dialect、图融合与 UB 感知分块**

技术栈：`C++17 / LLVM-MLIR 18 / ODS-TableGen / PatternRewriter / Dialect Conversion / CMake / Ninja / Linux`

## 一页简历版（推荐直接使用）

1. 基于 LLVM/MLIR 18 构建独立 `mininpu-opt` 编译器驱动，使用 ODS/TableGen
   定义 MatMul、BiasAdd、ReLU 及融合算子，并实现秩、数据类型和静态维度
   一致性 Verifier。
2. 基于 `OpRewritePattern` 实现 MatMul-BiasAdd-ReLU 三算子融合，引入单用户
   约束避免共享中间值被错误删除，并通过正例、共享值负例及幂等性测试。
3. 设计 UB 容量约束的 M/N/K 分块搜索与算术强度代价模型；在
   `128×256×512 f32` 用例中，64 KiB/256 KiB 分别选择
   `(48,48,48)`/`(112,96,96)`，工作集均满足容量约束；随后通过 Dialect
   Conversion Lowering 到 Tensor/Linalg/Arith，并保留分块元数据。

## 更短的三行版

- 基于 MLIR 18 开发 MiniNPU 自定义 Dialect、ODS 算子和静态形状 Verifier；
- 实现带共享值安全检查的 MatMul-BiasAdd-ReLU 融合及 UB 感知分块 Pass；
- 将融合算子 Lowering 到 Tensor/Linalg/Arith，完成 v0-v4 正负例与幂等性回归。

## 不应写入简历的表述

- 不写“实现商用 NPU 编译器”——这是教育型编译器原型；
- 不写“生成 NPU 机器码”——当前终点是 Tensor/Linalg/Arith；
- 不写“性能提升若干倍”——项目 B 没有真实硬件延迟基准；
- 不写“最优分块”——代价模型只是在给定候选和假设下选择最优项；
- 不写“支持动态形状 Lowering”——v4 Lowering 当前要求静态浮点 Tensor。

## 与项目 A 的简历排序

建议项目顺序：

1. Ascend C Add-RMSNorm 融合算子优化（真实 Ascend 310B1 与性能数据）；
2. MiniNPU Compiler（编译器前端、Pass、代价模型和 Lowering 完整链路）；
3. 硕士课题（机器人/端侧智能系统）。

项目 A 证明你能在真实 NPU 上开发和优化算子；项目 B 证明你理解编译器 IR、
Rewrite、规划和 Lowering。两者组合比两个同类型 Demo 更有说服力。
