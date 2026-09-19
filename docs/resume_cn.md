# 简历项目描述

## 推荐标题

**MiniNPU Compiler：基于 MLIR 的多算子 Lowering、跨算子规划与 UB 感知调度**

技术栈：`C++17 / LLVM-MLIR 18 / ODS-TableGen / PatternRewriter / Dialect Conversion / One-Shot Bufferization / SCF / LLVM IR / CMake / Ninja / Linux`

## 一页简历版（推荐直接使用）

1. 基于 LLVM/MLIR 18 构建独立 `mininpu-opt` 编译器驱动，使用 ODS/TableGen
   定义 MatMul、BatchMatMul、Softmax、RMSNorm 等算子，并实现秩、广播、
   归约轴、权重形状及数据类型 Verifier。
2. 基于 `OpRewritePattern` 实现 MatMul-BiasAdd-ReLU 三算子融合，引入单用户
   约束避免共享中间值被错误删除，并通过正例、共享值负例及幂等性测试。
3. 实现广播 BatchMatMul、max-subtract 数值稳定 Softmax 及 FP32 累积
   RMSNorm 的结构化 Linalg Lowering，覆盖负轴归约、F16 类型转换、动态形状
   诊断，并逐级 Lowering 到 SCF/LLVM IR，通过本机程序完成数值验证。
4. 设计 UB 容量约束的 M/N/K 分块搜索与算术强度代价模型；在
   `128×256×512 f32` 用例中，64 KiB/256 KiB 分别选择
   `(48,48,48)`/`(112,96,96)`；将计划实际物化为带尾块边界处理的三层
   SCF 循环及 `tensor.extract_slice/insert_slice`，并验证 Pass 幂等性与
   不完整计划诊断；通过 Dialect Conversion Lowering 到 Tensor/Linalg/Arith；集成
   One-Shot Bufferization，消除 Tensor 并完成局部缓冲区所有权回收；继续
   Lowering 到显式 SCF 循环和 LLVM 方言，生成宿主 LLVM IR，并通过 C 运行时
   对 2x2 用例的输出元素完成端到端数值校验；进一步识别
   QK-Softmax-PV-RMSNorm 子图，基于 use-def 单用户约束规划流式融合候选与
   安全物化边界，并估算可避免的中间张量读写流量。

## 更短的三行版

- 基于 MLIR 18 开发多算子 MiniNPU Dialect、ODS 定义及形状/广播/归约 Verifier；
- 实现 BatchMatMul、稳定 Softmax、FP32 累积 RMSNorm 的 Linalg/SCF/LLVM
  Lowering，以及 UB 感知分块与边界安全的 Tensor 切片；
- 识别 QK-Softmax-PV-RMSNorm 子图，基于 use-def 规划融合/物化边界并估算
  中间流量，通过整图本机执行验证数值正确性。

## 不应写入简历的表述

- 不写“实现商用 NPU 编译器”——这是教育型编译器原型；
- 不写“生成 NPU 机器码”——v9 生成的是宿主 x86-64 LLVM IR；
- 不写“已实现融合 Attention 内核”——当前实现的是跨算子融合候选规划；
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
