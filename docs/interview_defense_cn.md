# MiniNPU Compiler 项目面试讲解与追问题

## 30 秒介绍

我基于 LLVM/MLIR 18 做了一个小型 NPU 编译器原型。前端用 ODS/TableGen
定义 MiniNPU Dialect 和 MatMul、BiasAdd、ReLU 等算子，并用 Verifier 检查
形状与类型；中端用 PatternRewriter 完成带单用户保护的三算子融合；目标规划
阶段根据 UB 容量枚举 M/N/K Tile，用工作集约束和算术强度选型；最后通过
Dialect Conversion Lowering 到 Tensor/Linalg/Arith，并在 v5 中接入 One-Shot
Bufferization 转换为 MemRef 语义。项目在 Ubuntu 24.04、LLVM/MLIR 18.1.3
环境完成 v0-v5 正例、负例、边界和内存所有权回归。

## 三分钟讲解顺序

1. **为什么做**：补齐“自定义 IR—优化 Pass—目标规划—Lowering”的编译器
   完整链路，而不仅是调用现成框架。
2. **前端 IR**：ODS 生成操作类与注册代码，C++ Verifier 检查秩、元素类型、
   `MxK · KxN -> MxN` 和 Bias 的 N 维匹配。
3. **融合**：从 ReLU 反向匹配 BiasAdd 和 MatMul，只在两个中间结果均为单用户
   时替换，避免破坏其他消费者；测试重复运行 Pass 不再改变 IR。
4. **分块**：先用 UB 容量做硬过滤，再按 MAC/驻留字节评分；容量变化会产生
   不同 Tile，128 B 无合法方案时明确报错，动态形状则延迟到后续阶段。
5. **Lowering**：声明 MiniNPU Dialect 非法，使用 ConversionPattern 生成
   `tensor.empty/linalg.fill/linalg.matmul/linalg.generic`；Bias 通过 AffineMap
   广播，ReLU 用 `maximumf`，最终不允许残留 MiniNPU Op。
6. **Bufferization**：先把 `tensor.empty` 显式化为 `alloc_tensor`，再分析
   inplace/out-of-place，转换函数边界并用所有权管线回收局部分配。
7. **边界**：当前未完成 LLVM IR、运行时和真实 NPU 指令生成，
   分块结果是代价模型输出而非硬件性能结论。

## 高频追问与答案

### 1. 为什么使用 MLIR，而不是直接基于 LLVM IR？

LLVM IR 更接近通用低层控制流和机器代码，难以保留 Tensor 形状、算子语义、
布局等高层信息。MLIR 支持多级 Dialect，使图级融合、目标规划、结构化算子和
低层代码生成可以分阶段表达。

### 2. Dialect、Operation、Type 和 Attribute 分别是什么？

Dialect 是一组 IR 语义的命名空间；Operation 表达计算或结构；Type 描述 SSA
值的类型；Attribute 是编译期常量或配置。本项目把算子定义为 Operation，输入
输出用 RankedTensorType，UB 容量和 Tile 参数使用 Attribute。

### 3. ODS/TableGen 帮你生成了什么？

它根据 `.td` 声明生成操作类、访问器、注册和基础验证框架，减少模板代码。
无法仅靠声明表达的矩阵维度约束由手写 C++ Verifier 完成。

### 4. 为什么 Verifier 必须在优化前工作？

后续 Pass 默认输入 IR 满足语义约束。如果 K 维不匹配仍进入融合和分块，可能
产生无意义甚至错误的目标 IR。Verifier 把错误尽早定位在产生非法算子的地方。

### 5. 融合为什么从 ReLU 向前匹配？

ReLU 是目标模式的根和最终结果，向前沿 defining-op 链可以直接确认
`Relu(BiasAdd(MatMul))`，并用一个融合算子替换根结果。

### 6. 为什么要检查 `hasOneUse()`？

若 MatMul 或 BiasAdd 的结果还有其他消费者，删除原操作会破坏这些使用者。
单用户条件保证被融合链是封闭的。共享值测试专门验证这一安全条件。

### 7. 什么是 Pass 幂等性？

同一个 Pass 对已经处理过的 IR 再运行一次，结果不应继续变化。幂等性有利于
组合 Pipeline、调试和避免重复重写。本项目对融合和 Lowering 都做了二次运行
对比。

### 8. 分块为什么受 UB 容量约束？

Tile 的输入、累加器、输出和 Bias 必须在计算阶段驻留片上存储。超出容量的
方案即使理论复用更好也无法执行，因此容量是硬约束，代价评分只比较合法方案。

### 9. 工作集公式包含哪些部分？

`2*b*(tm*tk + tk*tn)` 是双缓冲的左右输入；`4*tm*tn` 是 fp32 累加器；
`b*tm*tn` 是输出 Tile；`b*tn` 是 Bias。公式是明确记录假设的教育模型。

### 10. 为什么输入双缓冲？

双缓冲允许当前 Tile 计算时预取下一 Tile，用来重叠搬运与计算。但它会增加
片上存储占用，所以公式对左右输入乘以 2。

### 11. 为什么用算术强度作为分数？

MAC/驻留字节近似反映单位片上数据能完成多少计算，偏向数据复用更高的方案。
它不包含真实带宽、指令吞吐、Bank Conflict 和启动开销，因此只是启发式指标。

### 12. 64 KiB 和 256 KiB 为什么产生不同 Tile？

容量增大后更大的 Tile 变为合法，可以提高复用并减少 Tile 数。本例分别得到
`(48,48,48)` 和 `(112,96,96)`，工作集为 55,488 B 和 246,144 B。

### 13. 为什么不能说这些 Tile 是硬件最优？

搜索只在指定粒度候选与当前工作集/评分模型中选优，未纳入 DMA 对齐、核数、
流水线、Bank Conflict、频率和实测延迟。真正最优需要目标细节与 Benchmark。

### 14. 动态形状怎么处理？

v3 无法在编译期枚举确定的 M/N/K，因此发出 Remark 并延迟规划；v4 当前只
Lower 静态浮点 Tensor。后续可增加 Shape Constraint、运行时特化或保守回退。

### 15. Dialect Conversion 与普通 Rewrite 有什么区别？

普通 Rewrite 关注局部模式替换；Dialect Conversion 还能声明合法/非法 Dialect、
使用 TypeConverter 并检查最终合法性。本项目声明 MiniNPU 非法，确保 Lowering
后不能静默残留自定义算子。

### 16. 为什么先 `tensor.empty`，再 `linalg.fill`？

Tensor 语义的 Linalg 采用 destination-passing style，需要输出初值。MatMul 的
累加初值为 0，因此先创建目标 Tensor，再用 0 填充后作为 `outs` 传入。

### 17. Bias 广播怎么表示？

`linalg.generic` 的迭代域是 `(d0,d1)`；Bias 使用 AffineMap
`(d0,d1)->(d1)`，同一列共享一个 Bias 元素；输出使用恒等映射
`(d0,d1)->(d0,d1)`。

### 18. 为什么 ReLU 使用 `arith.maximumf`？

ReLU 的标量语义是 `max(x,0)`。Generic Region 中先用 `arith.addf` 加 Bias，
再与浮点 0 做 `maximumf`，最后通过 `linalg.yield` 返回当前元素。

### 19. 为什么保留 Tile Attribute？

自定义融合算子被消除后，目标规划信息仍需供后续调度或代码生成使用，因此把
`mininpu.tile_*` 等元数据迁移到对应的 `linalg.matmul`。

### 20. 为什么错误 Pass 顺序应失败？

Lowering Pattern 只处理融合算子，而 ConversionTarget 把整个 MiniNPU Dialect
标记为非法。若未先融合，原始 MatMul/BiasAdd/ReLU 无法合法化，Pass 必须报错，
从而暴露 Pipeline 配置错误。

### 21. 项目如何测试？

覆盖解析打印、常量折叠、非法形状、融合正例、共享使用负例、融合幂等性、
64/256 KiB 容量差异、无合法 Tile、动态形状回退、Lowering 结构、属性保留、
Lowering 幂等性和错误顺序拒绝。

### 22. 下一步最有价值的扩展是什么？

先做 One-Shot Bufferization，将 Tensor Linalg 转为 MemRef 语义；再做 Linalg
Tiling/Loop Lowering、Vectorization 和目标内存空间映射；随后设计运行时 ABI 与
模拟器或真实后端。每一步都需要新增数值正确性和性能验证。

### 23. 如果面试官让你现场改项目，你会改什么？

可以新增 LeakyReLU/Activation 属性、动态形状 Lowering、更多融合模式，或把
Tile 搜索抽成独立可测试策略。修改时先补负例和边界测试，再实现 Pattern/Pass。

### 24. 这个项目最难的地方是什么？

不是写一个算子，而是保持各阶段语义一致：Verifier 保证输入，融合必须保护
共享 SSA 值，分块必须先满足容量，Lowering 必须完全合法化并保留目标元数据。

### 25. 项目与真实 NPU 工作有什么联系？

它覆盖了真实岗位常见的 IR 定义、Rewrite、算子融合、片上存储建模、Pass
Pipeline、Lowering 和测试方法；但真实产品还包含复杂硬件约束、运行时、指令
选择、多核调度、性能工具和大量算子生态。

### 26. Tensor 和 MemRef 的核心区别是什么？

Tensor 表达值语义，通常不显式描述地址、别名和生命周期；MemRef 表达带形状、
步长、布局与存储空间信息的内存引用。Bufferization 的任务是把 Tensor SSA 值
映射到具体缓冲区，并决定哪些结果可以复用已有目标缓冲区。

### 27. One-Shot Bufferize 为什么适合 Linalg？

Linalg 使用 destination-passing style，输出目标已经作为 `outs` 操作数传入。
One-Shot Bufferize 可以沿 SSA use-def 链分析读写冲突，在安全时让结果与目标
缓冲区原地复用，不能原地复用时才插入分配或复制。

### 28. 为什么 One-Shot Bufferize 后还要做 Deallocation？

One-Shot Bufferize 负责分配和别名决策，不负责完整生命周期回收。v5 使用
ownership-based deallocation pipeline 跟踪局部缓冲区所有权，最终物化
`memref.dealloc`。跨函数返回的缓冲区则由调用者接管所有权。

### 29. 为什么必须先 Lower MiniNPU，再做 Bufferization？

MiniNPU 自定义 Tensor 算子没有实现 `BufferizableOpInterface`。先执行
One-Shot Bufferize 会遇到未知、不可 Bufferize 的算子而失败。先转换到已提供
Bufferization Interface 的 Linalg/Tensor/Arith，内存化过程才有完整语义。
