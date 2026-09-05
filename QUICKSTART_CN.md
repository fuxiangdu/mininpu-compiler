# 快速开始

## 环境

- Ubuntu 24.04 x86-64
- Clang/LLVM/MLIR 18.1.3
- CMake 3.28、Ninja 1.11、Python 3

## 构建

```bash
cd ~/mininpu-compiler
chmod +x scripts/*.sh
set -o pipefail
bash scripts/01_build.sh 2>&1 | tee build.log
echo "BUILD_EXIT_CODE=$?"
```

## 全量测试

```bash
set -o pipefail
bash scripts/run_v4.sh 2>&1 | tee regression.log
echo "TEST_EXIT_CODE=$?"
```

成功标志：

```text
[PASS] MiniNPU compiler v4 completed
TEST_EXIT_CODE=0
```

## 只运行各阶段

```bash
bash scripts/02_test.sh
bash scripts/03_test_dialect.sh
bash scripts/04_test_fusion.sh
bash scripts/05_test_tiling.sh
bash scripts/06_test_lowering.sh
```

`invalid MatMul`、`128-byte UB` 和 `wrong pipeline` 测试中的 error 是程序刻意
触发并检查的负例；只要脚本最后输出对应 PASS，它们就不是构建故障。
