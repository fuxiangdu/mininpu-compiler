# WSL 开发环境

## 进入 WSL

在 Windows Terminal 或 PowerShell 中查看已安装的发行版：

```powershell
wsl -l -v
```

进入 Ubuntu（名称以命令输出为准）：

```powershell
wsl -d Ubuntu
```

出现类似下面的提示符即表示已经进入 Linux：

```text
fuxiangdu@MSI:~$
```

进入项目并确认路径：

```bash
cd ~/projects/mininpu-compiler
pwd
uname -a
```

退出 WSL 使用 `exit`。在 Windows 中完全停止所有 WSL 实例可执行
`wsl --shutdown`。

## 安装 LLVM/MLIR 18 环境

在 Ubuntu 24.04 WSL 中执行：

```bash
cd ~/projects/mininpu-compiler
bash scripts/setup_ubuntu_24_04.sh
```

该脚本安装 C/C++ 工具链、CMake、Ninja、LLVM/MLIR 18 开发包和 Python 3。
不要在 Windows PowerShell 中执行这个 Bash 脚本。

## 构建与完整回归

```bash
cd ~/projects/mininpu-compiler
chmod +x scripts/*.sh
bash scripts/run_v7.sh 2>&1 | tee regression_v7.log
```

完成标志：

```text
[PASS] MiniNPU compiler v7 completed
```

如果使用自行编译的 LLVM，而不是 `/usr/lib/llvm-18`，可显式指定：

```bash
export LLVM_ROOT="$HOME/toolchains/llvm-build-18"
export MLIR_DIR="$LLVM_ROOT/lib/cmake/mlir"
export LLVM_DIR="$LLVM_ROOT/lib/cmake/llvm"
export PATH="$LLVM_ROOT/bin:$PATH"
bash scripts/run_v7.sh
```

建议把代码和构建目录放在 WSL 的 Linux 文件系统（例如 `~/projects`），
不要放到 `/mnt/c`；大量 C++ 小文件的编译在 Linux 文件系统中通常更稳定。
