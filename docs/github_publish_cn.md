# GitHub 发布步骤

## 1. 在 GitHub 创建空仓库

仓库名建议使用：

```text
mininpu-compiler
```

选择 Public；不要在网页端额外创建 README、`.gitignore` 或 License，因为本
项目已经包含这些文件。

## 2. 本地初始化

```bash
cd ~/mininpu-compiler

git init
git add .
git commit -m "feat: add MiniNPU MLIR compiler pipeline"
git branch -M main
```

## 3. 关联远端并推送

将下面的 `YOUR_GITHUB_NAME` 替换为自己的 GitHub 用户名：

```bash
git remote add origin \
  https://github.com/YOUR_GITHUB_NAME/mininpu-compiler.git

git push -u origin main
```

GitHub 已停止接受账户密码进行 Git 推送。如果 HTTPS 要求认证，应使用浏览器
授权、Git Credential Manager 或 Personal Access Token，不要把 Token 写进脚本、
README 或提交记录。

## 4. 发布前检查

```bash
git status
git log --oneline -1
git remote -v
```

确认 GitHub Actions 页面中的 `MLIR 18 build and regression` 工作流通过后，再
把仓库链接加入简历。若 CI 失败，先保留本地 LLVM/MLIR 18.1.3 的通过证据，
根据 Actions 日志修复，不要删除负例测试来制造绿色结果。
