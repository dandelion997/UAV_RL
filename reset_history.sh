#!/bin/bash

# 当前操作目录请务必是你的项目根目录
echo "⚠️ 警告：此操作将删除所有 Git 提交历史，只保留当前文件内容"
read -p "是否继续？(y/n): " confirm
if [ "$confirm" != "y" ]; then
  echo "❌ 已取消"
  exit 1
fi

# 设置匿名身份
git config user.name "anonymous"
git config user.email "anon@example.com"

# 创建备份目录
mkdir ../project_backup_before_reset
cp -r . ../project_backup_before_reset

# 删除旧 Git 历史
rm -rf .git
git init
git add .
git commit -m "Initial anonymous commit"

# 添加远程地址（替换为你的远程仓库）
git remote add origin https://github.com/dandelion997/UAV_RL.git

# 强制推送，覆盖远程历史
git push --force --set-upstream origin master

echo "✅ 历史已重写完成。GitHub 上将只看到匿名版本。"
