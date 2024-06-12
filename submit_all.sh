#!/bin/bash

# 检查是否提供了任务数量参数
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_tasks>"
  exit 1
fi

# 提交指定数量的 submit.sh 文件
for i in $(seq 1 $1); do
    sbatch submit$i.sh
done
