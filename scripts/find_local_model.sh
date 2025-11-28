#!/bin/bash
# 查找 HuggingFace 模型的本地缓存路径
# 用法: 
#   ./find_local_model.sh <模型名>        # 返回指定模型的本地路径
#   ./find_local_model.sh --list          # 列出所有本地缓存的模型
#   ./find_local_model.sh -l              # 同上
# 示例: 
#   ./find_local_model.sh Qwen/Qwen2.5-0.5B-Instruct

set -e

# HuggingFace 缓存目录
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

# 列出所有本地模型
list_models() {
    echo "📦 本地缓存的 HuggingFace 模型:"
    echo "================================"
    
    if [ ! -d "$HF_CACHE" ]; then
        echo "❌ 缓存目录不存在: $HF_CACHE"
        exit 1
    fi
    
    for dir in "$HF_CACHE"/models--*; do
        if [ -d "$dir" ]; then
            # 转换目录名为模型名 (models--Qwen--Qwen2.5-0.5B-Instruct -> Qwen/Qwen2.5-0.5B-Instruct)
            model_name=$(basename "$dir" | sed 's/^models--//' | sed 's/--/\//g')
            
            # 获取大小
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            
            # 获取 snapshot 数量
            snapshot_count=$(ls "$dir/snapshots" 2>/dev/null | wc -l)
            
            echo "  $model_name  ($size, $snapshot_count snapshot(s))"
        fi
    done
    echo "================================"
}

# 处理参数
MODEL_NAME=${1:-""}

if [ -z "$MODEL_NAME" ]; then
    echo "用法:"
    echo "  $0 <模型名>    返回指定模型的本地路径"
    echo "  $0 --list      列出所有本地缓存的模型"
    echo "  $0 -l          同上"
    echo ""
    echo "示例:"
    echo "  $0 Qwen/Qwen2.5-0.5B-Instruct"
    exit 1
fi

# 如果是 --list 或 -l 参数，列出所有模型
if [ "$MODEL_NAME" = "--list" ] || [ "$MODEL_NAME" = "-l" ]; then
    list_models
    exit 0
fi

# 转换模型名为缓存目录格式 (Qwen/Qwen2.5-0.5B-Instruct -> models--Qwen--Qwen2.5-0.5B-Instruct)
CACHE_NAME="models--${MODEL_NAME//\//--}"

MODEL_DIR="${HF_CACHE}/${CACHE_NAME}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 未找到本地缓存: $MODEL_NAME" >&2
    echo "   期望路径: $MODEL_DIR" >&2
    exit 1
fi

# 查找 snapshots 目录下的最新版本
SNAPSHOTS_DIR="${MODEL_DIR}/snapshots"

if [ ! -d "$SNAPSHOTS_DIR" ]; then
    echo "❌ 未找到 snapshots 目录: $SNAPSHOTS_DIR" >&2
    exit 1
fi

# 获取最新的 snapshot（按修改时间排序）
LATEST_SNAPSHOT=$(ls -t "$SNAPSHOTS_DIR" | head -1)

if [ -z "$LATEST_SNAPSHOT" ]; then
    echo "❌ snapshots 目录为空" >&2
    exit 1
fi

LOCAL_PATH="${SNAPSHOTS_DIR}/${LATEST_SNAPSHOT}"

echo "$LOCAL_PATH"
