#!/bin/bash
# vLLM Docker 启动脚本
# 用法: ./run_vllm.sh [模型路径或HF模型名]

set -e

# ========== 配置 ==========
MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}  # 默认模型
PORT=8000
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096
TENSOR_PARALLEL_SIZE=1

# ========== 判断模型来源 ==========
if [[ -d "$MODEL" ]]; then
    # 本地模型目录
    MODEL_PATH=$(realpath "$MODEL")
    MOUNT_ARGS="-v ${MODEL_PATH}:/model"
    MODEL_ARG="/model"
    echo "📦 使用本地模型: ${MODEL_PATH}"
else
    # HuggingFace 模型
    MOUNT_ARGS=""
    MODEL_ARG="$MODEL"
    echo "🤗 使用 HuggingFace 模型: ${MODEL}"
fi

# ========== HuggingFace 缓存挂载 ==========
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# ========== 启动容器 ==========
echo "🚀 启动 vLLM 服务..."
echo "   端口: ${PORT}"
echo "   GPU 显存使用率: ${GPU_MEMORY_UTILIZATION}"
echo "   最大上下文长度: ${MAX_MODEL_LEN}"
echo ""

docker run --rm -it \
    --gpus all \
    -p ${PORT}:8000 \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    ${MOUNT_ARGS} \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model "${MODEL_ARG}" \
    --served-model-name "${MODEL}" \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --trust-remote-code

echo ""
echo "✅ vLLM 服务已启动"
echo "   API 地址: http://localhost:${PORT}/v1"
echo "   文档地址: http://localhost:${PORT}/docs"
