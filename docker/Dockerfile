# Self Model SOP - 开发/训练环境
# ============================================================
# 用途：统一开发环境、训练环境复现
# 
# 构建镜像：
#   docker build -t self-model-sop:dev -f docker/Dockerfile .
#
# 运行容器（GPU 训练）：
#   docker run --gpus all -it --rm \
#     -v $(pwd):/workspace \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     self-model-sop:dev
#
# 注意：
#   - vLLM 推理部署请使用 inference/vllm/run_vllm.sh（官方镜像）
#   - 本镜像主要用于开发和训练场景
# ============================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（推荐的包管理器）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# 复制项目文件
COPY pyproject.toml uv.lock ./

# 安装 Python 依赖
RUN uv sync --frozen

# 复制其余项目文件
COPY . .

# 默认命令
CMD ["/bin/bash"]
