# Docker 部署指南

## 概述

本目录提供 Docker 配置，用于：

1. **开发环境统一**：确保团队成员使用相同的环境
2. **训练环境复现**：方便在不同机器上复现训练结果
3. **推理服务部署**：快速启动 vLLM 推理服务

---

## 快速开始

### 1. 开发/训练环境

```bash
# 构建镜像
docker build -t self-model-sop:dev -f docker/Dockerfile .

# 运行容器（GPU 训练）
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  self-model-sop:dev

# 在容器内运行训练
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml
```

### 2. 使用 Docker Compose

```bash
# 启动开发环境
docker-compose -f docker/docker-compose.yml up -d dev

# 进入容器
docker-compose -f docker/docker-compose.yml exec dev bash

# 停止
docker-compose -f docker/docker-compose.yml down
```

### 3. vLLM 推理服务

```bash
# 启动 vLLM 服务（需要先有训练好的模型）
docker-compose -f docker/docker-compose.yml --profile inference up -d vllm

# 测试 API
curl http://localhost:8000/v1/models

# 发送请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "self-model",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `Dockerfile` | 开发/训练环境镜像定义 |
| `docker-compose.yml` | 多服务编排配置 |
| `README.md` | 本文件 |

---

## 注意事项

⚠️ **GPU 支持**：需要安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

⚠️ **HuggingFace 缓存**：建议挂载 `~/.cache/huggingface` 避免重复下载模型

⚠️ **vLLM 模型路径**：使用前需修改 `docker-compose.yml` 中的模型路径

---

## 何时使用 Docker？

| 场景 | 推荐方式 |
|------|---------|
| 本地开发 | 直接使用 `uv sync` 安装依赖 |
| 团队协作 | Docker 确保环境一致 |
| 生产训练 | Docker 便于环境复现 |
| 推理部署 | vLLM Docker 镜像 |

**结论**：Docker 是可选的，主要用于环境一致性和推理部署。本地开发建议直接使用 uv。
