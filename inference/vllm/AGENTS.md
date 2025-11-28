# vLLM Docker 部署 - AGENTS.md

## 概述

本目录提供 vLLM 的 Docker 部署脚本，支持快速启动 OpenAI 兼容的 LLM 推理服务。

**为什么用 Docker？**
- 环境隔离，避免依赖冲突
- 官方镜像开箱即用
- 支持 GPU 直通
- 便于生产部署

---

## 文件说明

| 文件 | 说明 |
|-----|------|
| `run_vllm.sh` | Linux/Mac 启动脚本 |
| `run_vllm.bat` | Windows 启动脚本 |

---

## 使用方法

### Linux/Mac

```bash
# 添加执行权限
chmod +x run_vllm.sh

# 运行（默认使用 Qwen2.5-0.5B-Instruct）
./run_vllm.sh

# 指定模型
./run_vllm.sh /path/to/your/model

# 指定 HuggingFace 模型
./run_vllm.sh Qwen/Qwen2.5-7B-Instruct
```

### Windows

```batch
# 运行
run_vllm.bat

# 指定模型
run_vllm.bat D:\models\Qwen2.5-7B-Instruct
```

---

## API 访问

启动后访问：
- **API 文档**: `http://localhost:8000/docs`
- **健康检查**: `http://localhost:8000/health`

### Python 调用示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM 默认不需要 API key
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # 模型名需与启动时一致
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

### cURL 调用

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 常用参数

脚本中可修改的参数：

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--model` | Qwen2.5-0.5B-Instruct | 模型路径或 HF 模型名 |
| `--port` | 8000 | API 端口 |
| `--tensor-parallel-size` | 1 | GPU 并行数（多卡时设置） |
| `--max-model-len` | 4096 | 最大上下文长度 |
| `--gpu-memory-utilization` | 0.9 | GPU 显存使用率 |

---

## 前置要求

1. **Docker** + **NVIDIA Container Toolkit**
   ```bash
   # 安装 nvidia-docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **NVIDIA GPU** 驱动已安装

3. **拉取镜像**
   ```bash
   docker pull vllm/vllm-openai:latest
   ```

---

## 常见问题

### Q: 显存不足
```bash
# 减少显存使用
--gpu-memory-utilization 0.7

# 或使用量化模型
./run_vllm.sh /path/to/model-awq --quantization awq
```

### Q: 多 GPU 部署
```bash
# 2 张 GPU tensor parallel
--tensor-parallel-size 2
```

### Q: 挂载本地模型
确保脚本中 `-v` 参数正确挂载模型目录。

---

## 注意事项

⚠️ 需要 NVIDIA GPU 和 nvidia-docker

⚠️ 首次运行会下载模型（如使用 HF 模型名）

⚠️ Windows 需要 WSL2 + Docker Desktop with GPU support
