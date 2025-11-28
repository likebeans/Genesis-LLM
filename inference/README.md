# 模型推理与部署模块

## 概述

`inference/` 模块提供模型推理和部署的完整解决方案，包括：

- **vLLM 部署** - 高性能 LLM 推理服务（Docker 方式）
- **llama.cpp 转换** - 模型转 GGUF 格式 + 量化，支持 CPU/边缘设备部署
- **量化工具** - AWQ/GPTQ 等量化方法，减少显存占用

---

## 目录结构

```
inference/
├── vllm/              # vLLM Docker 部署脚本
│   ├── run_vllm.sh    # Linux/Mac 启动脚本
│   └── run_vllm.bat   # Windows 启动脚本
├── llama.cpp/         # GGUF 转换与量化
│   └── convert_to_gguf.py  # HF 模型 → GGUF 转换脚本
├── quantize/          # 训练后量化（PTQ）
│   ├── awq_quantize.py     # AWQ 量化脚本
│   └── gptq_quantize.py    # GPTQ 量化脚本
└── README.md          # 本文件
```

---

## 快速开始

### 1. vLLM 部署（推荐用于生产）

```bash
# Linux/Mac
cd inference/vllm
./run_vllm.sh

# Windows
cd inference\vllm
run_vllm.bat
```

访问 `http://localhost:8000/docs` 查看 OpenAI 兼容 API。

### 2. llama.cpp 转换（边缘部署）

```bash
# 转换 HF 模型为 GGUF
uv run python inference/llama.cpp/convert_to_gguf.py \
  --model_path /path/to/hf_model \
  --output_path /path/to/model.gguf \
  --quantize q4_k_m
```

### 3. AWQ/GPTQ 量化

```bash
# AWQ 量化（推荐）
uv run python inference/quantize/awq_quantize.py \
  --model_path /path/to/model \
  --output_path /path/to/model-awq \
  --bits 4

# GPTQ 量化
uv run python inference/quantize/gptq_quantize.py \
  --model_path /path/to/model \
  --output_path /path/to/model-gptq \
  --bits 4
```

---

## 方案对比

| 方案 | 适用场景 | 优势 | 依赖 |
|-----|---------|------|------|
| **vLLM** | 生产服务、高并发 | 高吞吐、PagedAttention | Docker + GPU |
| **llama.cpp** | 边缘设备、CPU 推理 | 轻量、跨平台 | llama.cpp 编译 |
| **AWQ** | GPU 推理、显存受限 | 精度损失小 | autoawq |
| **GPTQ** | GPU 推理、兼容性好 | 生态成熟 | auto-gptq |

---

## 依赖安装

```bash
# vLLM (Docker 方式，无需本地安装)
docker pull vllm/vllm-openai:latest

# llama.cpp 转换工具
pip install llama-cpp-python

# AWQ 量化
pip install autoawq

# GPTQ 量化
pip install auto-gptq optimum
```

---

## 注意事项

⚠️ vLLM Docker 需要 NVIDIA GPU 和 nvidia-docker

⚠️ llama.cpp 量化需要先编译 llama.cpp 或使用 Python 绑定

⚠️ AWQ/GPTQ 量化需要校准数据集（默认使用 C4）

⚠️ 量化后模型需要测试精度损失是否可接受
