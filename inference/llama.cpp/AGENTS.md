# llama.cpp GGUF 转换与量化 - AGENTS.md

## 概述

本目录提供将 HuggingFace 模型转换为 GGUF 格式并进行量化的完整流程，支持：

- **HF → GGUF 转换** - 将 Transformers 模型转为 llama.cpp 格式
- **量化** - 支持多种量化精度（Q4_K_M, Q5_K_M, Q8_0 等）
- **边缘部署** - 转换后可在 CPU/低显存设备上运行

---

## 文件说明

| 文件 | 说明 |
|-----|------|
| `convert_to_gguf.py` | HF 模型 → GGUF 转换脚本 |

---

## 完整工作流程

### 1. 安装依赖

```bash
# 方式一：使用 llama-cpp-python（推荐）
uv add llama-cpp-python

# 方式二：编译 llama.cpp（更灵活）
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
uv pip install -r requirements.txt  # 用于转换脚本
```

### 2. 转换模型

```bash
# 基本用法
uv run python inference/llama.cpp/convert_to_gguf.py \
  --model_path /path/to/hf_model \
  --output_path /path/to/model.gguf

# 转换并量化（推荐 Q4_K_M）
uv run python inference/llama.cpp/convert_to_gguf.py \
  --model_path /path/to/hf_model \
  --output_path /path/to/model-q4km.gguf \
  --quantize q4_k_m

# 指定 llama.cpp 路径（如果使用编译版本）
uv run python inference/llama.cpp/convert_to_gguf.py \
  --model_path /path/to/hf_model \
  --output_path /path/to/model.gguf \
  --llama_cpp_path /path/to/llama.cpp
```

### 3. 运行推理

```bash
# 使用 llama.cpp CLI
./llama.cpp/main -m model-q4km.gguf -p "Hello, world!" -n 128

# 或使用 Python
python -c "
from llama_cpp import Llama
llm = Llama(model_path='model-q4km.gguf')
output = llm('Hello, world!', max_tokens=128)
print(output['choices'][0]['text'])
"
```

---

## 量化选项

| 量化类型 | 大小压缩比 | 精度损失 | 推荐场景 |
|---------|-----------|---------|---------|
| `q8_0` | ~50% | 极小 | 高精度需求 |
| `q5_k_m` | ~35% | 较小 | 平衡精度与大小 |
| `q4_k_m` | ~25% | 中等 | **推荐**，性价比最高 |
| `q4_0` | ~25% | 中等 | 兼容性好 |
| `q3_k_m` | ~20% | 较大 | 极限压缩 |
| `q2_k` | ~15% | 大 | 实验性质 |

**推荐选择**：
- 7B 模型：`q4_k_m`（约 4GB）
- 13B 模型：`q4_k_m`（约 8GB）或 `q5_k_m`
- 70B 模型：`q4_k_m`（约 40GB）

---

## 转换流程详解

```
HuggingFace 模型
       │
       ▼
┌─────────────────┐
│ convert.py      │  ← llama.cpp 官方转换脚本
│ (HF → GGUF)     │
└─────────────────┘
       │
       ▼
   FP16 GGUF
       │
       ▼
┌─────────────────┐
│ quantize        │  ← llama.cpp 量化工具
│ (FP16 → Q4_K_M) │
└─────────────────┘
       │
       ▼
  量化后 GGUF
```

---

## 支持的模型架构

- **LLaMA 系列** - LLaMA 1/2/3, Alpaca, Vicuna
- **Qwen 系列** - Qwen, Qwen2
- **Mistral 系列** - Mistral, Mixtral
- **其他** - Phi, Gemma, Yi, DeepSeek 等

> 注意：部分新架构可能需要最新版 llama.cpp 支持

---

## 使用 llama-cpp-python

```python
from llama_cpp import Llama

# 加载量化模型
llm = Llama(
    model_path="model-q4km.gguf",
    n_ctx=4096,         # 上下文长度
    n_threads=8,        # CPU 线程数
    n_gpu_layers=35,    # GPU 层数（0 = 纯 CPU）
)

# 文本补全
output = llm(
    "The meaning of life is",
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
)
print(output["choices"][0]["text"])

# Chat 模式
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
)
print(output["choices"][0]["message"]["content"])
```

---

## 注意事项

⚠️ 转换需要足够内存（约为模型大小的 2 倍）

⚠️ 量化会有精度损失，建议转换后测试

⚠️ 部分模型需要 `--trust-remote-code`

⚠️ llama.cpp 版本需与模型架构兼容

---

## 常见问题

### Q: 转换失败 - 不支持的架构
更新 llama.cpp 到最新版本，或检查模型是否被支持。

### Q: 量化后精度下降严重
尝试使用更高精度的量化类型（如 `q5_k_m` 或 `q8_0`）。

### Q: CPU 推理太慢
- 确保使用了 AVX2/AVX512 编译
- 增加线程数 `n_threads`
- 考虑使用 GPU 加速 `n_gpu_layers`
