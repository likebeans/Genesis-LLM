# 模型量化（AWQ/GPTQ）- AGENTS.md

## 概述

本目录提供训练后量化（Post-Training Quantization, PTQ）工具，支持：

- **AWQ（Activation-aware Weight Quantization）** - 基于激活值的权重量化，精度损失小
- **GPTQ（Generative Pre-trained Transformer Quantization）** - 基于 Hessian 的权重量化，生态成熟

两种方法都可将模型权重从 FP16 量化到 INT4/INT8，显著减少显存占用。

---

## 文件说明

| 文件 | 说明 |
|-----|------|
| `awq_quantize.py` | AWQ 量化脚本 |
| `gptq_quantize.py` | GPTQ 量化脚本 |

---

## AWQ vs GPTQ 对比

| 特性 | AWQ | GPTQ |
|-----|-----|------|
| 量化原理 | 基于激活值分布 | 基于 Hessian 矩阵 |
| 精度损失 | 较小 | 中等 |
| 量化速度 | 较快 | 较慢 |
| 推理速度 | 快（需要 AutoAWQ） | 快（需要 ExLlamaV2） |
| 生态支持 | vLLM, HF | vLLM, HF, text-generation-webui |
| 推荐场景 | 生产部署 | 兼容性需求 |

**推荐选择**：
- 优先使用 AWQ，精度保持更好
- 如需更广泛兼容性，使用 GPTQ

---

## AWQ 量化

### 安装依赖

```bash
pip install autoawq
```

### 使用方法

```bash
# 基本用法（4-bit 量化）
uv run python inference/quantize/awq_quantize.py \
  --model_path /path/to/model \
  --output_path /path/to/model-awq \
  --bits 4

# 指定量化参数
uv run python inference/quantize/awq_quantize.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --output_path ./qwen2.5-7b-awq \
  --bits 4 \
  --group_size 128 \
  --calib_data wikitext \
  --calib_samples 512
```

### 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--bits` | 4 | 量化位数（4 或 8） |
| `--group_size` | 128 | 量化分组大小 |
| `--calib_data` | wikitext | 校准数据集 |
| `--calib_samples` | 512 | 校准样本数 |
| `--zero_point` | True | 是否使用零点量化 |

---

## GPTQ 量化

### 安装依赖

```bash
pip install auto-gptq optimum
```

### 使用方法

```bash
# 基本用法（4-bit 量化）
uv run python inference/quantize/gptq_quantize.py \
  --model_path /path/to/model \
  --output_path /path/to/model-gptq \
  --bits 4

# 指定量化参数
uv run python inference/quantize/gptq_quantize.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --output_path ./qwen2.5-7b-gptq \
  --bits 4 \
  --group_size 128 \
  --desc_act true \
  --calib_samples 1024
```

### 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--bits` | 4 | 量化位数（2, 3, 4, 8） |
| `--group_size` | 128 | 量化分组大小 |
| `--desc_act` | True | 是否按激活值排序（精度更高但更慢） |
| `--calib_samples` | 1024 | 校准样本数 |
| `--damp_percent` | 0.1 | 阻尼系数 |

---

## 加载量化模型

### AWQ 模型

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized(
    "path/to/model-awq",
    fuse_layers=True,  # 融合层加速
)
tokenizer = AutoTokenizer.from_pretrained("path/to/model-awq")

# 推理
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### GPTQ 模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "path/to/model-gptq",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("path/to/model-gptq")

# 推理
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### vLLM 部署（支持 AWQ/GPTQ）

```bash
# AWQ 模型
./run_vllm.sh /path/to/model-awq --quantization awq

# GPTQ 模型
./run_vllm.sh /path/to/model-gptq --quantization gptq
```

---

## 量化后模型大小

以 7B 模型为例：

| 量化方式 | 模型大小 | 显存占用 |
|---------|---------|---------|
| FP16 | ~14 GB | ~16 GB |
| INT8 | ~7 GB | ~9 GB |
| INT4 (AWQ/GPTQ) | ~4 GB | ~6 GB |

---

## 注意事项

⚠️ 量化需要校准数据，影响最终精度

⚠️ 量化过程需要较大显存（加载完整模型）

⚠️ 量化后建议在验证集上测试精度

⚠️ 不同模型架构可能需要调整参数

---

## 常见问题

### Q: AWQ 量化后精度下降
- 增加 `--calib_samples` 校准样本数
- 尝试不同的 `--group_size`（64, 128, 256）

### Q: GPTQ 量化太慢
- 减少 `--calib_samples`
- 设置 `--desc_act false`（会略微降低精度）

### Q: 量化模型推理失败
- 确保使用对应的推理库（AutoAWQ / auto-gptq）
- 检查 CUDA 版本兼容性
