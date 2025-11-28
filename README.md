<div align="center">
  <img src="assets/logo.png" alt="Genesis-LLM Logo" width="400">
  
  <h1>Genesis-LLM</h1>
  
  <p><strong>中文/领域大模型训练全流程工具包</strong></p>
  
  <p>
    <a href="#快速开始">快速开始</a> •
    <a href="#功能模块">功能模块</a> •
    <a href="#项目结构">项目结构</a> •
    <a href="https://github.com/likebeans/Genesis-LLM">GitHub</a>
  </p>
  
  ![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
  ![Transformers](https://img.shields.io/badge/Transformers-4.40+-yellow.svg)
  ![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
</div>

---

## 目录

- [项目简介](#项目简介)
- [功能模块](#功能模块)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [License](#license)

---

## 项目简介

**Genesis-LLM** 是一个开箱即用的大语言模型训练工具包，旨在帮助开发者**快速复用**，无需每次从零编写训练脚本。

### 为什么需要这个项目？

- **避免重复造轮子** - 数据处理、模型训练、推理部署的脚本已经写好，修改配置即可使用
- **降低入门门槛** - 项目对 LLM 训练全流程进行清晰拆解，配合详细注释，帮助新手快速理解每个环节的原理
- **最佳实践沉淀** - 集成了 LoRA/QLoRA、DPO、vLLM 等主流技术，避免踩坑

### 适合谁？

- **LLM 初学者**：通过完整的训练流程示例，快速了解数据处理 → 词表扩充 → CPT → SFT → RLHF → 部署的全链路
- **算法工程师**：开箱即用的脚本和配置模板，专注于数据和模型调优，而非工程细节
- **研究人员**：灵活的配置系统，方便快速实验不同的训练策略

### 核心特性

- **数据处理管线** - HuggingFace 数据集采集、清洗、去重、质量筛选
- **词表扩充** - 中文/领域词表训练与合并，Embedding 扩展
- **高效微调** - 支持 LoRA/QLoRA 参数高效微调，显存占用低
- **RLHF 对齐** - 支持 DPO、PPO、GRPO 等对齐算法
- **配置驱动** - 所有参数通过 YAML 配置管理，易于复现
- **实验追踪** - 集成 WandB / SwanLab 实时监控
- **推理部署** - vLLM 高性能推理、llama.cpp 边缘部署、AWQ/GPTQ 量化

### 技术栈

`Python 3.10+` · `PyTorch 2.1+` · `Transformers` · `PEFT` · `TRL` · `SentencePiece` · `Datasets` · `Accelerate` · `vLLM`

---

## 功能模块

### 1. 数据处理 (data_process/)

从 HuggingFace Hub 采集数据，进行清洗、去重和质量筛选。

**采集数据**（保持原始格式，不做修改）：
```bash
uv run python data_process/fetch_dataset.py \
  --dataset_name shibing624/medical \
  --data_subdir pretrain \
  --output_dir data_process/fetch_data \
  --sample_limit 1000
```

**处理数据**（清洗、去重、过滤）：
```bash
uv run python data_process/process_data.py \
  --input data_process/fetch_data/pretrain/shibing624_medical__train.jsonl \
  --output_dir data_process/final_data \
  --subdir pretrain \
  --dedup --near_dedup \
  --to_parquet
```

**支持功能**：
- 精确去重（MD5 哈希）和相似去重（Jaccard 相似度）
- 长度过滤、情感过滤
- JSONL / Parquet 双格式输出

### 2. 词表扩充 (tokenizer/)

为中文/领域模型扩展词表，减少专业术语被拆分，降低 OOV 率。

**完整工作流**：
```bash
# 1. 提取基座 tokenizer 配置
uv run python tokenizer/aux_tokenizer/extract_tokenizer_config.py \
  --config config/tokenizer_config/config.yaml

# 2. 准备训练语料
uv run python tokenizer/data/fetch_and_clean.py \
  --config config/tokenizer_config/config.yaml

# 3. 训练辅助 tokenizer (SentencePiece)
uv run python tokenizer/aux_tokenizer/train_aux_tokenizer.py \
  --config config/tokenizer_config/config.yaml

# 4. 筛选新增 token
uv run python tokenizer/aux_tokenizer/select_new_tokens.py \
  --config config/tokenizer_config/config.yaml

# 5. 合并到基座 tokenizer
uv run python tokenizer/aux_tokenizer/extend_base_tokenizer.py \
  --config config/tokenizer_config/config.yaml

# 6. 扩展模型 Embedding
uv run python tokenizer/aux_tokenizer/resize_embeddings.py \
  --config config/tokenizer_config/config.yaml
```

### 3. 模型训练 (self_model/)

支持完整的 LLM 训练流程：继续预训练（CPT）→ 监督微调（SFT）→ RLHF 对齐。

**继续预训练（CPT）**：
```bash
uv run python self_model/pretrain/train_pretrain.py \
  --config config/self_model_config/pretrain.yaml \
  --trace swanlab
```

**监督微调（SFT）** - 支持 LoRA/QLoRA：
```bash
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --trace swanlab
```

**RLHF 对齐**：
```bash
# DPO 训练
uv run python self_model/rlhf/train_dpo.py \
  --config config/self_model_config/rlhf.yaml

# PPO/GRPO 训练 (基于 VERL)
uv run python self_model/rlhf/train_ppo.py \
  --config config/self_model_config/rlhf.yaml
```

**支持的对话模板**：Qwen2 (ChatML)、Llama3、ChatGLM3、Baichuan2、DeepSeek 等。

### 4. 模型推理与部署 (inference/)

提供多种推理和部署方案。

**vLLM 高性能推理**（推荐用于生产）：
```bash
cd inference/vllm && ./run_vllm.sh
# 访问 http://localhost:8000/docs 使用 OpenAI 兼容 API
```

**llama.cpp 边缘部署**：
```bash
uv run python inference/llama.cpp/convert_to_gguf.py \
  --model_path /path/to/model \
  --output_path /path/to/model.gguf \
  --quantize q4_k_m
```

**AWQ/GPTQ 量化**：
```bash
# AWQ 量化（推荐，精度损失小）
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

| 方案 | 适用场景 | 优势 |
|-----|---------|------|
| **vLLM** | 生产服务、高并发 | 高吞吐、PagedAttention |
| **llama.cpp** | 边缘设备、CPU 推理 | 轻量、跨平台 |
| **AWQ/GPTQ** | GPU 推理、显存受限 | 4-bit 量化，显存占用减少 70% |

---

## 快速开始

### 1. 环境安装

```bash
# 推荐使用 uv（更快）
uv sync

# 或使用 pip
pip install -e .
```

### 2. 完整训练流程示例

```bash
# Step 1: 采集数据
uv run python data_process/fetch_dataset.py \
  --dataset_name shibing624/medical \
  --data_subdir finetune \
  --output_dir data_process/fetch_data

# Step 2: 处理数据
uv run python data_process/process_data.py \
  --input data_process/fetch_data/finetune/shibing624_medical__train.jsonl \
  --output_dir data_process/final_data \
  --subdir finetune \
  --dedup

# Step 3: LoRA 微调
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --trace swanlab

# Step 4: 合并 LoRA 权重
uv run python self_model/merge_lora.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --lora_path self_model/checkpoints/finetune \
  --output_path self_model/checkpoints/finetune_merged
```

### 3. Docker 部署（可选）

```bash
# 构建镜像
docker build -t genesis-llm:dev -f docker/Dockerfile .

# 运行训练
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  genesis-llm:dev

# 或使用 docker-compose
docker-compose -f docker/docker-compose.yml up -d dev
```

---

## 项目结构

```
Genesis-LLM/
├── config/                      # 配置文件中心
│   ├── model_config/           # HuggingFace 镜像配置
│   ├── self_model_config/      # 训练配置 (pretrain/finetune/rlhf)
│   ├── tokenizer_config/       # 词表扩充配置
│   └── trace_config/           # 实验追踪配置 (wandb/swanlab)
├── data_process/               # 数据处理管线
│   ├── fetch_dataset.py       # 数据采集脚本
│   ├── process_data.py        # 数据清洗脚本
│   ├── fetch_data/            # 原始数据目录
│   └── final_data/            # 处理后数据目录
├── tokenizer/                  # 词表扩充模块
│   ├── aux_tokenizer/         # 辅助词表训练脚本
│   ├── data/                  # 词表训练语料
│   ├── evaluate/              # 词表评估脚本
│   ├── merged_tokenizer/      # 扩展后的 tokenizer
│   └── merged_model/          # 扩展 Embedding 后的模型
├── self_model/                 # 模型训练模块
│   ├── fine_tuning/           # 监督微调 (SFT)
│   ├── pretrain/              # 继续预训练 (CPT)
│   ├── rlhf/                  # RLHF 对齐 (DPO/PPO/GRPO)
│   ├── template/              # 对话模板
│   └── checkpoints/           # 训练检查点
├── inference/                  # 推理与部署
│   ├── vllm/                  # vLLM Docker 部署
│   ├── llama.cpp/             # GGUF 转换
│   └── quantize/              # AWQ/GPTQ 量化
└── docker/                     # Docker 配置
    ├── Dockerfile             # 训练环境镜像
    └── docker-compose.yml     # 多服务编排
```

---

## 环境要求

| 组件 | 要求 |
|------|------|
| Python | >= 3.10 |
| PyTorch | >= 2.1.0 |
| CUDA | >= 11.8（GPU 训练） |
| 显存 | >= 8GB（LoRA）/ >= 24GB（全参微调） |

**核心依赖**：
- `transformers>=4.57.3` - HuggingFace Transformers
- `peft>=0.18.0` - 参数高效微调
- `trl>=0.25.1` - RLHF 训练
- `datasets>=4.4.1` - 数据集加载
- `accelerate>=1.12.0` - 分布式训练
- `sentencepiece>=0.2.1` - 词表训练
- `wandb>=0.23.0` / `swanlab>=0.7.2` - 实验追踪

---

## 配置说明

所有训练参数通过 YAML 配置管理，位于 `config/` 目录：

```yaml
# config/self_model_config/finetune.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
tokenizer_name_or_path: Qwen/Qwen2.5-0.5B-Instruct

data:
  train_file: data_process/final_data/finetune/train.jsonl
  eval_file: data_process/final_data/finetune/validation.jsonl
  max_length: 512

peft:
  enabled: true      # 启用 LoRA
  qlora: false       # 启用 QLoRA (4-bit)
  lora_r: 8
  lora_alpha: 16
  target_modules: [q_proj, k_proj, v_proj, o_proj]

trace: swanlab       # 实验追踪：wandb / swanlab / none

training_args:
  output_dir: self_model/checkpoints/finetune
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  num_train_epochs: 1
  fp16: true
```

**命令行覆盖配置**：
```bash
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --trace wandb  # 覆盖 YAML 中的 trace 设置
```

---

## 常见问题

### Q: 如何使用国内 HuggingFace 镜像？

在 `config/model_config/config.yaml` 中配置：
```yaml
huggingface:
  endpoint: https://hf-mirror.com
```

### Q: LoRA 和 QLoRA 怎么选？

- **LoRA**：显存 >= 16GB，训练速度快
- **QLoRA**：显存 8-16GB，使用 4-bit 量化基座模型

在配置中设置 `peft.qlora: true` 即可启用 QLoRA。

### Q: 如何进行多卡训练？

```bash
# 使用 torchrun
torchrun --nproc_per_node=4 \
  self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml

# 使用 DeepSpeed
deepspeed --num_gpus=4 \
  self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --deepspeed config/self_model_config/deepspeed/ds_config.json
```

### Q: 训练完成后如何推理？

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载合并后的模型
model = AutoModelForCausalLM.from_pretrained("self_model/checkpoints/finetune_merged")
tokenizer = AutoTokenizer.from_pretrained("self_model/checkpoints/finetune_merged")

# 推理
inputs = tokenizer("你好", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## License

Apache License 2.0

---

## 致谢

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
