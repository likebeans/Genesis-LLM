# 模型训练模块 (self_model/) - AGENTS.md

## 模块概览

`self_model/` 模块是 LLM 训练的核心模块，负责**继续预训练（CPT）**、**监督微调（SFT）**和 **RLHF 对齐**三个阶段的训练工作。

**技术栈**：
- **CPT/SFT**: `transformers` + `peft` (LoRA/QLoRA)
- **RLHF**: `verl` + `transformers` (DPO/PPO/GRPO)
- **实验追踪**: WandB / SwanLab
- **配置管理**: YAML 驱动，所有参数集中在 `config/self_model_config/`

**设计理念**：
- 配置驱动：所有训练参数写在 YAML 文件中，脚本读取后直接开训
- 模块化：CPT、SFT、RLHF 分离，各自独立运行
- PEFT 优先：默认使用 LoRA/QLoRA 进行参数高效微调
- 可观测性：集成 WandB/SwanLab 实时监控训练过程

---

## 目录结构

```
self_model/
├── fine_tuning/           # 监督微调（SFT）脚本
│   ├── train_finetune.py # SFT 训练入口
│   └── ...               # 其他辅助脚本
├── pretrain/              # 继续预训练（CPT）脚本
│   ├── train_pretrain.py # CPT 训练入口
│   └── README.md
├── rlhf/                  # RLHF 对齐脚本
│   ├── train_dpo.py      # DPO 训练（已实现）
│   ├── train_ppo.py      # PPO 训练（VERL）
│   └── train_grpo.py     # GRPO 训练（VERL）
├── template/              # 对话模板（ChatML 等）
│   ├── qwen2_chatml.json # Qwen2 ChatML 模板
│   ├── llama3_instruct.json # Llama3 Instruct 模板
│   ├── chatglm3.json     # ChatGLM3 模板
│   ├── baichuan2_chat.json # Baichuan2-Chat 模板
│   └── deepseek_chat.json # DeepSeek-LLM-Chat 模板
├── checkpoints/           # 训练输出目录
│   ├── finetune/         # SFT 检查点
│   ├── pretrain/         # CPT 检查点
│   └── rlhf/             # RLHF 检查点
└── README.md              # 原始开发文档
```

---

## 配置文件

所有训练参数集中在 `config/self_model_config/` 目录：

| 配置文件 | 用途 | 状态 |
|---------|------|------|
| `pretrain.yaml` | 继续预训练（CPT）配置 | ✅ 基础版 |
| `finetune.yaml` | 监督微调（SFT/PEFT）配置 | ✅ 已完成 |
| `rlhf.yaml` | RLHF 对齐（DPO/PPO/GRPO）配置 | ✅ DPO 模板 |

**配置示例** (`finetune.yaml`):
```yaml
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
tokenizer_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
template_path: self_model/template/qwen2_chatml.json

data:
  train_file: data_process/final_data/finetune/train.jsonl
  eval_file: data_process/final_data/finetune/validation.jsonl
  max_length: 512

peft:
  enabled: true
  lora_r: 8
  lora_alpha: 16
  target_modules: [q_proj, k_proj, v_proj, o_proj]

trace: swanlab  # wandb/swanlab/none

training_args:
  output_dir: self_model/checkpoints/finetune
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  num_train_epochs: 1
  fp16: true
```

---

## 训练流程

### 1. 继续预训练（CPT）✅

**功能**：在领域或大规模无监督数据上继续预训练，增强模型的基础知识与风格对齐。

**使用方法**：
```bash
# 基础 CPT 训练
uv run python self_model/pretrain/train_pretrain.py \
  --config config/self_model_config/pretrain.yaml

# torchrun 多卡
torchrun --nproc_per_node=4 \
  self_model/pretrain/train_pretrain.py \
  --config config/self_model_config/pretrain.yaml

# DeepSpeed ZeRO
deepspeed --num_gpus=4 \
  self_model/pretrain/train_pretrain.py \
  --config config/self_model_config/pretrain.yaml \
  --deepspeed config/self_model_config/deepspeed/ds_config.json
```

**关键参数**：
- `model_name_or_path` / `tokenizer_name_or_path`: 基座模型与分词器
- `data.train_file` / `data.eval_file`: 纯文本或 JSONL（包含 `text` 或配置的 `text_fields`）
- `data.text_fields`: 指定优先读取的字段列表，缺失时自动拼接 messages/conversations
- `data.block_size`: 文本拼接后切块长度
- `peft.enabled` / `peft.qlora`: 低显存场景可选 LoRA/QLoRA
- `training_args.*`: Transformers 训练超参（batch size、lr、save_steps 等）

**输出**：
- 训练后的模型/适配器保存在 `output_dir` 指定的目录
- 训练日志上传到 WandB/SwanLab（如果启用）

---

### 2. 监督微调（SFT）✅

**功能**：在预训练模型基础上进行指令微调，使模型学会遵循指令。

**使用方法**：
```bash
# 基础 SFT 训练
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml

# 使用 SwanLab 追踪
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --trace swanlab

# 使用 WandB 追踪
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --trace wandb
```

**关键参数**：
- `model_name_or_path`: 基础模型路径（可使用扩展词表后的模型）
- `tokenizer_name_or_path`: Tokenizer 路径
- `template_path`: 对话模板路径
- `data.train_file`: 训练数据（JSONL 格式）
- `data.eval_file`: 验证数据
- `peft.enabled`: 是否启用 LoRA
- `peft.lora_r`: LoRA 秩
- `peft.target_modules`: LoRA 目标模块

**输出**：
- 训练后的模型/适配器保存在 `output_dir` 指定的目录
- 训练日志上传到 WandB/SwanLab（如果启用）

---

### 3. RLHF 对齐（DPO + PPO/GRPO）

**功能**：利用人类偏好数据提升模型有用性/安全性。

**当前支持**：
- **DPO**: `train_dpo.py`（TRL）
- **PPO/GRPO**: `train_ppo.py` / `train_grpo.py`（VERL，依赖 ray + vLLM/HF）

**使用方法（示例）**：
```bash
# PPO
uv run python self_model/rlhf/train_ppo.py \
  --config config/self_model_config/rlhf.yaml

# GRPO
uv run python self_model/rlhf/train_grpo.py \
  --config config/self_model_config/rlhf.yaml

# DPO
uv run python self_model/rlhf/train_dpo.py \
  --config config/self_model_config/rlhf.yaml
```

**关键参数**：
- 通用：`model_name_or_path` / `tokenizer_name_or_path` / `reward_model_name_or_path`（PPO/GRPO）
- DPO：`data.train_file`、`beta`、`loss_type`、`training_args.*`
- PPO/GRPO（`ppo.*` 节）：`train_file`/`eval_file`、`max_prompt_length`/`max_response_length`、`rollout_backend`（vllm/hf）、`rollout_n`（GRPO >=4 常用）、`actor_lr`、`ppo_mini_batch_size`、`ppo_micro_batch_size_per_gpu`、`kl_loss_coef`、`use_kl_loss`（GRPO 自动启用）、`logger`、`n_gpus_per_node`/`nnodes`

---

## 训练输出（checkpoints/）

训练完成后，模型检查点会保存在 `self_model/checkpoints/` 目录下：

```
checkpoints/
├── finetune/              # SFT 检查点
│   ├── checkpoint-1000/  # 中间检查点
│   ├── checkpoint-2000/
│   └── final/            # 最终模型
├── pretrain/              # CPT 检查点
└── rlhf/                  # RLHF 检查点
```

**检查点内容**：
- **全量微调**：完整的模型权重 + Tokenizer + 配置文件
- **PEFT (LoRA)**：LoRA 适配器权重 + Tokenizer + 配置文件

**使用检查点**：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载 LoRA 模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "self_model/checkpoints/finetune/final")
tokenizer = AutoTokenizer.from_pretrained("self_model/checkpoints/finetune/final")
```

---

## 对话模板（template/）

对话模板定义了多轮对话的格式化方式，用于 SFT 训练和推理。

**当前支持**：
- `qwen2_chatml.json`: Qwen2 系列的 ChatML 格式

**模板格式示例**：
```json
{
  "system_prefix": "<|im_start|>system\n",
  "system_suffix": "<|im_end|>\n",
  "user_prefix": "<|im_start|>user\n",
  "user_suffix": "<|im_end|>\n",
  "assistant_prefix": "<|im_start|>assistant\n",
  "assistant_suffix": "<|im_end|>\n"
}
```

**扩展模板**：
可根据不同模型的对话格式添加新的模板文件（如 Llama3、ChatGLM 等）。

---

## 实验追踪

### WandB 配置

在 `config/trace_config/wandb.yaml` 中配置：
```yaml
project: self-model-training
entity: your-team
api_key: your-api-key
mode: online  # online/offline/disabled
```

### SwanLab 配置

在 `config/trace_config/swanlab.yaml` 中配置：
```yaml
project: self-model-training
workspace: your-workspace
api_key: your-api-key
mode: cloud  # cloud/local/offline
```

### 追踪指标

训练过程中会自动记录：
- **Loss**: 训练损失和验证损失
- **Learning Rate**: 学习率变化曲线
- **Gradient Norm**: 梯度范数
- **Perplexity**: 困惑度（语言模型评估指标）
- **Training Speed**: 训练速度（samples/sec）
- **GPU Utilization**: GPU 利用率

---

## 最佳实践

### 1. 训练前准备

✅ **推荐做法**：
- 确保数据已完成清洗和去重（使用 `data_process/` 模块）
- 如果需要领域词表，先完成 Tokenizer 扩展（使用 `tokenizer/` 模块）
- 在 YAML 中明确指定所有路径（模型、数据、输出）
- 先用小数据集测试训练流程，确认无误后再全量训练

❌ **避免做法**：
- 不要在命令行中硬编码参数，统一使用 YAML 配置
- 不要跳过数据质量检查直接训练
- 不要在训练过程中修改配置文件

### 2. PEFT 训练建议

**LoRA 参数选择**：
- `lora_r`: 通常选择 8、16、32（越大表达能力越强，但训练成本越高）
- `lora_alpha`: 通常设为 `lora_r` 的 2 倍
- `target_modules`: 
  - 最小配置：`[q_proj, v_proj]`
  - 推荐配置：`[q_proj, k_proj, v_proj, o_proj]`
  - 完整配置：`[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`

**训练超参建议**：
- Learning Rate: 5e-5 ~ 1e-4（LoRA 可以用更高的学习率）
- Batch Size: 根据 GPU 显存调整，配合梯度累积
- Epochs: 通常 1-3 个 epoch 即可（避免过拟合）

### 3. 训练监控

**关键指标**：
- **Training Loss 下降**：正常训练应该持续下降
- **Validation Loss 稳定**：如果验证损失不再下降或上升，考虑早停
- **Perplexity 降低**：困惑度越低，模型生成质量越好
- **GPU 利用率**：应保持在 80% 以上

**异常处理**：
- Loss 爆炸（NaN）：降低学习率或检查数据质量
- Loss 不下降：增大学习率或检查数据格式
- 显存溢出：减小 batch size 或使用梯度检查点

---

## 依赖项

核心依赖（见 `pyproject.toml`）：
- `torch>=2.1.0` - PyTorch 框架
- `transformers>=4.57.3` - HuggingFace Transformers
- `peft>=0.18.0` - 参数高效微调
- `accelerate>=1.12.0` - 分布式训练加速
- `datasets>=4.4.1` - 数据集加载
- `wandb>=0.23.0` - 实验追踪（可选）
- `swanlab>=0.7.2` - 实验追踪（可选）
- `verl` - RLHF 训练框架（PPO/GRPO）

---

## 相关文档

- [项目根目录 AGENTS.md](../AGENTS.md) - 项目整体说明
- [配置目录 AGENTS.md](../config/AGENTS.md) - 配置文件详解
- [数据处理 AGENTS.md](../data_process/AGENTS.md) - 数据采集和处理
- [词表扩充文档](../tokenizer/扩充词表开发方案.md) - Tokenizer 扩展流程

---

## 注意事项

⚠️ **模型路径**：
- 优先使用扩展词表后的模型：`tokenizer/merged_tokenizer`、`tokenizer/merged_model`
- 确保 Tokenizer 和模型路径一致，避免词表不匹配

⚠️ **数据格式**：
- CPT 数据需包含 `text` 或 `data.text_fields` 指定的字段（JSONL/TXT）
- SFT 数据必须包含 `input` 和 `output` 字段（或按模板定义的字段）
- RLHF 数据格式取决于算法（DPO 需要偏好对，PPO 需要 Prompt 和 Reward）

⚠️ **训练顺序**：
- 推荐顺序：CPT（可选）→ SFT → RLHF
- RLHF 必须在 SFT 之后进行，使用 SFT 的检查点作为初始化

⚠️ **检查点管理**：
- 定期清理旧检查点，避免占用过多磁盘空间
- 重要检查点建议备份到云存储

---

## 开发计划

### 待完成功能

1. **继续预训练（CPT）**
   - [x] 实现 `pretrain/train_pretrain.py`
   - [x] 完善 `config/self_model_config/pretrain.yaml`
   - [ ] 添加 MLM 训练模式（BERT/编码器场景）

2. **RLHF 对齐**
   - [x] 集成 VERL 框架（PPO/GRPO）
   - [x] 实现 DPO 训练脚本
   - [x] 实现 PPO 训练脚本
   - [x] 实现 GRPO 训练脚本
   - [x] 完善 `config/self_model_config/rlhf.yaml`

3. **评估与推理**
   - [ ] 集成 lm-eval 或 OpenCompass 评估框架
   - [ ] 添加推理 demo（vLLM/transformers）
   - [ ] 添加模型对比评估脚本

4. **工具与优化**
   - [ ] 添加模型合并脚本（LoRA → 全量模型）
   - [ ] 添加模型量化脚本（INT8/INT4）
   - [ ] 支持多 GPU 训练（DDP/FSDP）

---

## 快速开始

### 1. 准备数据
```bash
# 采集数据
uv run python data_process/fetch_dataset.py \
  --dataset_name shibing624/medical \
  --data_subdir finetune

# 清洗数据
uv run python data_process/process_data.py \
  --input data_process/fetch_data/finetune/shibing624_medical__train.jsonl \
  --output_dir data_process/final_data \
  --subdir finetune \
  --dedup --near_dedup
```

### 2. 配置训练参数
编辑 `config/self_model_config/finetune.yaml`，设置模型路径、数据路径、训练超参等。

### 3. 开始训练
```bash
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --trace swanlab
```

### 4. 监控训练
访问 SwanLab/WandB 控制台查看训练曲线和指标。

### 5. 使用模型
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "self_model/checkpoints/finetune/final")
tokenizer = AutoTokenizer.from_pretrained("self_model/checkpoints/finetune/final")

# 推理
inputs = tokenizer("你好，请介绍一下糖尿病。", return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0]))
```

---

## 常见问题

### Q1: 如何选择 LoRA 还是全量微调？

**LoRA（推荐）**：
- 优点：训练快、显存占用小、易于管理
- 适用场景：大部分微调任务
- 建议：优先使用 LoRA

**全量微调**：
- 优点：表达能力更强
- 缺点：训练慢、显存占用大
- 适用场景：需要大幅改变模型行为的任务

### Q2: 训练时显存不足怎么办？

1. 减小 `per_device_train_batch_size`
2. 增大 `gradient_accumulation_steps`（保持有效 batch size 不变）
3. 启用梯度检查点：`gradient_checkpointing: true`
4. 使用 LoRA 而非全量微调
5. 使用混合精度训练：`fp16: true` 或 `bf16: true`

### Q3: 如何判断训练是否收敛？

观察以下指标：
- Training Loss 持续下降并趋于平稳
- Validation Loss 不再下降或开始上升（早停信号）
- Perplexity 降低到合理范围
- 生成样本质量符合预期

### Q4: 如何使用扩展词表后的模型？

在 YAML 配置中指定扩展后的路径：
```yaml
model_name_or_path: tokenizer/merged_model
tokenizer_name_or_path: tokenizer/merged_tokenizer
```

### Q5: 多 GPU 训练如何配置？

使用 `accelerate` 启动：
```bash
accelerate launch --num_processes 4 \
  self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml
```

或使用 `torchrun`：
```bash
torchrun --nproc_per_node 4 \
  self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml
```
