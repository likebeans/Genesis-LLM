# Configuration Directory (config/) - AGENTS.md

## 目录概览

`config/` 目录是整个项目的配置中心，采用 YAML 格式管理所有训练、数据处理、词表扩充和实验追踪的参数。所有脚本都从这里读取配置，避免硬编码。

## 目录结构

```
config/
├── model_config/           # 模型相关的全局配置
│   └── config.yaml        # HuggingFace 镜像地址配置
├── self_model_config/      # 训练流程配置
│   ├── pretrain.yaml      # 继续预训练（CPT）配置
│   ├── finetune.yaml      # 监督微调（SFT）配置
│   └── rlhf.yaml          # 强化学习对齐（RLHF）配置
├── tokenizer_config/       # 词表扩充配置
│   └── config.yaml        # 词表训练、合并、评估的完整配置
└── trace_config/           # 实验追踪配置
    ├── wandb.yaml         # Weights & Biases 配置
    └── swanlab.yaml       # SwanLab 配置
```

## 配置文件详解

### 1. model_config/config.yaml

**用途**：全局模型配置，主要用于配置 HuggingFace 镜像加速下载。

**使用场景**：
- 数据采集脚本 `fetch_dataset.py` 读取此配置设置 HF_ENDPOINT
- 在国内网络环境下加速模型和数据集下载

**关键字段**：
- `huggingface.endpoint`: HuggingFace 镜像地址（如 https://hf-mirror.com）

**示例用法**：
```bash
# fetch_dataset.py 会自动读取此配置
uv run python data_process/fetch_dataset.py --config config/model_config/config.yaml
```

---

### 2. self_model_config/finetune.yaml

**用途**：监督微调（SFT）的完整配置，支持 PEFT（LoRA/QLoRA）。

**使用场景**：
- 在已有基座模型或 CPT 检查点上进行指令微调
- 使用 LoRA 等参数高效方法减少显存占用
- 配合 WandB/SwanLab 进行实验追踪

**关键配置项**：
- **模型路径**：`model_name_or_path`, `tokenizer_name_or_path`
- **数据配置**：`data.train_file`, `data.eval_file`, `data.max_length`
- **PEFT 配置**：`peft.enabled`, `peft.lora_r`, `peft.target_modules`
- **训练参数**：`training_args.*`（batch size, learning rate, epochs 等）
- **实验追踪**：`trace`, `trace_project`

**调用方式**：
```bash
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --trace swanlab
```

**配置层级**：
1. 从 YAML 读取所有参数
2. 命令行参数可覆盖（如 `--trace`）
3. 自动加载对应的 trace_config（wandb.yaml 或 swanlab.yaml）

---

### 3. self_model_config/pretrain.yaml

**用途**：继续预训练（Continued PreTraining, CPT）配置。

**使用场景**：
- 在基座模型上使用领域语料继续预训练
- 适应特定领域（医疗、法律、金融等）的语言分布
- 通常在 SFT 之前进行

**核心配置项**：
- 模型/Tokenizer 路径（可使用扩展词表后的模型）
- 数据：`train_file` / `eval_file`，`text_fields`（优先读取字段列表），`block_size`（切块长度）
- PEFT：`peft.enabled` / `peft.qlora`（默认关闭，低显存可选）
- 实验追踪：`trace`, `trace_project`
- 训练超参：batch size、learning rate、warmup、save_steps、lr_scheduler_type
- 分布式：可选 `deepspeed` 配置

---

### 4. self_model_config/rlhf.yaml

**用途**：强化学习人类反馈（RLHF）对齐配置，支持 DPO/PPO/GRPO。

**使用场景**：
- 在 SFT 模型基础上进行偏好对齐
- 使用偏好数据（chosen/rejected pairs）训练
- 提升模型的有用性、无害性、诚实性

**当前配置项**：
- 模型：`model_name_or_path`（策略）、`ref_model_name_or_path`（参考，可与策略相同）、`reward_model_name_or_path`（PPO/GRPO）
- DPO：`data.train_file` / `eval_file`（prompt/chosen/rejected）、`beta`、`loss_type`（sigmoid/ipo/pai）、`training_args.*`
- PPO/GRPO（`ppo.*` 节，VERL）：`train_file`/`eval_file`、`prompt_key`、`max_prompt_length`/`max_response_length`、`rollout_backend`（vllm/hf）、`rollout_n`、`actor_lr`、`ppo_mini_batch_size`、`ppo_micro_batch_size_per_gpu`、`kl_loss_coef`/`use_kl_loss`、`logger`、`n_gpus_per_node`/`nnodes`
- PEFT：`peft.enabled` / `peft.qlora` 低显存可选
- 追踪：`trace`（wandb/swanlab/none，VERL 支持 console/wandb）

**框架**：DPO 基于 TRL `DPOTrainer`，PPO/GRPO 基于 VERL `main_ppo`。

---

### 5. tokenizer_config/config.yaml

**用途**：词表扩充的完整工作流配置，涵盖数据准备、辅助词表训练、合并、评估。

**使用场景**：
- 为基座模型（如 LLaMA2, Qwen）扩充中文/领域词表
- 减少专业术语被拆分，降低 OOV 率
- 保持与基座模型的兼容性

**配置结构**：
1. **基座配置**：`base_tokenizer`, `base_model`
2. **数据配置**：`data.*`（数据集名称、子目录、采样限制）
3. **基线配置**：`baseline.config_path`（基座 tokenizer 配置快照）
4. **辅助词表**：`aux_tokenizer.*`（vocab_size, model_type, character_coverage 等）
5. **Token 筛选**：`selection.*`（频率阈值、最大拆分数）
6. **数据处理**：`process.*`（清洗、去重参数）
7. **输出路径**：`extended_tokenizer`, `extended_model`

**工作流**：
```bash
# 1. 提取基座配置
uv run python tokenizer/aux_tokenizer/extract_tokenizer_config.py --config config/tokenizer_config/config.yaml

# 2. 准备训练语料
uv run python tokenizer/data/fetch_and_clean.py --config config/tokenizer_config/config.yaml

# 3. 训练辅助 tokenizer
uv run python tokenizer/aux_tokenizer/train_aux_tokenizer.py --config config/tokenizer_config/config.yaml

# 4. 筛选新增 token
uv run python tokenizer/aux_tokenizer/select_new_tokens.py --config config/tokenizer_config/config.yaml

# 5. 合并到基座
uv run python tokenizer/aux_tokenizer/extend_base_tokenizer.py --config config/tokenizer_config/config.yaml

# 6. 扩展模型 embedding
uv run python tokenizer/aux_tokenizer/resize_embeddings.py --config config/tokenizer_config/config.yaml

# 7. 评估效果
uv run python tokenizer/evaluate/eval_tokenizer.py --config config/tokenizer_config/config.yaml
```

**关键原则**：
- 只追加新 token，不删除或重排基座 token
- 特殊 token 顺序必须一致
- 新增 token 需要通过 CPT/SFT 获得语义

---

### 6. trace_config/wandb.yaml

**用途**：Weights & Biases 实验追踪配置。

**使用场景**：
- 训练脚本通过 `--trace wandb` 启用
- 自动记录 loss、learning rate、梯度等指标
- 支持团队协作和实验对比

**关键字段**：
- `project`: W&B 项目名
- `entity`: W&B 团队/用户名
- `api_key`: API 密钥（从 https://wandb.ai/settings 获取）
- `mode`: `online`（实时上传）/ `offline`（本地保存）/ `disabled`（禁用）
- `run_name`: 实验运行名称
- `notes`: 实验描述
- `tags`: 实验标签（用于筛选和分组）

**集成方式**：
- 训练脚本读取此配置，设置环境变量（`WANDB_PROJECT`, `WANDB_API_KEY` 等）
- Transformers Trainer 通过 `report_to=["wandb"]` 自动集成
- 支持自定义 callback 扩展

**安全提示**：
- `api_key` 是敏感信息，建议使用环境变量或 `.env` 文件
- 不要将包含真实 API Key 的配置提交到公开仓库

---

### 7. trace_config/swanlab.yaml

**用途**：SwanLab 实验追踪配置（国内替代方案）。

**使用场景**：
- 与 WandB 类似，但服务器在国内，访问更快
- 支持 Transformers 4.50.0+ 原生集成
- 适合国内团队使用

**关键字段**：
- `project`: SwanLab 项目名
- `workspace`: SwanLab 工作空间
- `api_key`: API 密钥
- `api_host`: API 地址（自托管可修改）
- `web_host`: Web 界面地址
- `mode`: `cloud`（云端）/ `offline`（离线）/ `local`（本地）
- `run_name`: 实验运行名称
- `description`: 实验描述

**集成方式**：
- Transformers >= 4.50.0：通过 `report_to=["swanlab"]` 原生支持
- Transformers < 4.50.0：使用 `SwanLabCallback` 手动集成
- 训练脚本自动检测版本并选择合适的集成方式

**优势**：
- 国内访问速度快
- 中文界面友好
- 支持私有化部署

---

## 配置使用规范

### 1. 配置优先级

```
命令行参数 > YAML 配置 > 脚本默认值
```

**示例**：
```bash
# YAML 中设置 trace: swanlab
# 命令行覆盖为 wandb
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --trace wandb  # 覆盖 YAML 中的设置
```

### 2. 路径规范

**相对路径**：所有配置中的路径都相对于项目根目录（`/home/admin1/work/self_model_sop/`）

**示例**：
```yaml
# ✅ 正确
train_file: data_process/final_data/finetune/train.jsonl

# ❌ 错误（绝对路径）
train_file: /home/admin1/work/self_model_sop/data_process/final_data/finetune/train.jsonl
```

### 3. 配置复用

**模板化**：可以创建多个配置文件用于不同实验

```bash
config/self_model_config/
├── finetune.yaml              # 基础配置
├── finetune_medical.yaml      # 医疗领域
├── finetune_legal.yaml        # 法律领域
└── finetune_qlora.yaml        # QLoRA 4bit 量化
```

### 4. 敏感信息处理

**不要在配置文件中硬编码敏感信息**：
- API Keys
- 访问令牌
- 密码

**推荐方式**：
```yaml
# 方式 1：使用占位符，从环境变量读取
api_key: ${WANDB_API_KEY}

# 方式 2：在脚本中从环境变量读取
api_key: ""  # 留空，脚本从 os.environ 读取
```

### 5. 版本控制

**建议**：
- 将配置文件纳入 Git 版本控制
- 敏感信息使用 `.env` 文件（加入 `.gitignore`）
- 每次实验记录使用的配置版本（commit hash）

---

## 配置验证

### 检查配置完整性

```python
import yaml
from pathlib import Path

def validate_config(config_path: str):
    """验证配置文件格式和必需字段"""
    cfg = yaml.safe_load(Path(config_path).read_text())
    
    # 检查必需字段
    required_fields = ["model_name_or_path", "data", "training_args"]
    for field in required_fields:
        assert field in cfg, f"Missing required field: {field}"
    
    # 检查路径存在性
    if "train_file" in cfg.get("data", {}):
        train_file = Path(cfg["data"]["train_file"])
        assert train_file.exists(), f"Train file not found: {train_file}"
    
    print(f"✅ Config validation passed: {config_path}")

# 使用
validate_config("config/self_model_config/finetune.yaml")
```

---

## 常见问题

### Q1: 如何在多 GPU 上训练？

在 `training_args` 中添加：
```yaml
training_args:
  # ... 其他参数
  ddp_find_unused_parameters: false
  fsdp: "full_shard auto_wrap"  # 使用 FSDP
  fsdp_config:
    fsdp_offload_params: true
    fsdp_state_dict_type: "FULL_STATE_DICT"
```

然后使用 `torchrun` 或 `accelerate launch` 启动。

### Q2: 如何切换实验追踪后端？

**方式 1**：修改 YAML
```yaml
trace: wandb  # 或 swanlab / none
```

**方式 2**：命令行覆盖
```bash
--trace wandb
```

### Q3: 如何禁用实验追踪？

```bash
--trace none
```

或在 YAML 中：
```yaml
trace: none
training_args:
  report_to: []
```

### Q4: 配置文件太大怎么办？

使用 YAML 的锚点和引用功能：
```yaml
# 定义公共配置
common_training: &common_training
  per_device_train_batch_size: 2
  learning_rate: 5.0e-5
  fp16: true

# 复用
training_args:
  <<: *common_training
  num_train_epochs: 3
```

### Q5: 如何管理多个实验配置？

使用配置继承或模板工具（如 Hydra）：
```bash
# 基础配置
config/self_model_config/base.yaml

# 实验配置（继承 base）
config/self_model_config/experiments/
├── exp001_baseline.yaml
├── exp002_lora_r16.yaml
└── exp003_qlora.yaml
```

---

## 最佳实践

### 1. 配置即文档

在配置文件中添加详细注释，说明每个参数的作用和取值范围。

### 2. 小步迭代

先用小数据集、小模型验证配置正确性，再扩展到完整训练。

### 3. 记录实验

每次训练记录：
- 配置文件版本（Git commit）
- 数据集版本（哈希值）
- 训练时间和资源
- 最终指标

### 4. 自动化验证

在 CI/CD 中添加配置验证步骤，确保配置文件格式正确。

### 5. 文档同步

配置更新时同步更新此 AGENTS.md 文档。

---

## 相关文档

- [项目根目录 AGENTS.md](../AGENTS.md) - 项目整体说明
- [词表开发规则](../docs/词表开发规则.md) - 词表扩充详细流程
- [数据处理 README](../data_process/README.md) - 数据采集和处理说明
- [训练 README](../self_model/README.md) - 训练流程说明

---

## 更新日志

- **2025-11-27**: 创建配置目录 AGENTS.md，详细说明各配置文件用途
- **2025-11-27**: 补充 rlhf.yaml DPO 模板说明
