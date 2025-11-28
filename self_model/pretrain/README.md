# 继续预训练（CPT）

使用 Transformers 对基座模型进行继续预训练（Causal LM）。

## 快速开始

```bash
# 单卡
uv run python self_model/pretrain/train_pretrain.py \
  --config config/self_model_config/pretrain.yaml

# torchrun 多卡
torchrun --nproc_per_node=4 \
  self_model/pretrain/train_pretrain.py \
  --config config/self_model_config/pretrain.yaml

# DeepSpeed
deepspeed --num_gpus=4 \
  self_model/pretrain/train_pretrain.py \
  --config config/self_model_config/pretrain.yaml \
  --deepspeed config/self_model_config/deepspeed/ds_config.json
```

## 目录结构

```
pretrain/
├── train_pretrain.py  # CPT 训练入口
└── README.md          # 本文档
```

## 配置文件

- 基础模板：`config/self_model_config/pretrain.yaml`
- 可选追踪：`config/trace_config/wandb.yaml` / `config/trace_config/swanlab.yaml`

核心字段：
- `model_name_or_path` / `tokenizer_name_or_path`: 基座模型与分词器
- `data.train_file` / `data.eval_file`: JSONL 或 TXT 文本数据
- `data.text_fields`: 优先读取的字段列表（默认 `text`），缺失时尝试 messages/conversations 拼接
- `data.block_size`: 文本拼接后分块长度
- `training_args`: Transformers `TrainingArguments`
- `peft`: 可选 LoRA/QLoRA（默认关闭，显存受限时可开启）

## 数据格式

推荐 JSONL，至少包含 `text` 字段：
```json
{"text": "这是继续预训练的原始文本。"}
```

其他支持：
- 文本文件：`.txt`（一行一条样本）
- 对话/ShareGPT：包含 `messages` 或 `conversations` 时会拼接为纯文本

> 提示：在 `data_process/` 里清洗/去重预训练数据，输出到 `data_process/final_data/pretrain/` 后直接使用。

## 训练技巧

- 选择合适的 `block_size`（常用 512/1024/2048），避免超过显存上限
- 若显存不足可开启 `peft.enabled` + `peft.qlora` 进行参数高效继续预训练
- 大规模数据建议开启 DeepSpeed ZeRO-2/3 或梯度检查点
- 追踪平台可通过 `--trace wandb` 或 `--trace swanlab` 临时覆盖 YAML 配置
