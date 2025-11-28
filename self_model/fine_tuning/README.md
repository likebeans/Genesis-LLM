# 监督微调训练 (SFT)

使用 Transformers + PEFT (LoRA/QLoRA) 进行监督微调。

## 快速开始

```bash
# 基础训练
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml

# 自定义配置 + 实验追踪
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune_medical.yaml \
  --trace swanlab
```

## 目录结构

```
fine_tuning/
├── train_finetune.py    # 训练脚本
└── README.md            # 本文档
```

## 配置文件

位于 `config/self_model_config/`：

| 文件 | 说明 |
|------|------|
| `finetune.yaml` | 基础模板，简洁配置 |
| `finetune_medical.yaml` | 医疗领域示例 |
| `finetune_reference.yaml` | 详细参数说明 |
| `deepspeed/` | 分布式训练配置 |

## 对话模板选择

- 训练时会优先使用 tokenizer 自带的 `apply_chat_template`，若模型未内置则根据 `template_path` 加载并将 JSON 模板格式化成字符串。
- 常用模板路径（`self_model/template/`）：`qwen2_chatml.json`（Qwen 系列）、`llama3_instruct.json`（Llama3 Instruct）、`chatglm3.json`（ChatGLM3）、`baichuan2_chat.json`（Baichuan2-Chat）、`deepseek_chat.json`（DeepSeek-LLM-Chat）。
- 切换模型时同步修改 YAML 配置中的 `template_path`，必要时自定义新模板放在该目录并指向对应路径。

## 创建自定义配置

```bash
# 1. 复制模板
cp config/self_model_config/finetune.yaml \
   config/self_model_config/finetune_my_task.yaml

# 2. 编辑配置（修改模型、数据路径等）

# 3. 运行训练
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune_my_task.yaml
```

## 分布式训练

```bash
# torchrun 多卡
torchrun --nproc_per_node=4 \
  self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml

# DeepSpeed ZeRO-2
deepspeed --num_gpus=4 \
  self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --deepspeed config/self_model_config/deepspeed/ds_config.json

# DeepSpeed ZeRO-3（大模型）
deepspeed --num_gpus=4 \
  self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --deepspeed config/self_model_config/deepspeed/ds_config_zero3.json
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `config/self_model_config/finetune.yaml` |
| `--trace` | 实验追踪 (wandb/swanlab/none) | YAML 配置 |
| `--deepspeed` | DeepSpeed 配置路径 | 无 |
| `--local_rank` | 分布式 rank（自动） | -1 |

## 数据格式

支持多种 JSONL 格式：

**1. 多轮对话（推荐）**
```json
{"messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}
```

**2. ShareGPT 格式**
```json
{"conversations": [{"from": "human", "value": "问题"}, {"from": "gpt", "value": "回答"}]}
```

**3. 单轮问答**
```json
{"input": "用户问题", "output": "模型回答"}
```

**4. 指令格式**
```json
{"instruction": "任务说明", "input": "输入内容", "output": "输出内容"}
```

**5. Alpaca 格式**
```json
{"instruction": "...", "input": "...", "response": "..."}
```

> Labels 处理：自动对 prompt 部分 mask（设为 -100），只计算 response 损失

## 输出

- 检查点：`training_args.output_dir` 目录
- 实验日志：WandB/SwanLab 平台（启用时）

## 参考

- 详细参数说明：`config/self_model_config/finetune_reference.yaml`
- 对话模板：`self_model/template/`
- DeepSpeed 配置：`config/self_model_config/deepspeed/`
