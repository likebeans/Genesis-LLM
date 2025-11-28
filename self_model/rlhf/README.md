# RLHF 对齐

当前实现：
- DPO（TRL）
- PPO / GRPO（VERL）

> 提示：PPO/GRPO 依赖 `verl`（Ray + vLLM/HF rollout）。安装示例：`pip install "verl>=0.3.0" ray omegaconf`

## 快速开始

```bash
# PPO（VERL）
uv run python self_model/rlhf/train_ppo.py \
  --config config/self_model_config/rlhf.yaml

# GRPO（VERL）
uv run python self_model/rlhf/train_grpo.py \
  --config config/self_model_config/rlhf.yaml

# DPO（TRL）
uv run python self_model/rlhf/train_dpo.py \
  --config config/self_model_config/rlhf.yaml

# torchrun 多卡（DPO 示例）
torchrun --nproc_per_node=4 \
  self_model/rlhf/train_dpo.py \
  --config config/self_model_config/rlhf.yaml

# DeepSpeed（DPO 示例）
deepspeed --num_gpus=4 \
  self_model/rlhf/train_dpo.py \
  --config config/self_model_config/rlhf.yaml \
  --deepspeed config/self_model_config/deepspeed/ds_config.json
```

## 数据格式

### DPO

JSONL，每行包含 `prompt`、`chosen`、`rejected`：
```json
{"prompt": "写一首关于秋天的诗", "chosen": "落叶金黄...", "rejected": "我不知道"}
```

兼容字段映射：
- `prompt` 亦可使用 `question`/`query`
- `chosen` 亦可使用 `response_chosen`/`answer`
- `rejected` 亦可使用 `response_rejected`/`negative`

### PPO / GRPO

- Parquet（推荐）或 JSONL，包含 `prompt` 字段（默认 `prompt_key=prompt`，可在 `ppo.prompt_key` 覆盖）
- 建议先将 JSONL 转 parquet 提速

## 配置文件

- 模板：`config/self_model_config/rlhf.yaml`
- 关键字段：
  - `model_name_or_path` / `ref_model_name_or_path`: 策略模型与参考模型（可相同）
  - `tokenizer_name_or_path`: Tokenizer 路径
  - `data.train_file` / `data.eval_file`: DPO 数据路径
  - `data.max_length`: 最大长度（会用于 prompt/response 拼接）
  - `beta` / `loss_type`: DPO 超参（sigmoid/ipo/pai）
  - `peft`: 可选 LoRA/QLoRA，显存紧张时启用
  - `training_args`: Transformers `TrainingArguments`
  - `trace`: wandb/swanlab/none（PPO/GRPO 支持 wandb / console）
- PPO/GRPO（VERL）字段（`ppo.*` 节）：
  - 数据：`train_file`/`eval_file`、`prompt_key`、`max_prompt_length`/`max_response_length`
  - rollout：`rollout_backend`（vllm/hf/sglang）、`rollout_n`、`rollout_tp`、`rollout_gpu_memory_utilization`、`load_format`
  - 超参：`actor_lr`、`ppo_epochs`、`ppo_mini_batch_size`、`ppo_micro_batch_size_per_gpu`、`kl_loss_coef`、`kl_loss_type`、`use_kl_loss`（GRPO 默认 true）、`use_kl_in_reward`
  - 奖励模型：`reward_model_name_or_path`、`reward_micro_batch_size_per_gpu`、`reward_max_length`
  - 资源/日志：`n_gpus_per_node`、`nnodes`、`logger`、`project_name`、`experiment_name`、`save_freq`、`test_freq`

## 依赖

- `trl`（DPOTrainer）
- `transformers`, `datasets`, `peft`（可选，用于 LoRA/QLoRA）
- `verl` + `ray`（PPO/GRPO），rollout 默认 vLLM；如需 HF rollout 可设置 `ppo.rollout_backend=hf` 且 `ppo.load_format=hf`

## TODO

- 奖励模型训练与推理示例
