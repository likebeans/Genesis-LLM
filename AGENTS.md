# AGENTS.md

## Project Overview

**self-model-sop** is a comprehensive LLM training pipeline for Chinese/domain-specific model development. It covers the complete workflow from tokenizer expansion, data collection/processing, to supervised fine-tuning (SFT) and RLHF alignment using PEFT (LoRA/QLoRA) with Transformers.

**Tech Stack**: Python 3.10+, PyTorch 2.1+, Transformers, PEFT, SentencePiece, Datasets, Accelerate, WandB/SwanLab

**Key Features**:
- Tokenizer vocabulary expansion for Chinese/domain terms
- HuggingFace dataset fetching and processing pipeline
- Config-driven training (YAML-based)
- PEFT support (LoRA/QLoRA) for efficient fine-tuning
- Integrated experiment tracking (WandB/SwanLab)

## Setup Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Quick Start

**1. Data Collection**
```bash
# Fetch dataset from HuggingFace
uv run python data_process/fetch_dataset.py \
  --dataset_name shibing624/medical \
  --data_subdir pretrain \
  --output_dir data_process/fetch_data \
  --sample_limit 1000
```

**2. Data Processing**
```bash
# Clean and process data
uv run python data_process/process_data.py \
  --input data_process/fetch_data/pretrain/shibing624_medical__train.jsonl \
  --output_dir data_process/final_data \
  --subdir pretrain \
  --dedup --near_dedup \
  --to_parquet
```

**3. Tokenizer Expansion** (Optional)
```bash
# Train auxiliary tokenizer
uv run python tokenizer/aux_tokenizer/train_aux_tokenizer.py \
  --config config/tokenizer_config/config.yaml

# Extend base tokenizer
uv run python tokenizer/aux_tokenizer/extend_base_tokenizer.py \
  --config config/tokenizer_config/config.yaml
```

**4. Fine-tuning**
```bash
# Run supervised fine-tuning with LoRA
uv run python self_model/fine_tuning/train_finetune.py \
  --config config/self_model_config/finetune.yaml \
  --trace swanlab
```

## Project Structure

```
self_model_sop/
├── config/                      # All configuration files
│   ├── model_config/           # HF mirror and model configs
│   ├── self_model_config/      # Training configs (pretrain/finetune/rlhf)
│   ├── tokenizer_config/       # Tokenizer expansion configs
│   └── trace_config/           # Experiment tracking (wandb/swanlab)
├── data_process/               # Data pipeline
│   ├── fetch_dataset.py       # HF dataset fetcher
│   ├── process_data.py        # Data cleaning and processing
│   ├── fetch_data/            # Raw downloaded data
│   └── final_data/            # Processed data (JSONL/Parquet)
├── tokenizer/                  # Tokenizer expansion workflow
│   ├── aux_tokenizer/         # Auxiliary tokenizer training scripts
│   ├── data/                  # Tokenizer training corpus
│   ├── evaluate/              # Tokenizer evaluation scripts
│   ├── merged_tokenizer/      # Extended tokenizer output
│   └── merged_model/          # Model with extended embeddings
├── self_model/                 # Training scripts
│   ├── fine_tuning/           # SFT training
│   ├── pretrain/              # CPT (Continued PreTraining)
│   ├── rlhf/                  # RLHF (DPO/PPO/GRPO via VERL)
│   ├── template/              # Chat templates (ChatML, etc.)
│   └── checkpoints/           # Training checkpoints
├── inference/                  # Model inference and deployment
│   ├── vllm/                  # vLLM Docker deployment scripts
│   ├── llama.cpp/             # GGUF conversion for edge deployment
│   └── quantize/              # AWQ/GPTQ quantization
├── docker/                     # Docker configuration
│   ├── Dockerfile             # Dev/training environment
│   └── docker-compose.yml     # Multi-service orchestration
└── docs/                       # Documentation
```

## Code Style & Conventions

### Python Style
- Python 3.10+ with type hints
- Use `from __future__ import annotations` for forward references
- Follow PEP 8 conventions
- Prefer dataclasses for configuration objects
- Use pathlib.Path for file operations

### Configuration Management
- **All training parameters in YAML files** under `config/`
- Config hierarchy: `config/self_model_config/*.yaml` for training, `config/trace_config/*.yaml` for logging
- Scripts read YAML first, allow CLI overrides
- Use `yaml.safe_load()` for parsing

### Data Processing
- **Fetch stage**: Keep original fields intact ("搬运工" principle)
- **Process stage**: Clean, deduplicate, filter quality
- Output format: JSONL (one JSON object per line)
- Optional Parquet output with `--to_parquet`

### Training Scripts
- Use `argparse` with `--config` pointing to YAML
- Support `--trace` flag for experiment tracking backend
- Load tokenizer/model from paths in YAML
- Apply PEFT (LoRA) when `peft.enabled: true` in config
- Save checkpoints to `output_dir` specified in YAML

### Tokenizer Workflow
- **Baseline extraction**: Record base tokenizer config
- **Auxiliary training**: Train new tokens on domain corpus
- **Token selection**: Compare tokenization, select high-frequency candidates
- **Merge**: Append new tokens to base tokenizer (no deletion/reordering)
- **Embedding extension**: Use `resize_token_embeddings()` for model
- **Evaluation**: Compare OOV rate, token length, perplexity

## Key Configuration Files

### Training Config (`config/self_model_config/finetune.yaml`)
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

### Tokenizer Config (`config/tokenizer_config/config.yaml`)
```yaml
base_tokenizer: bert-base-chinese
aux_tokenizer:
  vocab_size: 20000
  model_type: bpe
  character_coverage: 0.9995
  split_digits: true
  byte_fallback: true
```

## Testing & Validation

### Data Pipeline Tests
```bash
# Verify data fetching (shows first 5 rows)
uv run python data_process/fetch_dataset.py --dataset_name <name> --sample_limit 10

# Check processing stats in output logs
# Look for: total/kept/duplicates/too_short/too_long counts
```

### Tokenizer Evaluation
```bash
# Evaluate tokenizer coverage and efficiency
uv run python tokenizer/evaluate/eval_tokenizer.py \
  --baseline_tokenizer <path> \
  --extended_tokenizer <path> \
  --test_corpus <path>

# Evaluate MLM loss (if using MLM pretraining)
uv run python tokenizer/evaluate/eval_mlm_loss.py \
  --model_path <path> \
  --test_data <path>
```

### Training Validation
- Monitor loss/perplexity curves in WandB/SwanLab
- Check checkpoint saves at `output_dir`
- Verify tokenizer saved alongside model
- Test inference with saved checkpoint

## Dependencies

Core dependencies (see `pyproject.toml`):
- `torch>=2.1.0` - PyTorch framework
- `transformers>=4.57.3` - HuggingFace Transformers
- `peft>=0.18.0` - Parameter-Efficient Fine-Tuning
- `datasets>=4.4.1` - HuggingFace Datasets
- `sentencepiece>=0.2.1` - Tokenizer training
- `accelerate>=1.12.0` - Distributed training
- `wandb>=0.23.0` - Experiment tracking
- `swanlab>=0.7.2` - Alternative tracking
- `pyyaml>=6.0.3` - Config parsing

## Important Notes

### For AI Agents Working on This Project

**Configuration-Driven Design**:
- Never hardcode paths or hyperparameters in scripts
- Always read from YAML configs in `config/` directory
- Respect the config hierarchy: model_config → self_model_config → trace_config

**Data Pipeline Philosophy**:
- **Fetch stage**: Preserve all original fields, no modifications
- **Process stage**: Apply cleaning, deduplication, quality filters
- Keep fetch and process stages separate for flexibility

**Tokenizer Expansion**:
- Only append new tokens, never delete or reorder existing ones
- Special tokens (`<|system|>`, `
