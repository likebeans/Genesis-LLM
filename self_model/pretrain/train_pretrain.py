#!/usr/bin/env python3
"""
继续预训练（CPT）脚本
==============================

功能：
    使用 Transformers 对基座模型进行继续预训练（Causal LM）。
    支持配置驱动，从 YAML 文件读取训练参数。
    若模型未内置 chat template，则按纯文本块训练（不依赖对话模板）。

使用方式：
    uv run python self_model/pretrain/train_pretrain.py \
        --config config/self_model_config/pretrain.yaml

    # 多卡训练（torchrun）
    torchrun --nproc_per_node=4 self_model/pretrain/train_pretrain.py --config xxx.yaml

    # DeepSpeed 训练
    deepspeed --num_gpus=4 self_model/pretrain/train_pretrain.py \
        --config xxx.yaml --deepspeed config/self_model_config/deepspeed/ds_config.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import torch
import transformers
import yaml
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import WandbCallback

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


# =============================================================================
# 配置解析
# =============================================================================


@dataclass
class PretrainConfig:
    """命令行参数配置类"""

    config: str = "config/self_model_config/pretrain.yaml"
    trace: str = ""  # wandb / swanlab / none / empty
    deepspeed: str = ""  # DeepSpeed 配置文件路径
    local_rank: int = -1  # 分布式训练的 local_rank（torchrun 自动传入）


def load_yaml(path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[错误] 配置文件不存在: {path}")
    return yaml.safe_load(p.read_text())


def load_optional_yaml(path: str) -> Dict[str, Any]:
    """加载可选的 YAML 配置文件（不存在时返回空字典）"""
    p = Path(path)
    if not p.exists():
        return {}
    content = yaml.safe_load(p.read_text())
    return content or {}


def parse_args() -> PretrainConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="继续预训练（CPT）脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run python self_model/pretrain/train_pretrain.py --config config/self_model_config/pretrain.yaml
        """
    )
    parser.add_argument("--config", default="config/self_model_config/pretrain.yaml", help="Path to pretrain YAML.")
    parser.add_argument(
        "--trace",
        choices=["wandb", "swanlab", "none"],
        default=None,
        help="Override trace backend; leave empty to use YAML value.",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed 配置文件路径（如 config/ds_config.json）",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="分布式训练的 local_rank（torchrun 自动传入，无需手动指定）",
    )
    args = parser.parse_args()
    return PretrainConfig(
        config=args.config,
        trace=args.trace or "",
        deepspeed=args.deepspeed or "",
        local_rank=args.local_rank,
    )


# =============================================================================
# 数据处理
# =============================================================================


def build_dataset(
    train_file: str,
    eval_file: Optional[str],
    tokenizer,
    block_size: int,
    text_fields: List[str],
) -> Dict[str, datasets.Dataset]:
    """
    构建继续预训练数据集：把文本拆分成固定长度块，生成 input_ids/labels。

    支持的数据来源：
        - JSONL/JSON，包含 text 字段（推荐）
        - JSONL，包含其他字段（通过 text_fields 指定优先级）
        - JSONL，含 messages/conversations 字段时会拼接成纯文本
        - 纯文本 (.txt)
    """
    def load_file(fp: str):
        """加载数据文件"""
        p = Path(fp)
        if not p.exists():
            raise SystemExit(f"[错误] 数据文件不存在: {fp}")
        suffix = p.suffix.lower()
        if suffix in [".json", ".jsonl"]:
            return datasets.load_dataset("json", data_files=fp)["train"]
        if suffix in [".txt"]:
            return datasets.load_dataset("text", data_files=fp)["train"]
        return datasets.load_dataset("json", data_files=fp)["train"]

    def extract_text(example):
        """从多种字段中提取文本，fallback 为 messages/conversations"""
        text = ""
        for field in text_fields:
            if field in example and example[field]:
                val = example[field]
                if isinstance(val, list):
                    val = " ".join(str(v) for v in val if v is not None)
                else:
                    val = str(val)
                if val.strip():
                    text = val.strip()
                    break

        if not text and "messages" in example:
            parts = []
            for m in example["messages"]:
                role = m.get("role", "")
                content = m.get("content", "")
                seg = f"{role}: {content}".strip()
                if seg:
                    parts.append(seg)
            text = "\n".join(parts)

        if not text and "conversations" in example:
            role_map = {"human": "User", "gpt": "Assistant"}
            parts = []
            for conv in example["conversations"]:
                role = role_map.get(conv.get("from", ""), conv.get("from", ""))
                content = conv.get("value", "")
                seg = f"{role}: {content}".strip()
                if seg:
                    parts.append(seg)
            text = "\n".join(parts)

        return {"text": text}

    def tokenize_and_group(dataset: datasets.Dataset) -> datasets.Dataset:
        """分词并按 block_size 拼接/切块"""
        dataset = dataset.map(extract_text)
        dataset = dataset.filter(lambda x: x.get("text", "") not in [None, ""], desc="filter empty text")

        # 只保留 text，避免额外列占用显存/磁盘
        extra_cols = [c for c in dataset.column_names if c != "text"]
        if extra_cols:
            dataset = dataset.remove_columns(extra_cols)

        def tokenize_fn(batch):
            return tokenizer(batch["text"])

        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="tokenize",
        )

        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = [list(ids) for ids in result["input_ids"]]
            return result

        return tokenized.map(
            group_texts,
            batched=True,
            desc=f"group texts (block_size={block_size})",
        )

    train_ds = tokenize_and_group(load_file(train_file))
    eval_ds = tokenize_and_group(load_file(eval_file)) if eval_file else None
    return {"train": train_ds, "eval": eval_ds}


# =============================================================================
# 实验追踪
# =============================================================================


def init_trace(trace_type: str):
    """初始化实验追踪后端 (wandb 或 swanlab)"""
    if trace_type.lower() == "wandb":
        try:
            import wandb  # type: ignore
            return "wandb"
        except Exception:
            print("[warn] wandb not installed, fallback to no logging.")
    elif trace_type.lower() == "swanlab":
        try:
            import swanlab  # type: ignore
            return "swanlab"
        except Exception:
            print("[warn] swanlab not installed, fallback to no logging.")
    return None


# =============================================================================
# 主函数
# =============================================================================


def main():
    """训练主函数"""
    cfg_args = parse_args()
    yaml_cfg = load_yaml(cfg_args.config)

    # Model/tokenizer paths
    model_name = yaml_cfg.get("model_name_or_path", "tokenizer/merged_model_mlm")
    tokenizer_name = yaml_cfg.get("tokenizer_name_or_path", model_name)

    # Training args
    training_args_dict = yaml_cfg.get("training_args", {})
    # 兼容旧字段：evaluation_strategy -> eval_strategy
    if "evaluation_strategy" in training_args_dict and "eval_strategy" not in training_args_dict:
        training_args_dict["eval_strategy"] = training_args_dict.pop("evaluation_strategy")

    # 分布式配置
    deepspeed_config = cfg_args.deepspeed or yaml_cfg.get("deepspeed")
    if deepspeed_config:
        training_args_dict["deepspeed"] = deepspeed_config

    local_rank = cfg_args.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        training_args_dict["local_rank"] = local_rank

    is_distributed = (
        local_rank != -1
        or os.environ.get("WORLD_SIZE") is not None
        or deepspeed_config is not None
    )

    training_args = TrainingArguments(**training_args_dict)

    # Data 配置
    train_file = yaml_cfg["data"]["train_file"]
    eval_file = yaml_cfg["data"].get("eval_file")
    block_size = int(yaml_cfg["data"].get("block_size", 1024))
    text_fields = yaml_cfg["data"].get("text_fields", ["text"])
    if isinstance(text_fields, str):
        text_fields = [text_fields]

    # PEFT 配置（可选）
    peft_cfg = yaml_cfg.get("peft", {})
    use_lora = peft_cfg.get("enabled", False)
    lora_r = peft_cfg.get("lora_r", 8)
    lora_alpha = peft_cfg.get("lora_alpha", 16)
    lora_dropout = peft_cfg.get("lora_dropout", 0.05)
    target_modules = peft_cfg.get("target_modules", ["query", "key", "value"])
    use_4bit = peft_cfg.get("qlora", False)

    # Tokenizer/Model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"trust_remote_code": False}
    if not is_distributed and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if use_4bit and prepare_model_for_kbit_training:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_lora:
        if LoraConfig is None:
            raise SystemExit("peft not installed. Install `uv add peft`.")
        if use_4bit and prepare_model_for_kbit_training:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    datasets_dict = build_dataset(train_file, eval_file, tokenizer, block_size, text_fields)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trace
    tracer_choice = cfg_args.trace
    tracer = None
    callbacks = []
    if tracer_choice == "none":
        training_args.report_to = "none"
    else:
        tracer = init_trace(tracer_choice or yaml_cfg.get("trace", ""))

    if tracer == "wandb":
        wandb_cfg = load_optional_yaml("config/trace_config/wandb.yaml")
        project = wandb_cfg.get("project") or yaml_cfg.get("trace_project")
        entity = wandb_cfg.get("entity")
        api_key = wandb_cfg.get("api_key")
        mode = wandb_cfg.get("mode")
        base_url = wandb_cfg.get("base_url")
        run_name_override = wandb_cfg.get("run_name")
        notes = wandb_cfg.get("notes")
        tags = wandb_cfg.get("tags")

        if project:
            os.environ["WANDB_PROJECT"] = str(project)
        if entity:
            os.environ["WANDB_ENTITY"] = str(entity)
        if api_key:
            os.environ["WANDB_API_KEY"] = str(api_key)
        if mode:
            os.environ["WANDB_MODE"] = str(mode)
        if base_url:
            os.environ["WANDB_BASE_URL"] = str(base_url)
        if notes:
            os.environ["WANDB_NOTES"] = str(notes)
        if tags:
            os.environ["WANDB_TAGS"] = ",".join([str(t) for t in tags])
        if run_name_override and not training_args.run_name:
            training_args.run_name = str(run_name_override)

        def ensure_report_to_wandb():
            current = training_args.report_to
            if current is None:
                training_args.report_to = ["wandb"]
            elif isinstance(current, str):
                if current.lower() == "none":
                    training_args.report_to = ["wandb"]
                elif current != "wandb":
                    training_args.report_to = [current, "wandb"]
            else:
                if "wandb" not in current:
                    training_args.report_to = list(current) + ["wandb"]

        ensure_report_to_wandb()
        callbacks.append(WandbCallback())

    if tracer == "swanlab":
        swan_cfg = load_optional_yaml("config/trace_config/swanlab.yaml")
        project = swan_cfg.get("project") or yaml_cfg.get("trace_project")
        description = swan_cfg.get("description")
        run_name_override = swan_cfg.get("run_name")
        
        # 日志统一存储到项目根目录的 logs/ 下
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SWANLAB_LOG_DIR"] = str(log_dir)

        if run_name_override and not training_args.run_name:
            training_args.run_name = str(run_name_override)

        try:
            transformers_version = version.parse(transformers.__version__)
        except Exception:
            transformers_version = None

        def ensure_report_to_swanlab():
            current = training_args.report_to
            if current is None:
                training_args.report_to = ["swanlab"]
            elif isinstance(current, str):
                if current.lower() == "none":
                    training_args.report_to = ["swanlab"]
                elif current != "swanlab":
                    training_args.report_to = [current, "swanlab"]
            else:
                if "swanlab" not in current:
                    training_args.report_to = list(current) + ["swanlab"]

        if transformers_version and transformers_version >= version.parse("4.50.0"):
            ensure_report_to_swanlab()
        else:
            try:
                from swanlab.integration.transformers import SwanLabCallback  # type: ignore

                callbacks.append(
                    SwanLabCallback(
                        project=project,
                        experiment_name=training_args.run_name,
                        description=description,
                        logdir=str(log_dir),
                    )
                )
                training_args.report_to = "none"
            except Exception:
                print("[warn] failed to init swanlab callback; continue without it.")

    # 保留 Trainer 需要的列
    cols_to_remove = [c for c in datasets_dict["train"].column_names if c not in ["input_ids", "attention_mask", "labels"]]
    datasets_dict["train"] = datasets_dict["train"].remove_columns(cols_to_remove)
    if datasets_dict["eval"] is not None:
        datasets_dict["eval"] = datasets_dict["eval"].remove_columns(cols_to_remove)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets_dict["train"],
        eval_dataset=datasets_dict["eval"] if datasets_dict["eval"] is not None else None,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()

    if training_args.local_rank in [-1, 0]:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"[done] saved model+tokenizer to {training_args.output_dir}")


if __name__ == "__main__":
    main()
