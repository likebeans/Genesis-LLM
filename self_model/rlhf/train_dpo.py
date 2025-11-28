#!/usr/bin/env python3
"""
RLHF - DPO 训练脚本
====================

功能：
    使用 TRL 的 DPOTrainer 对模型进行偏好对齐。
    配置驱动，支持 WandB/SwanLab 追踪，可选 LoRA/QLoRA。

数据格式（JSONL，列名至少包含 prompt/chosen/rejected）：
    {"prompt": "问题", "chosen": "正反馈回答", "rejected": "负反馈回答"}

使用方式：
    uv run python self_model/rlhf/train_dpo.py \
        --config config/self_model_config/rlhf.yaml
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import datasets
import torch
import transformers
import yaml
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

try:
    from trl import DPOTrainer
    from trl.trainer.dpo_config import DPOConfig
except ImportError:
    raise SystemExit("trl not installed. Install with `uv add trl` before running DPO.")

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
class DPOArgs:
    """命令行参数配置类"""

    config: str = "config/self_model_config/rlhf.yaml"
    trace: str = ""  # wandb / swanlab / none / empty
    deepspeed: str = ""
    local_rank: int = -1


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[错误] 配置文件不存在: {path}")
    return yaml.safe_load(p.read_text())


def load_optional_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    content = yaml.safe_load(p.read_text())
    return content or {}


def parse_args() -> DPOArgs:
    parser = argparse.ArgumentParser(
        description="DPO RLHF 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run python self_model/rlhf/train_dpo.py --config config/self_model_config/rlhf.yaml
        """
    )
    parser.add_argument("--config", default="config/self_model_config/rlhf.yaml", help="Path to rlhf YAML.")
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
    return DPOArgs(
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
) -> Dict[str, datasets.Dataset]:
    """加载并规范化 DPO 数据集，确保包含 prompt/chosen/rejected 三列"""

    def load_file(fp: str):
        if not Path(fp).exists():
            raise SystemExit(f"[错误] 数据文件不存在: {fp}")
        return datasets.load_dataset("json", data_files=fp)["train"]

    def normalize(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            example.get("prompt")
            or example.get("question")
            or example.get("query")
            or ""
        )
        chosen = (
            example.get("chosen")
            or example.get("response_chosen")
            or example.get("answer")
            or ""
        )
        rejected = (
            example.get("rejected")
            or example.get("response_rejected")
            or example.get("negative")
            or ""
        )
        return {"prompt": str(prompt), "chosen": str(chosen), "rejected": str(rejected)}

    def filter_valid(example: Dict[str, Any]) -> bool:
        return all(example.get(k, "").strip() for k in ["prompt", "chosen", "rejected"])

    train_ds = load_file(train_file).map(normalize)
    train_ds = train_ds.filter(filter_valid, desc="filter empty rows")
    eval_ds = None
    if eval_file:
        eval_ds = load_file(eval_file).map(normalize)
        eval_ds = eval_ds.filter(filter_valid, desc="filter empty rows (eval)")
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
    cfg_args = parse_args()
    yaml_cfg = load_yaml(cfg_args.config)

    # Model/tokenizer paths
    model_name = yaml_cfg.get("model_name_or_path", "self_model/checkpoints/finetune/final")
    ref_model_name = yaml_cfg.get("ref_model_name_or_path", model_name)
    tokenizer_name = yaml_cfg.get("tokenizer_name_or_path", model_name)

    # Training args
    training_args_dict = yaml_cfg.get("training_args", {})
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

    # Data
    train_file = yaml_cfg["data"]["train_file"]
    eval_file = yaml_cfg["data"].get("eval_file")
    max_length = int(yaml_cfg["data"].get("max_length", 1024))
    beta = float(yaml_cfg.get("beta", 0.1))
    loss_type = yaml_cfg.get("loss_type", "sigmoid")  # sigmoid / ipo / pai

    # DPO 使用 TRL 的 DPOConfig（继承 TrainingArguments），同时带 beta/loss/max_length 等
    training_args = DPOConfig(
        **training_args_dict,
        beta=beta,
        loss_type=loss_type,
        max_length=max_length,
        max_prompt_length=max_length // 2,
    )

    # PEFT 配置
    peft_cfg = yaml_cfg.get("peft", {})
    use_lora = peft_cfg.get("enabled", False)
    lora_r = peft_cfg.get("lora_r", 8)
    lora_alpha = peft_cfg.get("lora_alpha", 16)
    lora_dropout = peft_cfg.get("lora_dropout", 0.05)
    target_modules = peft_cfg.get("target_modules", ["query", "key", "value"])
    use_4bit = peft_cfg.get("qlora", False)

    # Tokenizer/Model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=False, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model_kwargs = {"trust_remote_code": False}
    # 不使用 device_map="auto"，让 Trainer 自己管理设备
    # 这样 CUDA_VISIBLE_DEVICES 才能生效
    if use_4bit and prepare_model_for_kbit_training:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"  # QLoRA 需要 device_map

    policy_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, **model_kwargs) if ref_model_name else None

    if use_lora:
        if LoraConfig is None:
            raise SystemExit("peft not installed. Install `uv add peft`.")
        if use_4bit and prepare_model_for_kbit_training:
            policy_model = prepare_model_for_kbit_training(policy_model)
            # 注意：ref_model 不需要 prepare_model_for_kbit_training
            # 参考模型应保持冻结，作为计算 KL 散度的基线
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, lora_config)
        # 注意：ref_model 不应用 LoRA
        # DPO 中参考模型必须保持原始权重，用于提供基线 logits π_ref(y|x)
        # 如果对 ref_model 也加 LoRA，会篡改其输出并违背 DPO 目标
        policy_model.print_trainable_parameters()

    datasets_dict = build_dataset(train_file, eval_file)

    # Trace
    tracer_choice = cfg_args.trace
    tracer = None
    callbacks: list[TrainerCallback] = []
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

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        train_dataset=datasets_dict["train"],
        eval_dataset=datasets_dict["eval"],
        processing_class=tokenizer,  # TRL >= 0.12 使用 processing_class 替代 tokenizer
        args=training_args,
    )

    if callbacks:
        trainer.add_callback(callbacks[0])  # SwanLab 使用 callback 方式时添加

    trainer.train()

    if training_args.local_rank in [-1, 0]:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"[done] saved model+tokenizer to {training_args.output_dir}")


if __name__ == "__main__":
    main()
