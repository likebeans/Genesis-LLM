#!/usr/bin/env python3
"""
监督微调（SFT）训练脚本
==============================

功能：
    使用 Transformers + PEFT（LoRA/QLoRA）进行监督微调训练。
    支持配置驱动，从 YAML 文件读取所有训练参数。

使用方式：
    # 单卡训练
    uv run python self_model/fine_tuning/train_finetune.py \
        --config config/self_model_config/finetune.yaml

    # 多卡训练（torchrun）
    torchrun --nproc_per_node=4 train_finetune.py --config xxx.yaml

    # DeepSpeed 训练
    deepspeed --num_gpus=4 train_finetune.py --config xxx.yaml \
        --deepspeed config/self_model_config/deepspeed/ds_config.json

配置文件：
    - config/self_model_config/finetune.yaml（基础模板）
    - config/self_model_config/finetune_medical.yaml（医疗领域示例）
    - config/self_model_config/finetune_reference.yaml（参数参考文档）

作者：yff
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import datasets
import torch
import yaml
import transformers
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
class FinetuneConfig:
    """命令行参数配置类"""
    config: str = "config/self_model_config/finetune.yaml"
    trace: str = ""  # wandb / swanlab / none / empty
    deepspeed: str = ""  # DeepSpeed 配置文件路径
    local_rank: int = -1  # 分布式训练的 local_rank（torchrun 自动传入）


def load_yaml(path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件
    
    Args:
        path: YAML 文件路径
        
    Returns:
        解析后的配置字典
        
    Raises:
        SystemExit: 文件不存在时退出
    """
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


def parse_args() -> FinetuneConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SFT 微调训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run python train_finetune.py --config config/self_model_config/finetune.yaml
  uv run python train_finetune.py --config config/self_model_config/finetune_medical.yaml --trace swanlab
        """
    )
    parser.add_argument("--config", default="config/self_model_config/finetune.yaml", help="Path to finetune YAML.")
    parser.add_argument(
        "--trace",
        choices=["wandb", "swanlab", "none"],
        default=None,
        help="Override trace backend; leave empty to use YAML value.",
    )
    # 分布式训练参数
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
    return FinetuneConfig(
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
    max_length: int,
    template: Dict[str, Any]
) -> Dict[str, datasets.Dataset]:
    """
    构建训练和验证数据集
    
    Args:
        train_file: 训练数据文件路径（JSONL 格式）
        eval_file: 验证数据文件路径（可选）
        tokenizer: 分词器
        max_length: 最大序列长度
        template: 对话模板配置
        
    Returns:
        包含 'train' 和 'eval' 键的数据集字典
        
    支持的数据格式 (JSONL):
        1. 多轮对话: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        2. ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        3. 单轮问答: {"input": "问题", "output": "回答"}
        4. 指令格式: {"instruction": "任务说明", "input": "输入", "output": "输出"}
        5. Alpaca 格式: {"instruction": "...", "input": "...", "response": "..."}
    """
    def load_file(fp: str):
        """加载 JSONL 数据文件"""
        if not Path(fp).exists():
            raise SystemExit(f"[错误] 数据文件不存在: {fp}")
        return datasets.load_dataset("json", data_files=fp)["train"]

    def format_example(example):
        """
        格式化单条样本，支持多种数据格式：
        1. 多轮对话: {"messages": [{"role": "user", "content": "..."}, ...]}
        2. ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        3. 单轮问答: {"input": "...", "output": "..."}
        4. 指令格式: {"instruction": "...", "input": "...", "output": "..."}
        
        Labels 处理：对 prompt 部分 mask（设为 -100），只计算 response 的损失
        """
        # ShareGPT 格式转换为 messages 格式
        if "conversations" in example:
            # ShareGPT: {"conversations": [{"from": "human/gpt/system", "value": "..."}]}
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            messages = []
            for conv in example["conversations"]:
                role = role_map.get(conv.get("from", ""), conv.get("from", ""))
                content = conv.get("value", "")
                messages.append({"role": role, "content": content})
            example = {"messages": messages}
        
        # 尝试多种数据格式
        if "messages" in example:
            # 多轮对话格式 - 优先使用 tokenizer 的 chat template
            messages = example["messages"]
            if hasattr(tokenizer, "apply_chat_template"):
                # 使用 tokenizer 内置的 chat template
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                # 计算 prompt 长度（除最后一条 assistant 消息外的部分）
                prompt_messages = [m for m in messages if m.get("role") != "assistant"]
                if prompt_messages:
                    prompt_text = tokenizer.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_text = ""
            else:
                # 回退到模板配置
                text, prompt_text = _format_with_template(messages, template)
        else:
            # 单轮格式：input/output 或 instruction/input/output
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            out = example.get("output", example.get("response", ""))
            
            # 构建 prompt
            if instruction:
                prompt = f"{instruction}\n{inp}" if inp else instruction
            else:
                prompt = inp
            
            # 构建 messages 并使用 chat template
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": out}
            ]
            
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True
                )
            else:
                # 简单拼接
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
                prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        tok = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        
        # 创建 labels，对 prompt 部分和 padding 部分 mask
        labels = tok["input_ids"].copy()
        attention_mask = tok["attention_mask"]
        
        # 1. 对 prompt 部分 mask（设为 -100，不计算损失）
        if prompt_text:
            prompt_tok = tokenizer(prompt_text, truncation=True, max_length=max_length)
            prompt_len = len(prompt_tok["input_ids"])
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
        
        # 2. 对 padding 部分 mask（attention_mask=0 的位置设为 -100）
        for i in range(len(labels)):
            if attention_mask[i] == 0:
                labels[i] = -100
        
        tok["labels"] = labels
        return tok
    
    def _format_with_template(messages, template):
        """使用配置的模板格式化多轮对话（回退方案）"""
        sys_prompt = template.get("system", "You are a helpful assistant.")
        tpl_messages = template.get("messages", [])
        sep = template.get("separator", "")
        
        segments = []
        prompt_segments = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # 找到对应角色的模板
            tpl = ""
            for t in tpl_messages:
                if t.get("role") == role:
                    tpl = t.get("template", "{content}")
                    break
            
            if role == "system":
                formatted = tpl.format(system=content) if "{system}" in tpl else content
            else:
                formatted = tpl.format(content=content) if "{content}" in tpl else content
            
            segments.append(formatted)
            # prompt 包含除最后一条 assistant 外的所有内容
            if role != "assistant":
                prompt_segments.append(formatted)
        
        text = sep.join(segments)
        prompt_text = sep.join(prompt_segments)
        return text, prompt_text

    train_ds = load_file(train_file).map(format_example)
    eval_ds = load_file(eval_file).map(format_example) if eval_file else None
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
    # compatibility: rename evaluation_strategy -> eval_strategy
    if "evaluation_strategy" in training_args_dict and "eval_strategy" not in training_args_dict:
        training_args_dict["eval_strategy"] = training_args_dict.pop("evaluation_strategy")
    
    # === 分布式训练配置 ===
    # 优先使用命令行参数，其次使用 YAML 配置
    deepspeed_config = cfg_args.deepspeed or yaml_cfg.get("deepspeed")
    if deepspeed_config:
        training_args_dict["deepspeed"] = deepspeed_config
    
    # 设置 local_rank（torchrun/deepspeed 会自动设置环境变量）
    local_rank = cfg_args.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        training_args_dict["local_rank"] = local_rank
    
    # 检测是否为分布式训练环境
    is_distributed = (
        local_rank != -1 
        or os.environ.get("WORLD_SIZE") is not None
        or deepspeed_config is not None
    )
    
    training_args = TrainingArguments(**training_args_dict)

    # Data
    train_file = yaml_cfg["data"]["train_file"]
    eval_file = yaml_cfg["data"].get("eval_file")
    max_length = yaml_cfg["data"].get("max_length", 512)
    template_path = yaml_cfg.get("template_path")
    template = {}
    if template_path:
        template = json.loads(Path(template_path).read_text())

    # PEFT config
    peft_cfg = yaml_cfg.get("peft", {})
    use_lora = peft_cfg.get("enabled", False)
    lora_r = peft_cfg.get("lora_r", 8)
    lora_alpha = peft_cfg.get("lora_alpha", 16)
    lora_dropout = peft_cfg.get("lora_dropout", 0.05)
    target_modules = peft_cfg.get("target_modules", ["query", "key", "value"])
    use_4bit = peft_cfg.get("qlora", False)

    # Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
    
    # 确保 tokenizer 有 pad_token（某些模型如 LLaMA 没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 模型加载配置
    # 注意：分布式训练时不能使用 device_map="auto"，会与 DDP/FSDP/DeepSpeed 冲突
    model_kwargs = {
        "trust_remote_code": False,
    }
    
    # 非分布式训练时使用 device_map="auto" 自动分配设备
    if not is_distributed and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    
    # QLoRA 4-bit 量化
    if use_4bit and prepare_model_for_kbit_training:
        model_kwargs["load_in_4bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Apply PEFT if enabled
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

    # Data
    datasets_dict = build_dataset(train_file, eval_file, tokenizer, max_length, template)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trace
    # 日志统一存储到项目根目录的 logs/ 下
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    tracer_choice = cfg_args.trace
    tracer = None
    if tracer_choice == "none":
        training_args.report_to = "none"
    else:
        tracer = init_trace(tracer_choice or yaml_cfg.get("trace", ""))
    callbacks = []
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

        # 设置 wandb 日志目录
        os.environ["WANDB_DIR"] = str(log_dir)
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
            # WANDB_TAGS 逗号分隔
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
    elif tracer == "swanlab":
        swanlab_cfg = load_optional_yaml("config/trace_config/swanlab.yaml")
        project = swanlab_cfg.get("project") or yaml_cfg.get("trace_project", "self-model")
        workspace = swanlab_cfg.get("workspace")
        api_key = swanlab_cfg.get("api_key")
        api_host = swanlab_cfg.get("api_host")
        web_host = swanlab_cfg.get("web_host")
        mode = swanlab_cfg.get("mode")
        run_name_override = swanlab_cfg.get("run_name")
        description = swanlab_cfg.get("description")

        # 设置 swanlab 日志目录
        os.environ["SWANLAB_LOG_DIR"] = str(log_dir)
        if project:
            os.environ["SWANLAB_PROJECT"] = str(project)
        if workspace:
            os.environ["SWANLAB_WORKSPACE"] = str(workspace)
        if api_key:
            os.environ["SWANLAB_API_KEY"] = str(api_key)
        if api_host:
            os.environ["SWANLAB_API_HOST"] = str(api_host)
        if web_host:
            os.environ["SWANLAB_WEB_HOST"] = str(web_host)
        if mode:
            os.environ["SWANLAB_MODE"] = str(mode)
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
            # Fallback for transformers<4.50.0: use SwanLabCallback directly.
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

    # 移除数据集中不需要的列（只保留模型需要的 input_ids, attention_mask, labels）
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
        processing_class=tokenizer,  # 新版本使用 processing_class 替代 tokenizer
        callbacks=callbacks,
    )

    trainer.train()
    
    # 只在主进程保存模型（分布式训练时避免重复保存）
    if training_args.local_rank in [-1, 0]:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"[done] saved model+tokenizer to {training_args.output_dir}")


if __name__ == "__main__":
    main()
