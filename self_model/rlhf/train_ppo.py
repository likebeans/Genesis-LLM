#!/usr/bin/env python3
"""
RLHF - PPO/GRPO 训练脚本（基于 VERL）
=====================================

功能：
    使用 VERL 的 PPO 主入口（main_ppo.run_ppo）进行 PPO/GRPO 训练。
    通过本地 YAML（config/self_model_config/rlhf.yaml）提供少量关键字段，
    自动覆盖 VERL 默认的 Hydra 配置，避免直接写长串 CLI override。

前置依赖：
    pip install "verl>=0.3.0" ray omegaconf
    # rollout 默认用 vLLM，如需请确保安装对应版本（或改为 hf）

使用方式：
    # PPO
    uv run python self_model/rlhf/train_ppo.py --config config/self_model_config/rlhf.yaml
    # GRPO（自动设置 adv_estimator=grpo + use_kl_loss）
    uv run python self_model/rlhf/train_ppo.py --config config/self_model_config/rlhf.yaml --algorithm grpo
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List

import yaml
from omegaconf import OmegaConf


def _require_verl():
    """确保 verl 可用并返回 run_ppo 入口"""
    try:
        from verl.trainer.main_ppo import run_ppo  # type: ignore
    except Exception as exc:  # pragma: no cover - 明确提示依赖
        raise SystemExit(
            "[错误] 未找到 verl，请先安装：pip install \"verl>=0.3.0\" ray omegaconf\n"
            f"详情：{exc}"
        )
    return run_ppo


@dataclass
class PPOCliArgs:
    config: str = "config/self_model_config/rlhf.yaml"
    algorithm: str = "ppo"  # ppo 或 grpo


def parse_args() -> PPOCliArgs:
    parser = argparse.ArgumentParser(description="RLHF PPO/GRPO 训练脚本（VERL）")
    parser.add_argument("--config", default="config/self_model_config/rlhf.yaml", help="配置文件路径")
    parser.add_argument("--algorithm", choices=["ppo", "grpo"], default="ppo", help="选择 PPO 或 GRPO")
    args = parser.parse_args()
    return PPOCliArgs(config=args.config, algorithm=args.algorithm)


def load_yaml_cfg(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[错误] 配置文件不存在: {path}")
    return yaml.safe_load(p.read_text()) or {}


def load_verl_base_cfg():
    """加载 VERL 自带的 ppo_trainer.yaml 作为基底"""
    try:
        cfg_path = resources.files("verl.trainer.config").joinpath("ppo_trainer.yaml")
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"[错误] 找不到 VERL 的默认配置: {exc}")
    base_cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(base_cfg, False)  # 便于动态写入
    return base_cfg


def ensure_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def apply_overrides(base_cfg, yaml_cfg: Dict[str, Any], algo: str):
    """将自定义 YAML 字段覆盖到 VERL 默认配置"""
    model_path = yaml_cfg.get("model_name_or_path")
    if not model_path:
        raise SystemExit("[错误] 请在 rlhf.yaml 中设置 model_name_or_path（策略模型路径）。")
    tokenizer_path = yaml_cfg.get("tokenizer_name_or_path", model_path)
    reward_model_path = yaml_cfg.get("reward_model_name_or_path")
    trust_remote_code = bool(yaml_cfg.get("trust_remote_code", False))

    ppo_cfg = yaml_cfg.get("ppo", {})

    # 数据相关
    data_train = ensure_list(ppo_cfg.get("train_file"))
    data_val = ensure_list(ppo_cfg.get("eval_file"))
    if not data_train:
        raise SystemExit("[错误] 请在 rlhf.yaml 的 ppo.train_file 中指定训练数据路径（建议 parquet/jsonl 转 parquet）。")
    base_cfg.data.train_files = data_train
    if data_val:
        base_cfg.data.val_files = data_val
    base_cfg.data.max_prompt_length = int(ppo_cfg.get("max_prompt_length", base_cfg.data.max_prompt_length))
    base_cfg.data.max_response_length = int(ppo_cfg.get("max_response_length", base_cfg.data.max_response_length))
    base_cfg.data.train_batch_size = int(ppo_cfg.get("train_batch_size", base_cfg.data.train_batch_size))
    if ppo_cfg.get("val_batch_size") is not None:
        base_cfg.data.val_batch_size = int(ppo_cfg["val_batch_size"])
    if ppo_cfg.get("prompt_key"):
        base_cfg.data.prompt_key = str(ppo_cfg["prompt_key"])

    # 模型与 rollout
    base_cfg.actor_rollout_ref.model.path = model_path
    base_cfg.actor_rollout_ref.model.tokenizer_path = tokenizer_path
    base_cfg.actor_rollout_ref.model.trust_remote_code = trust_remote_code
    if ppo_cfg.get("gradient_checkpointing") is not None:
        base_cfg.actor_rollout_ref.model.enable_gradient_checkpointing = bool(ppo_cfg["gradient_checkpointing"])
    if ppo_cfg.get("use_remove_padding") is not None:
        base_cfg.actor_rollout_ref.model.use_remove_padding = bool(ppo_cfg["use_remove_padding"])
    if yaml_cfg.get("peft", {}).get("enabled"):
        peft_cfg = yaml_cfg["peft"]
        base_cfg.actor_rollout_ref.model.lora_rank = peft_cfg.get("lora_r", base_cfg.actor_rollout_ref.model.lora_rank)
        base_cfg.actor_rollout_ref.model.lora_alpha = peft_cfg.get("lora_alpha", base_cfg.actor_rollout_ref.model.lora_alpha)
        target_modules = peft_cfg.get("target_modules")
        if target_modules:
            base_cfg.actor_rollout_ref.model.target_modules = target_modules

    rollout_backend = ppo_cfg.get("rollout_backend", "vllm")
    base_cfg.actor_rollout_ref.rollout.name = rollout_backend
    base_cfg.actor_rollout_ref.rollout.load_format = ppo_cfg.get("load_format", "hf")
    base_cfg.actor_rollout_ref.rollout.n = int(ppo_cfg.get("rollout_n", 4 if algo == "grpo" else 1))
    base_cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = int(
        ppo_cfg.get("rollout_tp", base_cfg.actor_rollout_ref.rollout.tensor_model_parallel_size)
    )
    base_cfg.actor_rollout_ref.rollout.gpu_memory_utilization = float(
        ppo_cfg.get("rollout_gpu_memory_utilization", base_cfg.actor_rollout_ref.rollout.gpu_memory_utilization)
    )
    base_cfg.actor_rollout_ref.rollout.prompt_length = base_cfg.data.max_prompt_length
    base_cfg.actor_rollout_ref.rollout.response_length = base_cfg.data.max_response_length
    if ppo_cfg.get("log_prob_micro_batch_size_per_gpu") is not None:
        base_cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = int(
            ppo_cfg["log_prob_micro_batch_size_per_gpu"]
        )

    # Actor (policy) 超参
    base_cfg.actor_rollout_ref.actor.optim.lr = float(ppo_cfg.get("actor_lr", base_cfg.actor_rollout_ref.actor.optim.lr))
    if ppo_cfg.get("ppo_epochs") is not None:
        base_cfg.actor_rollout_ref.actor.ppo_epochs = int(ppo_cfg["ppo_epochs"])
    if ppo_cfg.get("ppo_mini_batch_size") is not None:
        base_cfg.actor_rollout_ref.actor.ppo_mini_batch_size = int(ppo_cfg["ppo_mini_batch_size"])
    if ppo_cfg.get("ppo_micro_batch_size_per_gpu") is not None:
        base_cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = int(ppo_cfg["ppo_micro_batch_size_per_gpu"])
    base_cfg.actor_rollout_ref.actor.use_kl_loss = bool(ppo_cfg.get("use_kl_loss", algo == "grpo"))
    if ppo_cfg.get("kl_loss_coef") is not None:
        base_cfg.actor_rollout_ref.actor.kl_loss_coef = float(ppo_cfg["kl_loss_coef"])
    if ppo_cfg.get("kl_loss_type"):
        base_cfg.actor_rollout_ref.actor.kl_loss_type = str(ppo_cfg["kl_loss_type"])
    if ppo_cfg.get("entropy_coeff") is not None:
        base_cfg.actor_rollout_ref.actor.entropy_coeff = float(ppo_cfg["entropy_coeff"])

    # Algorithm
    base_cfg.algorithm.adv_estimator = "grpo" if algo == "grpo" else ppo_cfg.get("adv_estimator", "gae")
    if ppo_cfg.get("use_kl_in_reward") is not None:
        base_cfg.algorithm.use_kl_in_reward = bool(ppo_cfg["use_kl_in_reward"])
    if ppo_cfg.get("kl_coef") is not None:
        base_cfg.algorithm.kl_ctrl.kl_coef = float(ppo_cfg["kl_coef"])
    if ppo_cfg.get("target_kl") is not None:
        base_cfg.algorithm.kl_ctrl.target_kl = float(ppo_cfg["target_kl"])

    # 奖励模型
    if reward_model_path:
        base_cfg.reward_model.enable = True
        base_cfg.reward_model.model.path = reward_model_path
        base_cfg.reward_model.model.input_tokenizer = tokenizer_path
        base_cfg.reward_model.model.trust_remote_code = trust_remote_code
        if ppo_cfg.get("reward_micro_batch_size_per_gpu") is not None:
            base_cfg.reward_model.micro_batch_size_per_gpu = int(ppo_cfg["reward_micro_batch_size_per_gpu"])
        if ppo_cfg.get("reward_max_length") is not None:
            base_cfg.reward_model.max_length = int(ppo_cfg["reward_max_length"])

    # Trainer/日志
    trainer_cfg = base_cfg.trainer
    trainer_cfg.total_epochs = int(ppo_cfg.get("total_epochs", trainer_cfg.total_epochs))
    trainer_cfg.save_freq = int(ppo_cfg.get("save_freq", trainer_cfg.save_freq))
    trainer_cfg.test_freq = int(ppo_cfg.get("test_freq", trainer_cfg.test_freq))
    trainer_cfg.n_gpus_per_node = int(ppo_cfg.get("n_gpus_per_node", trainer_cfg.n_gpus_per_node))
    trainer_cfg.nnodes = int(ppo_cfg.get("nnodes", trainer_cfg.nnodes))
    trainer_cfg.project_name = ppo_cfg.get("project_name", trainer_cfg.project_name)
    trainer_cfg.experiment_name = ppo_cfg.get("experiment_name", trainer_cfg.experiment_name)
    logger = ppo_cfg.get("logger", ["console"])
    if isinstance(logger, str):
        logger = [logger]
    trainer_cfg.logger = logger
    trainer_cfg.val_before_train = bool(ppo_cfg.get("val_before_train", trainer_cfg.val_before_train))
    trainer_cfg.resume_mode = ppo_cfg.get("resume_mode", trainer_cfg.resume_mode)

    # 追踪（沿用 trace 字段：wandb/none）
    trace_choice = yaml_cfg.get("trace", "")
    if trace_choice:
        if trace_choice.lower() == "wandb":
            if "wandb" not in trainer_cfg.logger:
                trainer_cfg.logger = list(trainer_cfg.logger) + ["wandb"]
        elif trace_choice.lower() == "none":
            trainer_cfg.logger = ["console"]

    return base_cfg


def main():
    cli_args = parse_args()
    yaml_cfg = load_yaml_cfg(cli_args.config)
    run_ppo = _require_verl()
    base_cfg = load_verl_base_cfg()
    merged_cfg = apply_overrides(base_cfg, yaml_cfg, cli_args.algorithm.lower())
    # 调用 VERL 主入口
    run_ppo(merged_cfg)


if __name__ == "__main__":
    main()
