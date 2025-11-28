#!/usr/bin/env python3
"""
RLHF - GRPO 训练脚本（基于 VERL）
=================================

功能：
    复用 PPO 入口，但自动设置 algorithm=grpo（adv_estimator=grpo, use_kl_loss=True）。

使用方式：
    uv run python self_model/rlhf/train_grpo.py --config config/self_model_config/rlhf.yaml
"""
from __future__ import annotations

from self_model.rlhf.train_ppo import main as _ppo_main  # type: ignore


def main():
    # 直接委托给 PPO 脚本，传入 grpo
    import sys

    sys.argv.extend(["--algorithm", "grpo"])
    _ppo_main()


if __name__ == "__main__":
    main()
