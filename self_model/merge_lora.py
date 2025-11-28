#!/usr/bin/env python3
"""
LoRA 权重合并脚本

将 LoRA 适配器权重合并到基座模型，生成完整的可部署模型。

用法：
    # 基本用法
    uv run python self_model/merge_lora.py \
        --base_model Qwen/Qwen2.5-0.5B-Instruct \
        --lora_path self_model/checkpoints/finetune \
        --output_path self_model/checkpoints/finetune_merged

    # 使用本地基座模型
    uv run python self_model/merge_lora.py \
        --base_model /path/to/local/model \
        --lora_path self_model/checkpoints/finetune \
        --output_path self_model/checkpoints/finetune_merged

    # 保存为 safetensors 格式
    uv run python self_model/merge_lora.py \
        --base_model Qwen/Qwen2.5-0.5B-Instruct \
        --lora_path self_model/checkpoints/finetune \
        --output_path self_model/checkpoints/finetune_merged \
        --safe_serialization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    safe_serialization: bool = True,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
):
    """
    合并 LoRA 权重到基座模型

    Args:
        base_model_path: 基座模型路径（本地路径或 HuggingFace 模型名）
        lora_path: LoRA 适配器路径
        output_path: 合并后模型的输出路径
        device_map: 设备映射策略
        torch_dtype: 模型精度
        safe_serialization: 是否使用 safetensors 格式保存
        push_to_hub: 是否推送到 HuggingFace Hub
        hub_repo_id: HuggingFace Hub 仓库 ID
    """

    print("=" * 60)
    print("LoRA 权重合并")
    print("=" * 60)
    print(f"  基座模型: {base_model_path}")
    print(f"  LoRA 路径: {lora_path}")
    print(f"  输出路径: {output_path}")
    print("=" * 60)

    # 确定数据类型
    if torch_dtype == "auto":
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"\n📦 加载基座模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print(f"\n🔧 加载 LoRA 适配器: {lora_path}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=dtype,
    )

    # 打印 LoRA 配置信息
    lora_config_path = Path(lora_path) / "adapter_config.json"
    if lora_config_path.exists():
        with open(lora_config_path) as f:
            lora_config = json.load(f)
        print(f"\n📋 LoRA 配置:")
        print(f"   - r (rank): {lora_config.get('r', 'N/A')}")
        print(f"   - lora_alpha: {lora_config.get('lora_alpha', 'N/A')}")
        print(f"   - target_modules: {lora_config.get('target_modules', 'N/A')}")

    print("\n🔀 合并权重...")
    merged_model = model.merge_and_unload()

    # 创建输出目录
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 保存合并后的模型到: {output_path}")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=safe_serialization,
    )

    # 加载并保存 tokenizer
    print("\n💾 保存 tokenizer...")
    # 优先从 LoRA 路径加载 tokenizer（可能有修改）
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    tokenizer.save_pretrained(output_path)

    # 计算模型大小
    total_size = sum(
        f.stat().st_size for f in output_dir.glob("**/*") if f.is_file()
    )

    print(f"\n✅ 合并完成！")
    print(f"   输出目录: {output_dir.absolute()}")
    print(f"   总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")

    # 推送到 Hub
    if push_to_hub and hub_repo_id:
        print(f"\n🚀 推送到 HuggingFace Hub: {hub_repo_id}")
        merged_model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)
        print("   推送完成！")

    return merged_model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 权重合并工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合并 LoRA 到基座模型
  python merge_lora.py \\
    --base_model Qwen/Qwen2.5-0.5B-Instruct \\
    --lora_path self_model/checkpoints/finetune \\
    --output_path self_model/checkpoints/finetune_merged

  # 使用 float32 精度
  python merge_lora.py \\
    --base_model Qwen/Qwen2.5-0.5B-Instruct \\
    --lora_path self_model/checkpoints/finetune \\
    --output_path self_model/checkpoints/finetune_merged \\
    --torch_dtype float32
        """
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="基座模型路径（本地路径或 HuggingFace 模型名）"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="LoRA 适配器路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="合并后模型的输出路径"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="设备映射策略（默认: auto）"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="模型精度（默认: auto）"
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="使用 safetensors 格式保存（默认: True）"
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="使用 pytorch bin 格式保存"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="推送到 HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="HuggingFace Hub 仓库 ID"
    )

    args = parser.parse_args()

    # 处理 safe_serialization
    safe_serialization = not args.no_safe_serialization

    merge_lora(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        safe_serialization=safe_serialization,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
    )


if __name__ == "__main__":
    main()
