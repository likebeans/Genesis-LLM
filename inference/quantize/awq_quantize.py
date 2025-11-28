#!/usr/bin/env python3
"""
AWQ 量化脚本

AWQ (Activation-aware Weight Quantization) 是一种基于激活值分布的权重量化方法，
能在保持较高精度的同时将模型压缩到 4-bit。

依赖：
    pip install autoawq transformers

用法：
    # 基本用法
    python awq_quantize.py --model_path /path/to/model --output_path /path/to/output

    # 自定义参数
    python awq_quantize.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --output_path ./qwen2.5-7b-awq \
        --bits 4 \
        --group_size 128 \
        --calib_samples 512
"""

from __future__ import annotations

import argparse
from pathlib import Path


def quantize_awq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    calib_data: str = "wikitext",
    calib_samples: int = 512,
    calib_seq_len: int = 512,
):
    """执行 AWQ 量化"""

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("   请安装: pip install autoawq transformers")
        return False

    print("=" * 60)
    print("AWQ 量化配置")
    print("=" * 60)
    print(f"  模型路径: {model_path}")
    print(f"  输出路径: {output_path}")
    print(f"  量化位数: {bits}")
    print(f"  分组大小: {group_size}")
    print(f"  零点量化: {zero_point}")
    print(f"  校准数据: {calib_data}")
    print(f"  校准样本: {calib_samples}")
    print("=" * 60)

    # 加载模型
    print("\n📦 加载模型...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 量化配置
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",  # GEMM 或 GEMV
    }

    # 执行量化
    print("\n🔧 开始量化...")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
        n_samples=calib_samples,
        seqlen=calib_seq_len,
    )

    # 保存模型
    print(f"\n💾 保存模型到: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 输出统计
    total_size = sum(f.stat().st_size for f in output_dir.glob("**/*") if f.is_file())
    print(f"\n✅ 量化完成！")
    print(f"   输出目录: {output_dir}")
    print(f"   总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="AWQ 量化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python awq_quantize.py --model_path ./model --output_path ./model-awq

  # 使用 HuggingFace 模型
  python awq_quantize.py --model_path Qwen/Qwen2.5-7B-Instruct --output_path ./qwen2.5-7b-awq

  # 自定义量化参数
  python awq_quantize.py --model_path ./model --output_path ./model-awq --bits 4 --group_size 64
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径（本地路径或 HuggingFace 模型名）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出目录路径"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="量化位数（默认: 4）"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="量化分组大小（默认: 128）"
    )
    parser.add_argument(
        "--zero_point",
        action="store_true",
        default=True,
        help="启用零点量化（默认启用）"
    )
    parser.add_argument(
        "--no_zero_point",
        action="store_true",
        help="禁用零点量化"
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default="wikitext",
        help="校准数据集（默认: wikitext）"
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=512,
        help="校准样本数（默认: 512）"
    )
    parser.add_argument(
        "--calib_seq_len",
        type=int,
        default=512,
        help="校准序列长度（默认: 512）"
    )

    args = parser.parse_args()

    # 处理 zero_point 参数：--no_zero_point 优先级高于 --zero_point
    zero_point = not args.no_zero_point
    
    success = quantize_awq(
        model_path=args.model_path,
        output_path=args.output_path,
        bits=args.bits,
        group_size=args.group_size,
        zero_point=zero_point,
        calib_data=args.calib_data,
        calib_samples=args.calib_samples,
        calib_seq_len=args.calib_seq_len,
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
