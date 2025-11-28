#!/usr/bin/env python3
"""
GPTQ 量化脚本

GPTQ (Generative Pre-trained Transformer Quantization) 是一种基于 Hessian 矩阵的
权重量化方法，通过逐层量化和误差补偿实现高压缩比。

依赖：
    pip install auto-gptq optimum transformers

用法：
    # 基本用法
    python gptq_quantize.py --model_path /path/to/model --output_path /path/to/output

    # 自定义参数
    python gptq_quantize.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --output_path ./qwen2.5-7b-gptq \
        --bits 4 \
        --group_size 128 \
        --desc_act true
"""

from __future__ import annotations

import argparse
from pathlib import Path


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = True,
    damp_percent: float = 0.1,
    calib_dataset: str = "wikitext2",
    calib_samples: int = 1024,
    calib_seq_len: int = 512,
    use_exllama: bool = True,
):
    """执行 GPTQ 量化"""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("   请安装: pip install auto-gptq optimum transformers")
        return False

    print("=" * 60)
    print("GPTQ 量化配置")
    print("=" * 60)
    print(f"  模型路径: {model_path}")
    print(f"  输出路径: {output_path}")
    print(f"  量化位数: {bits}")
    print(f"  分组大小: {group_size}")
    print(f"  激活排序: {desc_act}")
    print(f"  阻尼系数: {damp_percent}")
    print(f"  校准数据: {calib_dataset}")
    print(f"  校准样本: {calib_samples}")
    print("=" * 60)

    # 加载 tokenizer
    print("\n📦 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 准备校准数据
    print("\n📊 准备校准数据...")

    if calib_dataset == "wikitext2":
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        calib_texts = [text for text in dataset["text"] if len(text) > 100][:calib_samples]
    elif calib_dataset == "c4":
        from datasets import load_dataset
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        calib_texts = []
        for i, item in enumerate(dataset):
            if len(item["text"]) > 100:
                calib_texts.append(item["text"])
            if len(calib_texts) >= calib_samples:
                break
    else:
        # 从本地文件加载
        with open(calib_dataset, "r", encoding="utf-8") as f:
            calib_texts = [line.strip() for line in f if len(line.strip()) > 100][:calib_samples]

    # 按 calib_seq_len 截断校准文本（tokenize 后截断）
    # 这确保校准样本长度一致，避免过长样本影响量化效果
    truncated_texts = []
    for text in calib_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > calib_seq_len:
            tokens = tokens[:calib_seq_len]
        truncated_texts.append(tokenizer.decode(tokens))
    calib_texts = truncated_texts

    print(f"   已加载 {len(calib_texts)} 条校准样本（截断至 {calib_seq_len} tokens）")

    # 配置 GPTQ
    gptq_config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=damp_percent,
        dataset=calib_texts,
        tokenizer=tokenizer,
        use_exllama=use_exllama,
    )

    # 加载模型并量化
    print("\n🔧 加载模型并量化（这可能需要较长时间）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=gptq_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 保存模型
    print(f"\n💾 保存模型到: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 输出统计
    total_size = sum(f.stat().st_size for f in output_dir.glob("**/*") if f.is_file())
    print(f"\n✅ 量化完成！")
    print(f"   输出目录: {output_dir}")
    print(f"   总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ 量化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python gptq_quantize.py --model_path ./model --output_path ./model-gptq

  # 使用 HuggingFace 模型
  python gptq_quantize.py --model_path Qwen/Qwen2.5-7B-Instruct --output_path ./qwen2.5-7b-gptq

  # 高精度模式（更慢但精度更高）
  python gptq_quantize.py --model_path ./model --output_path ./model-gptq --desc_act true --calib_samples 2048
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
        choices=[2, 3, 4, 8],
        help="量化位数（默认: 4）"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="量化分组大小（默认: 128）"
    )
    parser.add_argument(
        "--desc_act",
        type=lambda x: x.lower() == "true",
        default=True,
        help="是否按激活值排序（默认: true）"
    )
    parser.add_argument(
        "--damp_percent",
        type=float,
        default=0.1,
        help="阻尼系数（默认: 0.1）"
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        help="校准数据集：wikitext2, c4, 或本地文件路径（默认: wikitext2）"
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=1024,
        help="校准样本数（默认: 1024）"
    )
    parser.add_argument(
        "--calib_seq_len",
        type=int,
        default=512,
        help="校准序列长度（默认: 512）"
    )
    parser.add_argument(
        "--use_exllama",
        type=lambda x: x.lower() == "true",
        default=True,
        help="是否使用 ExLlama 加速（默认: true）"
    )

    args = parser.parse_args()

    success = quantize_gptq(
        model_path=args.model_path,
        output_path=args.output_path,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        damp_percent=args.damp_percent,
        calib_dataset=args.calib_dataset,
        calib_samples=args.calib_samples,
        calib_seq_len=args.calib_seq_len,
        use_exllama=args.use_exllama,
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
