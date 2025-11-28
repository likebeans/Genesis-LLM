#!/usr/bin/env python3
"""
HuggingFace 模型转换为 GGUF 格式并量化

完整流程：
1. 加载 HuggingFace 模型
2. 转换为 GGUF 格式（FP16）
3. 可选：进行量化（Q4_K_M 等）

依赖：
- pip install llama-cpp-python
- 或编译 llama.cpp 并安装其 Python 依赖

用法：
    # 仅转换
    python convert_to_gguf.py --model_path /path/to/hf_model --output_path model.gguf

    # 转换并量化
    python convert_to_gguf.py --model_path /path/to/hf_model --output_path model-q4km.gguf --quantize q4_k_m

    # 使用本地 llama.cpp
    python convert_to_gguf.py --model_path /path/to/hf_model --output_path model.gguf --llama_cpp_path /path/to/llama.cpp
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# 支持的量化类型
QUANTIZE_TYPES = [
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_0", "q4_1", "q4_k_s", "q4_k_m",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m",
    "q6_k", "q8_0",
    "f16", "f32",
]


def find_llama_cpp_path(custom_path: str | None = None) -> Path | None:
    """查找 llama.cpp 路径"""
    if custom_path:
        path = Path(custom_path)
        if path.exists():
            return path
        print(f"警告：指定的 llama.cpp 路径不存在: {custom_path}")

    # 常见路径
    common_paths = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("./llama.cpp"),
        Path("../llama.cpp"),
    ]

    for p in common_paths:
        if p.exists() and (p / "convert_hf_to_gguf.py").exists():
            return p

    return None


def convert_hf_to_gguf(
    model_path: str,
    output_path: str,
    llama_cpp_path: Path | None = None,
) -> bool:
    """将 HuggingFace 模型转换为 GGUF 格式"""

    print(f"📦 开始转换模型: {model_path}")
    print(f"📁 输出路径: {output_path}")

    if llama_cpp_path:
        # 使用 llama.cpp 的转换脚本
        convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            # 尝试旧版脚本名
            convert_script = llama_cpp_path / "convert.py"

        if not convert_script.exists():
            print(f"❌ 找不到转换脚本: {convert_script}")
            return False

        cmd = [
            sys.executable,
            str(convert_script),
            model_path,
            "--outfile", output_path,
            "--outtype", "f16",
        ]
    else:
        # 尝试使用 transformers + llama-cpp-python 方式
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print("使用 transformers 加载模型...")
            print("⚠️  推荐使用 llama.cpp 的转换脚本以获得最佳兼容性")

            # 这里只是示例，实际需要 llama.cpp 的转换脚本
            print("❌ 请指定 --llama_cpp_path 参数或安装 llama.cpp")
            print("   git clone https://github.com/ggerganov/llama.cpp")
            print("   pip install -r llama.cpp/requirements.txt")
            return False

        except ImportError as e:
            print(f"❌ 缺少依赖: {e}")
            return False

    print(f"🚀 执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("✅ 转换完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def quantize_gguf(
    input_path: str,
    output_path: str,
    quantize_type: str,
    llama_cpp_path: Path,
) -> bool:
    """对 GGUF 模型进行量化"""

    print(f"🔧 开始量化: {quantize_type}")

    # 查找量化工具
    quantize_bin = llama_cpp_path / "quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp_path / "build" / "bin" / "quantize"

    if not quantize_bin.exists():
        print(f"❌ 找不到量化工具: {quantize_bin}")
        print("   请先编译 llama.cpp: cd llama.cpp && make")
        return False

    cmd = [str(quantize_bin), input_path, output_path, quantize_type.upper()]

    print(f"🚀 执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("✅ 量化完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 量化失败: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace 模型转 GGUF 并量化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅转换为 FP16 GGUF
  python convert_to_gguf.py --model_path ./my_model --output_path model.gguf

  # 转换并量化为 Q4_K_M
  python convert_to_gguf.py --model_path ./my_model --output_path model-q4km.gguf --quantize q4_k_m

量化类型说明:
  q8_0    - 8-bit 量化，精度最高
  q5_k_m  - 5-bit 量化，平衡精度与大小
  q4_k_m  - 4-bit 量化，推荐选择
  q4_0    - 4-bit 量化，兼容性好
  q3_k_m  - 3-bit 量化，极限压缩
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace 模型路径（本地路径或 HF hub 名称）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出 GGUF 文件路径"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=QUANTIZE_TYPES,
        default=None,
        help="量化类型（不指定则输出 FP16）"
    )
    parser.add_argument(
        "--llama_cpp_path",
        type=str,
        default=None,
        help="llama.cpp 目录路径"
    )

    args = parser.parse_args()

    # 查找 llama.cpp
    llama_cpp_path = find_llama_cpp_path(args.llama_cpp_path)

    if not llama_cpp_path:
        print("❌ 找不到 llama.cpp，请通过 --llama_cpp_path 指定")
        print("   或克隆到常见位置:")
        print("   git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        sys.exit(1)

    print(f"📂 使用 llama.cpp: {llama_cpp_path}")

    # 确定输出路径
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.quantize:
        # 需要先转换为 FP16，再量化
        fp16_path = output_path.with_suffix(".fp16.gguf")

        # Step 1: 转换为 FP16 GGUF
        if not convert_hf_to_gguf(args.model_path, str(fp16_path), llama_cpp_path):
            sys.exit(1)

        # Step 2: 量化
        if not quantize_gguf(str(fp16_path), str(output_path), args.quantize, llama_cpp_path):
            sys.exit(1)

        # 可选：删除中间 FP16 文件
        # fp16_path.unlink()
        print(f"💡 FP16 中间文件保留在: {fp16_path}")

    else:
        # 仅转换为 FP16
        if not convert_hf_to_gguf(args.model_path, str(output_path), llama_cpp_path):
            sys.exit(1)

    print(f"\n🎉 完成！输出文件: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
