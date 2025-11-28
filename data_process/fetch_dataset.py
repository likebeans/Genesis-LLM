#!/usr/bin/env python3
"""
HuggingFace 数据集采集脚本

功能：从 HuggingFace Hub 下载数据集，保存为 JSONL 格式
设计理念：作为"搬运工"，不修改任何字段，不改变数据结构，保持原始数据完整性

使用示例：
    # 下载完整数据集
    python fetch_dataset.py --dataset_name shibing624/medical
    
    # 下载指定子目录（Parquet 格式）
    python fetch_dataset.py --dataset_name shibing624/medical \\
        --data_subdir pretrain --revision refs/convert/parquet \\
        --sample_limit 1000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有命令行参数的命名空间对象
    """
    parser = argparse.ArgumentParser(
        description="从 HuggingFace Hub 下载数据集并保存为 JSONL 格式，不修改原始数据"
    )
    
    # 数据集相关参数
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="HuggingFace 数据集名称，例如：shibing624/medical"
    )
    parser.add_argument(
        "--config_name",
        default=None,
        help="数据集配置名称（可选），用于有多个配置的数据集"
    )
    parser.add_argument(
        "--data_subdir",
        default=None,
        help="Parquet 数据子目录（可选），例如：pretrain/finetune/reward"
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="数据集版本/分支（可选），例如：refs/convert/parquet"
    )
    
    # 输出相关参数
    parser.add_argument(
        "--output_dir",
        default="data_process/fetch_data",
        help="JSONL 文件输出目录"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=0,
        help="每个拆分的采样数量限制（0 表示全量下载）"
    )
    
    # 配置和缓存相关参数
    parser.add_argument(
        "--config",
        default="config/model_config/config.yaml",
        help="YAML 配置文件路径，用于读取 HuggingFace 镜像地址"
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="仅使用本地缓存，不访问网络"
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="忽略本地缓存，强制从 Hub 重新下载"
    )
    
    return parser.parse_args()


def apply_hf_mirror(config_path: str) -> Optional[str]:
    """
    从 YAML 配置文件读取 HuggingFace 镜像地址并设置环境变量
    
    功能：
        1. 读取配置文件中的 huggingface.endpoint 或 hf.mirror 字段
        2. 设置 HF_ENDPOINT 环境变量，用于加速国内访问
        3. 如果配置文件不存在或解析失败，静默返回 None
    
    Args:
        config_path: YAML 配置文件路径
    
    Returns:
        Optional[str]: 镜像地址（如果成功设置），否则返回 None
    
    示例配置文件格式：
        huggingface:
          endpoint: https://hf-mirror.com
    """
    path = Path(config_path)
    
    # 检查配置文件是否存在
    if not path.exists():
        return None
    
    # 尝试导入 yaml 库
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    
    # 尝试解析 YAML 配置文件
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None
    
    # 读取 HuggingFace 配置（支持多种字段名）
    hf_cfg = cfg.get("huggingface") or cfg.get("hf") or {}
    endpoint = hf_cfg.get("endpoint") or hf_cfg.get("mirror")
    
    # 如果找到镜像地址，设置环境变量
    if endpoint:
        os.environ["HF_ENDPOINT"] = str(endpoint)
        return str(endpoint)
    
    return None


def to_jsonl(ds, out_path: Path) -> None:
    """
    将数据集保存为 JSONL 格式
    
    功能：
        1. 自动创建输出目录（如果不存在）
        2. 逐行写入 JSON 对象，每行一个记录
        3. 使用 UTF-8 编码，ensure_ascii=False 保留中文字符
    
    Args:
        ds: HuggingFace 数据集对象（可迭代）
        out_path: 输出文件路径
    
    输出格式：
        每行一个 JSON 对象，例如：
        {"text": "示例文本", "id": 123}
        {"text": "另一个示例", "id": 124}
    """
    # 确保输出目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入 JSONL 文件
    with out_path.open("w", encoding="utf-8") as f:
        for rec in ds:
            # ensure_ascii=False 保留中文字符，不转义为 \uXXXX
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def preview(ds, n: int = 5) -> None:
    """
    在终端预览数据集的前 N 行
    
    功能：
        用于快速验证数据格式和内容，避免下载完整数据集后才发现问题
    
    Args:
        ds: HuggingFace 数据集对象（可迭代）
        n: 预览行数，默认 5 行
    
    输出示例：
        [preview] first rows:
        {"text": "糖尿病是一种代谢性疾病...", "id": 1}
        {"text": "高血压是指...", "id": 2}
        ...
    """
    print("[preview] first rows:")
    for i, rec in enumerate(ds):
        if i >= n:
            break
        print(json.dumps(rec, ensure_ascii=False))


def main() -> None:
    """
    主函数：协调数据采集流程
    
    流程：
        1. 解析命令行参数
        2. 应用 HuggingFace 镜像配置（如果有）
        3. 加载数据集（支持标准格式和 Parquet 格式）
        4. 对每个拆分（train/validation/test）：
           - 采样（如果指定了 sample_limit）
           - 保存为 JSONL
           - 预览前 5 行
        5. 输出统计信息
    """
    args = parse_args()
    
    # 尝试应用 HuggingFace 镜像配置
    hf_endpoint = apply_hf_mirror(args.config)

    # 检查 datasets 库是否已安装
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "datasets 库未安装。请运行: uv add datasets"
        ) from exc

    # 设置下载模式：强制重新下载或使用缓存
    download_mode = "force_redownload" if args.force_download else None

    # === 加载数据集 ===
    # 根据是否指定 data_subdir 选择不同的加载方式
    
    if args.data_subdir:
        # 方式 1：加载 Parquet 格式的子目录数据
        # 适用于数据集有多个子目录（如 pretrain/finetune/reward）的情况
        
        # 构建 HuggingFace 数据集路径
        base = f"hf://datasets/{args.dataset_name}"
        if args.revision:
            base = f"{base}@{args.revision}"
        
        # 为每个拆分指定 Parquet 文件路径
        data_files = {
            "train": f"{base}/{args.data_subdir}/train/*.parquet",
            "validation": f"{base}/{args.data_subdir}/validation/*.parquet",
            "test": f"{base}/{args.data_subdir}/test/*.parquet",
        }
        
        # 使用 parquet 加载器
        ds_dict = load_dataset(
            "parquet",
            data_files=data_files,
            streaming=False,  # 非流式，全量下载
            download_mode=download_mode,
        )
    else:
        # 方式 2：使用标准数据集加载器
        # 适用于标准格式的 HuggingFace 数据集
        
        try:
            ds_dict = load_dataset(
                args.dataset_name,
                args.config_name,
                streaming=False,
                trust_remote_code=False,  # 安全考虑，不信任远程代码
                local_files_only=args.local_files_only,
                download_mode=download_mode,
                revision=args.revision,
            )
        except Exception:
            # 如果指定了 local_files_only 但加载失败，直接抛出异常
            if args.local_files_only:
                raise
            
            # 否则重试，不使用 local_files_only（可能是缓存问题）
            ds_dict = load_dataset(
                args.dataset_name,
                args.config_name,
                streaming=False,
                trust_remote_code=False,
                download_mode=download_mode,
                revision=args.revision,
            )

    # === 处理每个拆分（train/validation/test）===
    total_saved = 0
    
    for split_name, ds in ds_dict.items():
        # 当前拆分的数据集
        cur_ds = ds
        
        # 如果指定了采样限制，进行采样
        if args.sample_limit and args.sample_limit > 0:
            # 取 sample_limit 和实际数据量的较小值
            sample_size = min(args.sample_limit, len(cur_ds))
            cur_ds = cur_ds.select(range(sample_size))

        # 构建输出路径
        base_out_dir = Path(args.output_dir)
        if args.data_subdir:
            # 如果指定了子目录，添加到输出路径
            base_out_dir = base_out_dir / args.data_subdir
        
        # 输出文件名格式：{dataset_name}__{split}.jsonl
        # 例如：shibing624_medical__train.jsonl
        out_name = f"{args.dataset_name.replace('/', '_')}__{split_name}.jsonl"
        out_path = base_out_dir / out_name
        
        # 保存为 JSONL 格式
        to_jsonl(cur_ds, out_path)
        print(f"[done] saved {len(cur_ds)} samples to {out_path}")
        
        # 预览前 5 行数据
        preview(cur_ds)
        
        # 累计保存的样本数
        total_saved += len(cur_ds)

    # === 输出总结信息 ===
    if hf_endpoint:
        print(f"[info] HF_ENDPOINT={hf_endpoint}")
    print(f"[summary] total saved samples across splits: {total_saved}")


if __name__ == "__main__":
    main()
