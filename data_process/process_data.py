#!/usr/bin/env python3
"""
数据处理脚本：清洗、去重和质量筛选

功能：
    1. 文本清洗：去除 HTML/Markdown 标签、控制字符、多余空白
    2. 长度过滤：过滤过短或过长的文本
    3. 精确去重：基于 MD5 哈希去除完全相同的样本
    4. 相似去重：基于 3-gram Jaccard 相似度去除近似重复样本
    5. 情感过滤：可选的情感分析过滤（保留正面或负面情感）
    6. 多轮对话处理：支持限制对话轮数
    7. 多格式输出：支持 JSONL 和 Parquet 格式

设计理念：
    - 提供灵活的清洗策略，可通过命令行参数控制
    - 输出详细的统计日志，便于评估数据质量
    - 支持批量处理和单文件处理

使用示例：
    # 基础清洗
    python process_data.py --input data.jsonl --output_dir output
    
    # 完整处理流程
    python process_data.py --input data.jsonl --output_dir output \\
        --dedup --near_dedup --sentiment_filter positive \\
        --min_chars 10 --max_chars 1000 --to_parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    配置输入输出路径、清洗规则（长度限制）、去重开关以及情感过滤设置。
    """
    parser = argparse.ArgumentParser(description="Process text dataset: clean, dedup, filter.")
    
    # 输入输出文件路径
    parser.add_argument("--input", required=True, help="Input JSONL file with 'text' field.")
    parser.add_argument("--output", default="", help="Output JSONL path for cleaned data. If empty, auto-generate under --output_dir.")
    parser.add_argument("--output_dir", default="data_process/final_data", help="Base directory for outputs when --output is empty.")
    parser.add_argument("--subdir", default=None, help="Optional subdir under output_dir (e.g., pretrain/finetune/reward). If omitted, try to infer from input path.")
    parser.add_argument("--text_fields", default="text", help="Comma-separated fields to concatenate if 'text' is absent (e.g., instruction,input,output).")
    
    # 字符长度过滤阈值
    parser.add_argument("--min_chars", type=int, default=5, help="Drop samples shorter than this.")
    parser.add_argument("--max_chars", type=int, default=2000, help="Drop samples longer than this.")
    
    # 去重开关
    parser.add_argument("--dedup", action="store_true", help="Enable exact dedup (hash-based).")
    parser.add_argument("--near_dedup", action="store_true", help="Enable simple near-duplicate filtering (3-gram Jaccard).")
    parser.add_argument("--jaccard_threshold", type=float, default=0.9, help="Near-dup Jaccard threshold.")
    
    # 情感过滤配置
    parser.add_argument("--sentiment_filter", choices=["positive", "negative"], default=None, help="Keep only this sentiment (requires model).")
    parser.add_argument("--sentiment_model", default="uer/roberta-base-finetuned-jd-binary-chinese", help="Transformers model for sentiment.")
    
    # 对话数据处理：限制最大轮数
    parser.add_argument("--max_turns", type=int, default=1, help="Max dialog turns allowed when input has `dialog` field (1 for single-turn). 0 or negative disables turn filtering.")
    
    # 可选：输出 parquet
    parser.add_argument("--to_parquet", action="store_true", help="Also write cleaned data to parquet.")
    
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """
    文本清洗与规范化函数
    
    功能：
        1. 去除 HTML 标签（如 <div>, <p> 等）
        2. 去除 Markdown 链接格式，保留链接文本
        3. 移除非打印字符（保留制表符、换行符等基础空白符）
        4. 将连续的空白字符合并为一个空格，并去除首尾空白
    
    Args:
        text: 待清洗的原始文本
    
    Returns:
        str: 清洗后的文本
    
    示例：
        输入: "<p>这是一个<b>示例</b>文本</p>  \n\n  多余空格"
        输出: "这是一个示例文本 多余空格"
    """
    # 步骤 1: 去除 HTML 标签（匹配 <...> 格式）
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 步骤 2: 去除 Markdown 链接格式 [文本](链接)，保留方括号内的文本
    # 例如：[百度](https://baidu.com) -> 百度
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    
    # 步骤 3: 过滤非打印字符，但保留基础空白符（\t, \n, \r, 空格）
    # 这可以去除控制字符、零宽字符等不可见字符
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\t\n\r ")
    
    # 步骤 4: 合并多余空格（包括制表符、换行符等）并去除首尾空格
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def shingle_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    生成文本的 N-gram 字符切片集合（Shingles）
    
    功能：
        将文本切分为连续的 n 个字符的子串集合，用于计算文本相似度
        这是 MinHash 和 Jaccard 相似度计算的基础
    
    Args:
        text: 输入文本
        n: N-gram 的大小，默认为 3（字符级别）
    
    Returns:
        Set[str]: N-gram 切片的集合
    
    示例：
        输入: "你好世界", n=3
        输出: {"你好世", "好世界"}
        
        输入: "hello", n=3
        输出: {"hel", "ell", "llo"}
    
    注意：
        - 使用字符级别的 N-gram，适合中文文本
        - 对于长度小于 n 的文本，返回包含整个文本的集合
    """
    return {text[i : i + n] for i in range(max(len(text) - n + 1, 1))}


def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    计算两个集合的 Jaccard 相似系数
    
    功能：
        衡量两个集合的相似程度，常用于文本去重
        公式：J(A, B) = |A ∩ B| / |A ∪ B|
    
    Args:
        a: 第一个集合（通常是文本的 N-gram 集合）
        b: 第二个集合
    
    Returns:
        float: Jaccard 相似系数，范围 [0, 1]
               - 0 表示完全不同
               - 1 表示完全相同
    
    示例：
        a = {"你好", "好世", "世界"}
        b = {"你好", "好啊"}
        jaccard(a, b) = 1/4 = 0.25  # 交集1个，并集4个
    
    特殊情况：
        - 两个空集返回 1.0（视为相同）
        - 并集为空返回 0.0
    """
    # 特殊情况：两个空集视为相同
    if not a and not b:
        return 1.0
    
    # 计算交集和并集的大小
    inter = len(a & b)  # 交集
    union = len(a | b)  # 并集
    
    # 返回 Jaccard 系数
    return inter / union if union else 0.0


def sentiment_keep(label: str, target: str) -> bool:
    """
    判断情感分析结果是否匹配目标情感
    
    功能：
        检查模型预测的情感标签是否与目标情感一致
        使用前缀匹配，忽略大小写
    
    Args:
        label: 模型预测的情感标签（如 "POSITIVE", "NEGATIVE", "positive"）
        target: 目标情感（如 "positive", "negative"）
    
    Returns:
        bool: 是否匹配
    
    示例：
        sentiment_keep("POSITIVE", "positive") -> True
        sentiment_keep("positive", "pos") -> True
        sentiment_keep("NEGATIVE", "positive") -> False
    
    注意：
        使用前缀匹配，因此 "pos" 可以匹配 "positive"
    """
    return label.lower().startswith(target.lower())


def load_sentiment_pipeline(model: str):
    """
    加载 HuggingFace Transformers 的情感分析 pipeline
    
    功能：
        创建一个情感分析模型的推理管道，用于对文本进行情感分类
    
    Args:
        model: HuggingFace 模型名称或本地路径
               例如："uer/roberta-base-finetuned-jd-binary-chinese"
    
    Returns:
        Pipeline: Transformers 的 sentiment-analysis pipeline 对象
    
    Raises:
        SystemExit: 如果 transformers 库未安装
    
    使用示例：
        pipe = load_sentiment_pipeline("uer/roberta-base-finetuned-jd-binary-chinese")
        result = pipe("这个产品很好用")
        # result: [{"label": "positive", "score": 0.98}]
    
    注意：
        - 首次使用会从 HuggingFace Hub 下载模型
        - 建议配置 HF 镜像加速国内下载
    """
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "transformers 库未安装。情感过滤需要此库。\n"
            "请运行: uv add transformers"
        ) from exc
    
    return pipeline("sentiment-analysis", model=model)


def process_file(args: argparse.Namespace, inp: Path) -> None:
    """
    处理单个 JSONL 文件的核心函数
    
    功能：
        对输入的 JSONL 文件进行完整的数据处理流程：
        1. 文本提取和拼接
        2. 文本清洗和规范化
        3. 长度过滤
        4. 精确去重（基于哈希）
        5. 相似去重（基于 Jaccard 相似度）
        6. 情感过滤（可选）
        7. 输出清洗后的数据和统计日志
    
    Args:
        args: 命令行参数对象，包含所有配置选项
        inp: 输入 JSONL 文件的路径
    
    处理流程：
        原始 JSONL -> 文本提取 -> 清洗 -> 长度过滤 -> 精确去重 -> 
        相似去重 -> 情感过滤 -> 输出 JSONL/Parquet + 统计日志
    
    输出文件：
        - {input}_clean.jsonl: 清洗后的数据
        - {input}_clean.jsonl.log.json: 处理统计日志
        - {input}_clean.jsonl.parquet: Parquet 格式（可选）
    
    性能注意：
        - 相似去重使用 O(N²) 算法，大数据集（>100万）会较慢
        - shingle_cache 会随数据量线性增长，占用内存
    """
    # === 步骤 1: 确定输出路径 ===
    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
    else:
        # 如果未指定输出路径，自动生成
        subdir = args.subdir
        
        # 尝试从输入路径推断子目录（pretrain/finetune/reward）
        if subdir is None:
            parts = inp.parts
            for i, p in enumerate(parts):
                # 查找 "fetch_data" 后的子目录名
                if p == "fetch_data" and i + 1 < len(parts):
                    subdir = parts[i + 1]
                    break
        
        # 如果无法推断，要求用户明确指定
        if subdir is None:
            raise SystemExit(
                "无法从输入路径推断子目录。\n"
                "请使用 --subdir 参数指定（例如：pretrain/finetune/reward）"
            )
        
        # 构建输出路径：output_dir/subdir/filename_clean.jsonl
        base_out = Path(args.output_dir) / subdir
        fname = inp.name.replace(".jsonl", "_clean.jsonl")
        outp = base_out / fname
        outp.parent.mkdir(parents=True, exist_ok=True)

    # === 步骤 2: 初始化情感分析模型（如果需要）===
    senti_pipe = load_sentiment_pipeline(args.sentiment_model) if args.sentiment_filter else None

    # === 步骤 3: 初始化统计信息字典 ===
    # 用于记录处理过程中各个环节的数据流向，便于评估数据质量
    stats = {
        "total": 0,                # 总样本数（输入行数）
        "empty_after_clean": 0,    # 清洗后为空的样本数
        "too_short": 0,            # 长度过短被过滤的样本数
        "too_long": 0,             # 长度过长被过滤的样本数
        "non_printable_ratio": 0,  # （保留字段，未实现）
        "dedup": 0,                # 精确去重过滤的样本数
        "near_dedup": 0,           # 相似去重过滤的样本数
        "sentiment_filtered": 0,   # 情感过滤掉的样本数
        "kept": 0,                 # 最终保留的样本数
        "too_many_turns": 0,       # 对话轮数过多被过滤的样本数
    }
    
    # === 步骤 4: 初始化去重相关数据结构 ===
    seen_hashes = set()  # 存储已处理文本的 MD5 哈希，用于精确去重
    shingle_cache: List[Tuple[Set[str], int]] = []  # 存储 (N-gram集合, 索引)，用于相似去重
    
    # === 步骤 5: 初始化 Parquet 输出（如果需要）===
    pq_writer = None
    pq_buffer: List[dict] = []
    pq_path = None
    if args.to_parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq_path = outp.with_suffix(outp.suffix + ".parquet")
    
    # 性能注意：shingle_cache 会随着数据量线性增长
    # 对于超大数据集（>100万样本），可能占用数GB内存

    # === 步骤 6: 解析文本字段候选列表 ===
    # 用于从 JSON 对象中提取文本（如果没有 "text" 字段）
    text_field_candidates = [f.strip() for f in args.text_fields.split(",") if f.strip()]

    # === 步骤 7: 开始处理文件 ===
    with inp.open("r", encoding="utf-8") as fin, outp.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            stats["total"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            text = obj.get("text")
            dialog = obj.get("dialog")

            # 如果没有 text，尝试从指定字段拼接
            if text is None and text_field_candidates:
                parts = []
                for field in text_field_candidates:
                    val = obj.get(field)
                    if val:
                        parts.append(str(val))
                if parts:
                    text = "\n".join(parts)

            # === 对话数据处理逻辑 ===
            # 如果包含 dialog 字段且开启了轮数限制
            if isinstance(dialog, list) and args.max_turns > 0:
                turns = len(dialog)
                # 过滤掉轮数过多的对话
                if turns > args.max_turns:
                    stats["too_many_turns"] += 1
                    continue
                
                # 如果原始 json 中没有 'text' 字段，尝试从 'dialog' 中构建文本
                if text is None:
                    # dialog 可能是字符串列表，也可能是包含 'content' 字段的字典列表
                    parts = []
                    for item in dialog:
                        if isinstance(item, dict):
                            parts.append(str(item.get("content", "")))
                        else:
                            parts.append(str(item))
                    # 将对话内容拼接，作为主要文本进行后续处理
                    text = "\n".join([p for p in parts if p.strip()])

            # 如果最终没有提取到文本，跳过
            if text is None:
                continue
            
            # === 文本清洗与长度过滤 ===
            text = normalize_text(str(text))
            
            # 清洗后为空
            if not text:
                stats["empty_after_clean"] += 1
                continue
            # 长度过短
            if len(text) < args.min_chars:
                stats["too_short"] += 1
                continue
            # 长度过长
            if len(text) > args.max_chars:
                stats["too_long"] += 1
                continue

            # === 精确去重 (MD5) ===
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if args.dedup and h in seen_hashes:
                stats["dedup"] += 1
                continue
            seen_hashes.add(h)

            # === 近似去重 (MinHash/Jaccard) ===
            # 注意：这里是 O(N) 的线性扫描对比，数据量非常大时速度会变慢
            if args.near_dedup:
                shingles = shingle_ngrams(text)
                is_dup = False
                for sh_set, _ in shingle_cache:
                    if jaccard(shingles, sh_set) >= args.jaccard_threshold:
                        is_dup = True
                        break
                if is_dup:
                    stats["near_dedup"] += 1
                    continue
                # 将当前样本特征加入缓存，供后续样本对比
                shingle_cache.append((shingles, idx))

            # === 情感分析过滤 ===
            if senti_pipe:
                try:
                    # 截取前 512 个字符进行预测，避免超过模型最大长度限制并提高速度
                    pred = senti_pipe(text[:512])[0]
                    label = pred.get("label", "")
                    # 如果预测结果不符合目标情感，则过滤
                    if not sentiment_keep(label, args.sentiment_filter):
                        stats["sentiment_filtered"] += 1
                        continue
                except Exception:
                    # 如果情感分析出错（如输入过长等边缘情况），默认保留样本，避免过度过滤
                    pass

            # === 写入结果 ===
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            if args.to_parquet:
                pq_buffer.append({"text": text})
                if len(pq_buffer) >= 1000:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    table = pa.Table.from_pylist(pq_buffer)
                    if pq_writer is None:
                        pq_writer = pq.ParquetWriter(str(pq_path), table.schema)
                    pq_writer.write_table(table)
                    pq_buffer = []

    # 保存统计日志
    log_path = outp.with_suffix(outp.suffix + ".log.json")
    log_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.to_parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if pq_buffer:
            table = pa.Table.from_pylist(pq_buffer)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(str(pq_path), table.schema)
            pq_writer.write_table(table)
        if pq_writer:
            pq_writer.close()
        print(f"[done] parquet saved to {pq_path}")
    
    print(f"[done] kept={stats['kept']} / total={stats['total']} -> {outp}")
    print(f"[log] stats -> {log_path}")


def main() -> None:
    args = parse_args()
    inp_path = Path(args.input)
    if inp_path.is_dir():
        files = sorted(inp_path.glob("*.jsonl"))
        if not files:
            raise SystemExit(f"No JSONL files found under {inp_path}")
        for f in files:
            print(f"[info] processing {f}")
            process_file(args, f)
    else:
        process_file(args, inp_path)


if __name__ == "__main__":
    main()
