#!/usr/bin/env python3
"""
RLHF 偏好数据清洗脚本
=====================

功能：
- 读取包含 prompt / chosen / rejected（或自定义字段名）的 JSONL
- 可选精确去重、近似去重（3-gram Jaccard）
- 长度过滤（按合并文本长度）
- 保留原始三列结构输出 JSONL，可选导出 Parquet

使用示例：
    uv run python data_process/process_preference.py \
      --input data_process/fetch_data/reward/shibing624_medical__train.jsonl \
      --output_dir data_process/final_data \
      --subdir reward \
      --prompt_key prompt --chosen_key response_chosen --rejected_key response_rejected \
      --dedup --near_dedup --jaccard_threshold 0.9 \
      --min_chars 5 --max_chars 2000 \
      --to_parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean RLHF preference data (prompt/chosen/rejected).")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", default="", help="Output JSONL path; if empty, auto-generate under --output_dir.")
    parser.add_argument("--output_dir", default="data_process/final_data", help="Base directory when --output is empty.")
    parser.add_argument("--subdir", default=None, help="Optional subdir under output_dir (e.g., reward).")
    parser.add_argument("--prompt_key", default="prompt", help="Field name for prompt.")
    parser.add_argument("--chosen_key", default="chosen", help="Field name for positive response.")
    parser.add_argument("--rejected_key", default="rejected", help="Field name for negative response.")
    parser.add_argument("--min_chars", type=int, default=5, help="Drop samples shorter than this (combined length).")
    parser.add_argument("--max_chars", type=int, default=4000, help="Drop samples longer than this (combined length).")
    parser.add_argument("--dedup", action="store_true", help="Enable exact deduplication (hash on prompt+chosen+rejected).")
    parser.add_argument("--near_dedup", action="store_true", help="Enable simple near-duplicate filtering (3-gram Jaccard).")
    parser.add_argument("--jaccard_threshold", type=float, default=0.9, help="Near-dup Jaccard threshold.")
    parser.add_argument("--to_parquet", action="store_true", help="Also write Parquet (requires pyarrow/pandas).")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """简洁的文本归一化：去首尾空白并压缩多余空格。"""
    return " ".join(text.strip().split())


def shingle_ngrams(text: str, n: int = 3) -> Set[str]:
    return {text[i : i + n] for i in range(max(len(text) - n + 1, 1))}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    base = Path(args.output_dir)
    if args.subdir:
        base = base / args.subdir
    base.mkdir(parents=True, exist_ok=True)
    name = Path(args.input).stem
    return base / f"{name}_clean.jsonl"


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"[错误] 输入文件不存在: {args.input}")

    out_path = build_output_path(args)
    dedup_hashes: Set[str] = set()
    near_cache: List[Tuple[str, Set[str]]] = []

    kept: List[Dict[str, str]] = []
    total = 0
    drop_len = 0
    drop_missing = 0
    drop_dup = 0
    drop_near = 0

    for rec in load_jsonl(in_path):
        total += 1
        p = str(rec.get(args.prompt_key, "") or "")
        c = str(rec.get(args.chosen_key, "") or "")
        r = str(rec.get(args.rejected_key, "") or "")

        if not (p and c and r):
            drop_missing += 1
            continue

        p, c, r = normalize_text(p), normalize_text(c), normalize_text(r)
        combo = f"{p}\n{c}\n{r}"
        if len(combo) < args.min_chars or len(combo) > args.max_chars:
            drop_len += 1
            continue

        if args.dedup:
            h = hashlib.md5(combo.encode("utf-8")).hexdigest()
            if h in dedup_hashes:
                drop_dup += 1
                continue
            dedup_hashes.add(h)

        if args.near_dedup:
            ngram = shingle_ngrams(combo, n=3)
            is_near = False
            for _, cached in near_cache:
                if jaccard(ngram, cached) >= args.jaccard_threshold:
                    is_near = True
                    break
            if is_near:
                drop_near += 1
                continue
            near_cache.append((combo, ngram))

        kept.append({args.prompt_key: p, args.chosen_key: c, args.rejected_key: r})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] kept={len(kept)} / total={total} -> {out_path}")

    # 写日志
    log = {
        "total": total,
        "kept": len(kept),
        "dropped_len": drop_len,
        "dropped_missing": drop_missing,
        "dropped_dup": drop_dup,
        "dropped_near_dup": drop_near,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "dedup": args.dedup,
        "near_dedup": args.near_dedup,
        "jaccard_threshold": args.jaccard_threshold,
        "prompt_key": args.prompt_key,
        "chosen_key": args.chosen_key,
        "rejected_key": args.rejected_key,
    }
    log_path = out_path.with_suffix(out_path.suffix + ".log.json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[log] stats -> {log_path}")

    if args.to_parquet:
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame(kept)
            pq_path = out_path.with_suffix(out_path.suffix + ".parquet")
            df.to_parquet(pq_path, index=False)
            print(f"[done] parquet saved to {pq_path}")
        except Exception as exc:
            print(f"[warn] parquet export failed: {exc}")


if __name__ == "__main__":
    main()
