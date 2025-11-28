#!/usr/bin/env python3
"""
Fetch and clean Chinese Wikipedia (or local) text into line-level corpus
for tokenizer training. Designed for reproducibility and light dependencies.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional


def _default_config_path() -> str:
    # repo root assumed two levels up: tokenizer/data/ -> tokenizer/ -> repo
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "config/model_config/config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch + clean text for tokenizer training."
    )
    parser.add_argument(
        "--output_dir",
        default="tokenizer/data/tokenizer_data",
        help="Where to write tokenizer_clean.txt and data_prep.log",
    )
    parser.add_argument(
        "--source",
        choices=["hf_wikipedia", "local"],
        default="hf_wikipedia",
        help="Data source: HuggingFace Wikipedia or local files",
    )
    parser.add_argument(
        "--dataset_name",
        default="wikimedia/wikipedia",
        help="Dataset name for hf_wikipedia source (default: wikimedia/wikipedia, parquet-based)",
    )
    parser.add_argument(
        "--hf_config",
        default="20231101.zh",
        help="Wikipedia config when source=hf_wikipedia (e.g., 20231101.zh)",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=0,
        help="Optional cap for number of records (0 means full dataset)",
    )
    parser.add_argument(
        "--input_glob",
        default="",
        help="Glob for local files when source=local (e.g., 'raw/*.txt')",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=10,
        help="Drop lines shorter than this after cleaning",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=2000,
        help="Drop lines longer than this after cleaning",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=0,
        help="Optional cap for output lines (0 means no cap)",
    )
    parser.add_argument(
        "--config",
        default=_default_config_path(),
        help="Path to config.yaml to read HF mirror (HF_ENDPOINT)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Set trust_remote_code=True when loading HF datasets (usually False for wikimedia/wikipedia parquet)",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar during cleaning/dedup",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Lightweight cleaning: remove HTML/Markdown, control chars, collapse spaces."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove Markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    # Replace control characters (except whitespace)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\t\n\r ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wrap_progress(it: Iterable[str], enabled: bool, desc: str) -> Iterable[str]:
    if not enabled:
        return it
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return it
    return tqdm(it, desc=desc)


def apply_hf_mirror(config_path: str) -> Optional[str]:
    """Load HF mirror from config.yaml and set HF_ENDPOINT. Best-effort; warns on failure."""
    path = Path(config_path)
    if not path.exists():
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        print(f"[warn] PyYAML not installed; skip config {config_path}", file=sys.stderr)
        return None
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] failed to read config {config_path}: {exc}", file=sys.stderr)
        return None

    hf_cfg = cfg.get("huggingface") or cfg.get("hf") or {}
    endpoint = hf_cfg.get("endpoint") or hf_cfg.get("mirror")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        return str(endpoint)
    return None


def iter_hf_wikipedia(
    dataset_name: str,
    config: str,
    sample_limit: int,
    trust_remote_code: bool,
) -> Iterator[str]:
    try:
        from datasets import load_dataset  # lazy import to keep dependency optional
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "datasets is required for hf_wikipedia source. Install via `pip install datasets`."
        ) from exc

    # parquet-based dataset; trust_remote_code typically not needed
    ds = load_dataset(dataset_name, config, split="train", trust_remote_code=trust_remote_code)
    if sample_limit and sample_limit > 0:
        ds = ds.select(range(sample_limit))
    for rec in ds:
        text = rec.get("text") or ""
        # Split by newline to avoid extremely long segments
        for line in text.split("\n"):
            if line.strip():
                yield line


def iter_local_files(pattern: str) -> Iterator[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched input_glob={pattern!r}")
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line


def dedup_and_filter(
    lines: Iterable[str],
    min_chars: int,
    max_chars: int,
    max_lines: int,
    show_progress: bool,
) -> tuple[list[str], Counter]:
    seen = set()
    stats = Counter()
    output: list[str] = []

    for raw in wrap_progress(lines, enabled=show_progress, desc="clean/dedup"):
        stats["total_lines"] += 1
        norm = normalize_text(raw)
        if not norm:
            stats["empty_after_clean"] += 1
            continue
        if len(norm) < min_chars:
            stats["too_short"] += 1
            continue
        if len(norm) > max_chars:
            stats["too_long"] += 1
            continue
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h in seen:
            stats["duplicates"] += 1
            continue
        seen.add(h)
        output.append(norm)
        stats["kept"] += 1
        if max_lines and stats["kept"] >= max_lines:
            break
    return output, stats


def save_corpus(lines: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "tokenizer_clean.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def save_log(stats: Counter, args: argparse.Namespace, output_dir: Path) -> None:
    log = {
        "source": args.source,
        "hf_config": args.hf_config,
        "input_glob": args.input_glob,
        "sample_limit": args.sample_limit,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "max_lines": args.max_lines,
        "hf_endpoint": getattr(args, "hf_endpoint", None),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
    }
    log_path = output_dir / "data_prep.log"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    hf_endpoint = apply_hf_mirror(args.config)
    args.hf_endpoint = hf_endpoint  # stash for logging
    if hf_endpoint:
        print(f"[info] HF_ENDPOINT set from config: {hf_endpoint}", file=sys.stderr)

    if args.source == "hf_wikipedia":
        lines_iter = iter_hf_wikipedia(
            args.dataset_name,
            args.hf_config,
            args.sample_limit,
            args.trust_remote_code,
        )
    else:
        if not args.input_glob:
            raise SystemExit("source=local requires --input_glob")
        lines_iter = iter_local_files(args.input_glob)

    cleaned, stats = dedup_and_filter(
        lines_iter,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_lines=args.max_lines,
        show_progress=not args.no_progress,
    )

    save_corpus(cleaned, output_dir)
    save_log(stats, args, output_dir)

    print(f"[done] kept={stats['kept']} lines, wrote to {output_dir / 'baike_clean.txt'}")
    print(f"[log] stats saved to {output_dir / 'data_prep.log'}")


if __name__ == "__main__":  # pragma: no cover
    main()
