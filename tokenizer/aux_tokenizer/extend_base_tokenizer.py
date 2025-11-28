#!/usr/bin/env python3
"""
Step 5: Extend base tokenizer with candidate tokens.
- Reads base tokenizer path from config/tokenizer_config/config.yaml (base_tokenizer) unless overridden.
- Loads candidate tokens from new_tokens_list.txt (token \t freq).
- Adds tokens to base tokenizer and saves merged tokenizer to output_dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
from transformers import AutoTokenizer  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend base tokenizer with new tokens.")
    parser.add_argument(
        "--token_config",
        default="config/tokenizer_config/config.yaml",
        help="YAML config containing base_tokenizer path.",
    )
    parser.add_argument(
        "--baseline_config",
        default="tokenizer/logs/baseline_tokenizer_config.json",
        help="Baseline snapshot (fallback for base tokenizer).",
    )
    parser.add_argument(
        "--base_tokenizer",
        default=None,
        help="Optional base tokenizer path/name (overrides config).",
    )
    parser.add_argument(
        "--tokens_file",
        default="tokenizer/aux_tokenizer/new_tokens_list.txt",
        help="TSV file with candidate tokens (token\\tfreq).",
    )
    parser.add_argument(
        "--output_dir",
        default="tokenizer/merged_tokenizer",
        help="Where to save the extended tokenizer.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def save_config(path: str, cfg: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def load_baseline_tokenizer_path(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return data.get("source_tokenizer")
    except Exception:
        return None


def load_tokens(path: str) -> List[str]:
    toks: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            tok = parts[0]
            if tok:
                toks.append(tok)
    return toks


def main() -> None:
    args = parse_args()

    cfg = load_config(args.token_config)
    base_path = args.base_tokenizer or cfg.get("base_tokenizer") or load_baseline_tokenizer_path(args.baseline_config)
    if not base_path:
        raise SystemExit("Base tokenizer path not found. Set base_tokenizer in token_config or pass --base_tokenizer.")

    toks = load_tokens(args.tokens_file)
    if not toks:
        raise SystemExit(f"No tokens loaded from {args.tokens_file}")

    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=False)
    added = tok.add_tokens(toks)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out_dir)

    cfg["base_tokenizer"] = str(base_path)
    cfg["extended_tokenizer"] = str(out_dir)
    save_config(args.token_config, cfg)

    print(f"[done] added {added} tokens to base tokenizer -> saved to {out_dir}")
    print(f"[info] base tokenizer: {base_path}")
    print(f"[info] config updated: {args.token_config}")


if __name__ == "__main__":
    main()
