#!/usr/bin/env python3
"""
Step 6: Resize model embeddings to match extended tokenizer and save.
Uses base_model (defaults to base_tokenizer) and extended_tokenizer from config/tokenizer_config/config.yaml unless overridden.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import yaml
from transformers import AutoTokenizer, AutoModelForMaskedLM  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize model embeddings for extended tokenizer.")
    parser.add_argument("--token_config", default="config/tokenizer_config/config.yaml", help="Tokenizer config YAML.")
    parser.add_argument("--base_model", default=None, help="Base model path/name (overrides config base_model/base_tokenizer).")
    parser.add_argument("--extended_tokenizer", default=None, help="Extended tokenizer path (overrides config extended_tokenizer).")
    parser.add_argument("--output_dir", default="tokenizer/merged_model", help="Where to save the resized model.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def save_config(path: str, cfg: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.token_config)

    base_model = args.base_model or cfg.get("base_model") or cfg.get("base_tokenizer")
    if not base_model:
        raise SystemExit("Base model not specified. Set base_model/base_tokenizer in config or pass --base_model.")
    extended_tokenizer = args.extended_tokenizer or cfg.get("extended_tokenizer")
    if not extended_tokenizer:
        raise SystemExit("Extended tokenizer not specified. Run extend_base_tokenizer first or pass --extended_tokenizer.")

    tok = AutoTokenizer.from_pretrained(extended_tokenizer, trust_remote_code=False)
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tok))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)  # save tokenizer alongside for convenience

    cfg["base_model"] = str(base_model)
    cfg["extended_tokenizer"] = str(extended_tokenizer)
    cfg["extended_model"] = str(out_dir)
    save_config(args.token_config, cfg)

    print(f"[done] resized embeddings to {len(tok)} tokens and saved model to {out_dir}")
    print(f"[info] base_model: {base_model}")
    print(f"[info] extended_tokenizer: {extended_tokenizer}")
    print(f"[info] config updated: {args.token_config}")


if __name__ == "__main__":
    main()
