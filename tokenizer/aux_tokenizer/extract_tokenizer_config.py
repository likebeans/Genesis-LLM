#!/usr/bin/env python3
"""
Extract baseline tokenizer configuration for compatibility when extending vocab.
Saves a JSON snapshot with key fields to align auxiliary tokenizer training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple


def _default_config_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "config/model_config/config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract baseline tokenizer config.")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or path (e.g., meta-llama/Llama-2-7b-hf or local dir).",
    )
    parser.add_argument(
        "--output",
        default="tokenizer/logs/baseline_tokenizer_config.json",
        help="Where to write the JSON snapshot.",
    )
    parser.add_argument(
        "--config",
        default=_default_config_path(),
        help="Path to YAML config for HF mirror (huggingface.endpoint).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code when loading tokenizer (use only if required).",
    )
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="Load tokenizer from local files only (no network). Fails if files not present locally.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Run `huggingface-cli login` before loading remote tokenizer (for gated/private repos).",
    )
    return parser.parse_args()


def apply_hf_mirror(config_path: str) -> str | None:
    """Load HF mirror from YAML and set HF_ENDPOINT. Best-effort."""
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
    except Exception as exc:  # pragma: no cover
        print(f"[warn] failed to read config {config_path}: {exc}", file=sys.stderr)
        return None
    hf_cfg = cfg.get("huggingface") or cfg.get("hf") or {}
    endpoint = hf_cfg.get("endpoint") or hf_cfg.get("mirror")
    if endpoint:
        os.environ["HF_ENDPOINT"] = str(endpoint)
        return str(endpoint)
    return None


def ensure_hf_login(do_login: bool) -> None:
    if not do_login:
        return
    try:
        subprocess.run(["hf", "auth", "login"], check=True)
    except FileNotFoundError:
        print("[error] hf CLI not found. Install `huggingface_hub` (>=0.25) or ensure it is in PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[error] hf auth login failed: {exc}", file=sys.stderr)
        sys.exit(1)


def _special_tokens(tok) -> Tuple[Dict[str, Any], Dict[str, int], Any]:
    token_map = {}
    token_ids = {}
    for name, val in tok.special_tokens_map.items():
        token_map[name] = str(val)
        token_ids[name] = tok.convert_tokens_to_ids(val)
    add_tokens = [str(t) for t in tok.additional_special_tokens]
    return token_map, token_ids, add_tokens


def _safe_init_kwargs(tok) -> Dict[str, Any]:
    """Filter init_kwargs to JSON-serializable primitives."""
    out = {}
    for k, v in getattr(tok, "init_kwargs", {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(
            isinstance(x, (str, int, float, bool)) or x is None for x in v
        ):
            out[k] = list(v)
    return out


def extract_summary(tok, source: str, config_path: str, hf_endpoint: str | None, trust_remote_code: bool) -> Dict[str, Any]:
    token_map, token_ids, add_tokens = _special_tokens(tok)
    summary: Dict[str, Any] = {
        "source_tokenizer": source,
        "hf_endpoint": hf_endpoint,
        "config_path": config_path,
        "trust_remote_code": trust_remote_code,
        "tokenizer_class": tok.__class__.__name__,
        "is_fast": tok.is_fast,
        "vocab_size": tok.vocab_size,
        "len_tokenizer": len(tok),
        "model_max_length": tok.model_max_length,
        "padding_side": tok.padding_side,
        "truncation_side": tok.truncation_side,
        "clean_up_tokenization_spaces": getattr(tok, "clean_up_tokenization_spaces", None),
        "special_tokens": token_map,
        "special_token_ids": token_ids,
        "additional_special_tokens": add_tokens,
        "added_vocab_size": len(tok.get_added_vocab()),
        "sp_model_kwargs": getattr(tok, "sp_model_kwargs", None),
        "init_kwargs": _safe_init_kwargs(tok),
    }
    return summary


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional login for gated/private repos
    ensure_hf_login(args.login)

    hf_endpoint = apply_hf_mirror(args.config)
    if hf_endpoint:
        print(f"[info] HF_ENDPOINT set from config: {hf_endpoint}", file=sys.stderr)

    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("transformers is required. Install via `uv add transformers sentencepiece`.") from exc

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_only,
    )
    summary = extract_summary(tok, args.tokenizer, args.config, hf_endpoint, args.trust_remote_code)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] baseline tokenizer config saved to {output_path}")
    print(f"[hint] key fields: vocab_size={summary['vocab_size']}, total_tokens={summary['len_tokenizer']}, special={summary['special_tokens']}")


if __name__ == "__main__":  # pragma: no cover
    main()
