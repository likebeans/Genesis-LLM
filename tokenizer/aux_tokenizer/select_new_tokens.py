#!/usr/bin/env python3
"""
Step 4: Select candidate new tokens by comparing base vs auxiliary tokenizers.
Outputs a frequency-sorted new_tokens_list.txt (tokens that are single/shorter in aux but fragmented/UNK in base).
"""
from __future__ import annotations

import argparse
import json
import os
import yaml
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Any, Dict, Callable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select candidate new tokens via base vs aux tokenizer comparison.")
    parser.add_argument("--baseline_config", default="tokenizer/logs/baseline_tokenizer_config.json", help="Baseline tokenizer snapshot JSON.")
    parser.add_argument("--aux_tokenizer", required=True, help="Path/name of auxiliary tokenizer (SentencePiece/HF).")
    parser.add_argument("--aux_sp_model", default=None, help="Optional explicit path to auxiliary SentencePiece model (.model).")
    parser.add_argument("--input_glob", required=True, help="Glob for text files (one sample per line).")
    parser.add_argument("--output", default="tokenizer/aux_tokenizer/new_tokens_list.txt", help="Where to write candidate tokens.")
    parser.add_argument("--max_samples", type=int, default=100000, help="Max lines to sample for analysis.")
    parser.add_argument("--min_freq", type=int, default=5, help="Minimum frequency to keep.")
    parser.add_argument("--max_base_pieces", type=int, default=3, help="Keep tokens where base splits into > max_base_pieces.")
    parser.add_argument("--token_config", default="config/tokenizer_config/config.yaml", help="Tokenizer config YAML (stores base_tokenizer).")
    parser.add_argument("--base_tokenizer", default=None, help="Optional base tokenizer; if set, will write to token_config.")
    return parser.parse_args()


def load_base_tokenizer(path: str):
    from transformers import AutoTokenizer  # type: ignore
    return AutoTokenizer.from_pretrained(path, trust_remote_code=False)


def load_aux_tokenizer(path: str, sp_model: Optional[str]):
    # Try HF first
    from transformers import AutoTokenizer  # type: ignore
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=False), True
    except Exception:
        # fall back to sentencepiece model
        import sentencepiece as spm  # type: ignore
        model_path = sp_model
        if model_path is None:
            p = Path(path)
            if p.is_dir():
                candidates = list(p.glob("*.model"))
                if not candidates:
                    raise SystemExit("No .model found for auxiliary tokenizer and HF load failed.")
                model_path = str(candidates[0])
            else:
                model_path = str(p)
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))

        class AuxWrapper:
            def tokenize(self, text: str, add_special_tokens: bool = False):
                return sp.encode(text, out_type=str)

        return AuxWrapper(), False


def load_baseline_path(baseline_cfg: str) -> str:
    p = Path(baseline_cfg)
    if not p.exists():
        raise SystemExit(f"baseline config not found: {baseline_cfg}")
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        raise SystemExit(f"failed to read baseline config: {exc}")
    base_path = data.get("source_tokenizer")
    if not base_path:
        raise SystemExit("baseline config missing 'source_tokenizer'")
    return str(base_path)


def load_token_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def save_token_config(path: str, cfg: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def iter_lines(glob_pattern: str, max_samples: int):
    import glob

    count = 0
    for fname in sorted(glob.glob(glob_pattern)):
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples and count >= max_samples:
                    return
                yield line.strip()
                count += 1


def select_tokens(
    base_tok,
    aux_tok,
    texts,
    max_base_pieces: int,
) -> Counter:
    freq = Counter()
    for txt in texts:
        if not txt:
            continue
        base_tokens = base_tok.tokenize(txt, add_special_tokens=False)
        aux_tokens = aux_tok.tokenize(txt, add_special_tokens=False)
        # identify spans where aux is shorter than base split
        # heuristic: tokens that appear as single piece in aux but base made many pieces/UNK
        # collect aux tokens if base split count > max_base_pieces or contains unk
        if len(base_tokens) > max_base_pieces or any(t == base_tok.unk_token for t in base_tokens):
            for t in aux_tokens:
                freq[t] += 1
    return freq


def main() -> None:
    args = parse_args()
    cfg = load_token_config(args.token_config)
    base_path = args.base_tokenizer or cfg.get("base_tokenizer")
    if base_path:
        cfg["base_tokenizer"] = base_path
        save_token_config(args.token_config, cfg)
    else:
        base_path = load_baseline_path(args.baseline_config)
        cfg["base_tokenizer"] = base_path
        save_token_config(args.token_config, cfg)

    base_tok = load_base_tokenizer(base_path)
    aux_tok, aux_is_hf = load_aux_tokenizer(args.aux_tokenizer, args.aux_sp_model)

    texts = iter_lines(args.input_glob, args.max_samples)
    freq = select_tokens(base_tok, aux_tok, texts, args.max_base_pieces)

    candidates: List[Tuple[str, int]] = [
        (tok, c) for tok, c in freq.items() if c >= args.min_freq and tok not in base_tok.get_vocab()
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for tok, c in candidates:
            f.write(f"{tok}\t{c}\n")

    print(f"[done] wrote {len(candidates)} candidates to {out_path}")
    print(f"[info] base tokenizer: {base_path}")
    print(f"[info] aux tokenizer: {args.aux_tokenizer} ({'hf' if aux_is_hf else 'sentencepiece'})")


if __name__ == "__main__":
    main()
