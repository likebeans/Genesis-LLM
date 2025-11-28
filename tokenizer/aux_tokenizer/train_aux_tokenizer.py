#!/usr/bin/env python3
"""
Train auxiliary SentencePiece tokenizer (BPE/Unigram) for vocab extension.
Aligns with baseline settings to ease merge/compatibility.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train auxiliary SentencePiece tokenizer.")
    parser.add_argument(
        "--input_glob",
        default="tokenizer/data/tokenizer_data/tokenizer_clean.txt",
        help="Input text files glob (one sentence per line).",
    )
    parser.add_argument(
        "--output_dir",
        default="tokenizer/aux_tokenizer",
        help="Where to write tokenizer files and logs.",
    )
    parser.add_argument(
        "--prefix",
        default="chinese_spm_20000",
        help="Output prefix for model/vocab files.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=20000,
        help="Vocabulary size for auxiliary tokenizer (new tokens portion).",
    )
    parser.add_argument(
        "--model_type",
        choices=["bpe", "unigram"],
        default="bpe",
        help="SentencePiece model type.",
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=0.9995,
        help="Character coverage.",
    )
    parser.add_argument(
        "--split_digits",
        action="store_true",
        default=True,
        help="Split digits into separate tokens (align with LLaMA-style).",
    )
    parser.add_argument(
        "--byte_fallback",
        action="store_true",
        default=True,
        help="Enable byte fallback for rare chars.",
    )
    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=24000,
        help="Max sentence length in bytes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reserved for logging (SentencePiece trainer is deterministic with input_sentence_size=0).",
    )
    parser.add_argument(
        "--baseline_config",
        default="tokenizer/logs/baseline_tokenizer_config.json",
        help="Baseline tokenizer snapshot (used for logging/reference).",
    )
    return parser.parse_args()


def load_baseline(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_files = sorted(glob.glob(args.input_glob))
    if not text_files:
        raise SystemExit(f"No input files matched: {args.input_glob}")

    prefix = output_dir / args.prefix

    # Train SentencePiece
    t0 = time.perf_counter()
    spm.SentencePieceTrainer.train(
        input=text_files,
        model_prefix=str(prefix),
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        split_digits=args.split_digits,
        byte_fallback=args.byte_fallback,
        max_sentence_length=args.max_sentence_length,
        seed_sentencepiece_size=1000000,
        shuffle_input_sentence=True,
        input_sentence_size=0,
        self_test_sample_size=0,
        unk_surface=r" \342\201\207 ",
        allow_whitespace_only_pieces=True,
        num_threads=os.cpu_count(),
    )
    elapsed = time.perf_counter() - t0

    # Write log
    log = {
        "input_glob": args.input_glob,
        "input_files": text_files,
        "output_dir": str(output_dir),
        "model_prefix": str(prefix),
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "character_coverage": args.character_coverage,
        "split_digits": args.split_digits,
        "byte_fallback": args.byte_fallback,
        "max_sentence_length": args.max_sentence_length,
        "seed": args.seed,
        "baseline_config": args.baseline_config,
        "baseline_snapshot": load_baseline(args.baseline_config),
        "elapsed_seconds": round(elapsed, 2),
    }
    (output_dir / "train_spm.log").write_text(json.dumps(log, ensure_ascii=False, indent=2))

    print(f"[done] tokenizer model saved to {prefix}.model / {prefix}.vocab")
    print(f"[time] training elapsed: {elapsed:.2f}s")
    print(f"[log] params saved to {output_dir / 'train_spm.log'}")


if __name__ == "__main__":
    main()
