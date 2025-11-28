#!/usr/bin/env python3
"""
Compare base vs new tokenizer: sample texts, show tokenization, and report length stats.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

from transformers import AutoTokenizer  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare tokenization between base and new tokenizer.")
    p.add_argument("--base_tokenizer", required=True, help="Path/name of base tokenizer.")
    p.add_argument("--new_tokenizer", required=True, help="Path/name of new tokenizer.")
    p.add_argument("--input_file", required=True, help="Text file (one sample per line).")
    p.add_argument("--num_samples", type=int, default=20, help="Number of samples to display.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--report", default="tokenizer/logs/eval_report.md", help="Append summary to this report file.")
    return p.parse_args()


def load_lines(path: str, max_lines: int = 10000) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            if line.strip():
                lines.append(line.strip())
    return lines


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer, trust_remote_code=False)
    new_tok = AutoTokenizer.from_pretrained(args.new_tokenizer, trust_remote_code=False)

    lines = load_lines(args.input_file)
    if len(lines) == 0:
        raise SystemExit("No input lines found.")

    samples = random.sample(lines, k=min(args.num_samples, len(lines)))

    stats: List[Dict[str, int]] = []
    for idx, text in enumerate(samples, 1):
        base_tokens = base_tok.tokenize(text, add_special_tokens=False)
        new_tokens = new_tok.tokenize(text, add_special_tokens=False)
        stats.append({"base_len": len(base_tokens), "new_len": len(new_tokens)})

        print(f"\n=== Sample {idx} ===")
        print(f"Text: {text}")
        print(f"Base ({len(base_tokens)}): {base_tokens}")
        print(f"New  ({len(new_tokens)}): {new_tokens}")

    avg_base = sum(s["base_len"] for s in stats) / len(stats)
    avg_new = sum(s["new_len"] for s in stats) / len(stats)
    print("\n=== Summary ===")
    print(f"Samples: {len(stats)}")
    print(f"Avg base len: {avg_base:.2f}")
    print(f"Avg new  len: {avg_new:.2f}")
    print(f"Delta      : {avg_new - avg_base:+.2f}")

    # Write summary to report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n\n## Tokenizer Comparison\n")
        f.write(f"- base: {args.base_tokenizer}\n")
        f.write(f"- new : {args.new_tokenizer}\n")
        f.write(f"- samples: {len(stats)} from {args.input_file}\n")
        f.write(f"- avg_len_base: {avg_base:.2f}\n")
        f.write(f"- avg_len_new : {avg_new:.2f}\n")
        f.write(f"- delta_len   : {avg_new - avg_base:+.2f} (negative = new splits shorter)\n")


if __name__ == "__main__":
    main()
