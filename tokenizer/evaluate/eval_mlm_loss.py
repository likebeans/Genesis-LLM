#!/usr/bin/env python3
"""
Compute MLM loss on a small sample to sanity-check extended tokenizer/model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MLM loss on a small sample.")
    p.add_argument("--model", required=True, help="Model path/name (extended).")
    p.add_argument("--tokenizer", required=True, help="Tokenizer path/name (extended).")
    p.add_argument("--data_file", required=True, help="Text file for evaluation.")
    p.add_argument("--max_samples", type=int, default=2000, help="Max samples to evaluate.")
    p.add_argument("--max_length", type=int, default=256, help="Max sequence length.")
    p.add_argument("--batch_size", type=int, default=8, help="Per-device batch size.")
    p.add_argument("--report", default="tokenizer/logs/eval_report.md", help="Append summary to this report file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = datasets.load_dataset("text", data_files={"eval": args.data_file})["eval"]
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(len(ds), args.max_samples)))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    total_loss = 0.0
    total_count = 0

    for start in range(0, len(tokenized), args.batch_size):
        sub = [tokenized[i] for i in range(start, min(len(tokenized), start + args.batch_size))]
        batch = collator(sub)
        batch = {k: torch.tensor(v, device=device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss.item()
            total_loss += loss * len(sub)
            total_count += len(sub)

    avg_loss = total_loss / max(total_count, 1)
    print(f"[result] MLM avg loss on {total_count} samples: {avg_loss:.4f}")

    # Write summary to report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n\n## MLM Loss\n")
        f.write(f"- model: {args.model}\n")
        f.write(f"- tokenizer: {args.tokenizer}\n")
        f.write(f"- data: {args.data_file}\n")
        f.write(f"- samples: {total_count}\n")
        f.write(f"- max_length: {args.max_length}\n")
        f.write(f"- batch_size: {args.batch_size}\n")
        f.write(f"- avg_loss: {avg_loss:.4f}\n")


if __name__ == "__main__":
    main()
