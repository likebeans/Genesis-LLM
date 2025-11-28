#!/usr/bin/env python3
"""
Simple MLM finetune to give new tokens semantics after vocab extension.
Uses extended tokenizer/model; trains briefly on tokenizer_clean.txt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import datasets
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a short MLM finetune for extended vocab.")
    p.add_argument("--tokenizer", default="tokenizer/merged_model", help="Path to extended tokenizer.")
    p.add_argument("--model", default="tokenizer/merged_model", help="Path to base model (after resize).")
    p.add_argument("--data_file", default="tokenizer/data/tokenizer_data/tokenizer_clean.txt", help="Cleaned text corpus.")
    p.add_argument("--output_dir", default="tokenizer/merged_model_mlm", help="Where to save finetuned model+tokenizer.")
    p.add_argument("--max_length", type=int, default=256, help="Max sequence length.")
    p.add_argument("--mlm_probability", type=float, default=0.15, help="Masking probability.")
    p.add_argument("--batch_size", type=int, default=16, help="Per-device train batch size.")
    p.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    p.add_argument("--max_steps", type=int, default=1000, help="Training steps; set small for quick adaptation.")
    p.add_argument("--logging_steps", type=int, default=100, help="Logging interval.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    ds = datasets.load_dataset("text", data_files={"train": args.data_file})["train"]

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir="tokenizer/mlm_tmp",
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=0,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[done] saved finetuned model+tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
