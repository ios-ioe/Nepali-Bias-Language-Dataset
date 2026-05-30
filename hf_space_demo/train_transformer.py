"""Fine-tune a transformer for multi-label bias classification (CPU-friendly defaults).

This script is adapted to run on CPU (or GPU if available). Default model is
`distilbert-base-multilingual-cased` which is lighter than XLM-R.

Usage example (from `hf_space_demo` root):
  python train_transformer.py \
    --model_name_or_path distilbert-base-multilingual-cased \
    --train_file ../data/train.json \
    --validation_file ../data/validation.json \
    --test_file ../data/test.json \
    --output_dir outputs/transformer_baseline \
    --epochs 3 \
    --per_device_train_batch_size 8 \
    --no_cuda

Notes for CPU training:
- Use small `per_device_train_batch_size` (4-16) and increase `gradient_accumulation_steps` if needed.
- Consider running fewer epochs or a smaller model for faster runs.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--train_file", type=str, default="../data/train.json")
    parser.add_argument("--validation_file", type=str, default="../data/validation.json")
    parser.add_argument("--test_file", type=str, default="../data/test.json")
    parser.add_argument("--output_dir", type=str, default="outputs/transformer_baseline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU training")
    return parser.parse_args()


def get_label_names(example_sample: dict) -> List[str]:
    reserved = {"text", "id", "comment"}
    return [k for k in example_sample.keys() if k not in reserved]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data_files = {
        "train": args.train_file,
        "validation": args.validation_file,
        "test": args.test_file,
    }
    raw = load_dataset("json", data_files=data_files)

    sample = raw["train"][0]
    label_names = get_label_names(sample)
    label_names = sorted(label_names)
    print("Label names:", label_names)

    def to_multi_label(example):
        example_labels = [int(example.get(k, 0)) for k in label_names]
        example["labels"] = example_labels
        return example

    raw = raw.map(to_multi_label)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_length)

    tokenized = raw.map(tokenize, batched=True)

    # remove text/id/comment columns to keep only model inputs + labels
    remove_cols = [c for c in tokenized["train"].column_names if c not in ["labels", "input_ids", "attention_mask", "token_type_ids"]]
    try:
        tokenized = tokenized.remove_columns(remove_cols)
    except Exception:
        # some tokenizers don't produce token_type_ids
        pass

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # save label names for inference
    with open(os.path.join(args.output_dir, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=False, indent=2)

    def compute_metrics(pred):
        from sklearn.metrics import f1_score

        logits = pred.predictions
        if logits.shape[-1] == 1:
            probs = logits.ravel()
            y_pred = (probs > 0.5).astype(int)
        else:
            probs = 1 / (1 + np.exp(-logits))
            y_pred = (probs >= 0.5).astype(int)

        y_true = pred.label_ids
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        exact_match = (y_true == y_pred).all(axis=1).mean()
        return {"f1_micro": f1_micro, "exact_match": float(exact_match)}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        seed=args.seed,
        no_cuda=args.no_cuda,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
