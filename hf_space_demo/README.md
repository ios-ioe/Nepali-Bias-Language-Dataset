---
license: cc-by-4.0
sdk: gradio
title: Nepali Bias Detection Demo
emoji: ⚖️
colorFrom: blue
colorTo: green
---

# Nepali Bias Detection Demo

This Hugging Face Space is intended to showcase the Nepali Bias Language Dataset.
It is a lightweight demo Space built to illustrate the dataset's multi-label bias categories.

## What is included
- `app.py`: Gradio demo for bias prediction
- `requirements.txt`: Python dependencies
- `assets/sample_examples.json`: Small set of sample Nepali texts

## Notes
- Keep this Space as a demo or showcase, not a production bias detector.
- The model should reflect the multi-label nature of the dataset.

## Algorithms and training

- Transformer baseline: `train_transformer.py` fine-tunes a multilingual transformer for multi-label classification. The default model is `distilbert-base-multilingual-cased` (CPU-friendly). The script can also use other models by passing `--model_name_or_path`.
- Problem type: multi-label classification (sigmoid activation per label + binary cross-entropy). The `Trainer` is configured with `problem_type="multi_label_classification"` and evaluation uses micro-F1 and exact-match metrics.
- Tokenization: standard Hugging Face tokenizer with truncation to `--max_length` tokens and padding handled by a data collator.
- CPU considerations: defaults favor smaller models, shorter sequence lengths, small batch sizes, and optional `--no_cuda` to force CPU-only training.

If you plan to run training on the Space, ensure the environment has sufficient CPU/RAM and long-running job support; otherwise run training offline (GPU) and upload the produced `outputs/transformer_baseline` directory to the Space repository.
