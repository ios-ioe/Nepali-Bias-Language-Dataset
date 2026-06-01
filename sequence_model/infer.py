import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHECKPOINT_DIR, ATTENTION_DIR,
    EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_DIM,
    NUM_LABELS, DROPOUT, LABEL_COLUMNS, THRESHOLD, MAX_PLOT_SAMPLES,
)
from utils.tokenizer         import NepaliTokenizer
from utils.visualize         import plot_attention_heatmap, plot_attention_bar
from models.bilstm_attention import build_model


def load_model_and_tokenizer(device: torch.device):
    tokenizer = NepaliTokenizer.load(os.path.join(CHECKPOINT_DIR, "vocab.json"))
    model = build_model(
        vocab_size    = tokenizer.vocab_size,
        embedding_dim = EMBEDDING_DIM,
        hidden_dim    = HIDDEN_DIM,
        attention_dim = ATTENTION_DIM,
        num_labels    = NUM_LABELS,
        dropout       = DROPOUT,
    )
    ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().to(device)
    return model, tokenizer


def predict(
    texts:      list[str],
    model,
    tokenizer:  NepaliTokenizer,
    device:     torch.device,
    threshold:  float = THRESHOLD,
    visualize:  bool  = True,
    plot_limit: int   = MAX_PLOT_SAMPLES,   # ← hard cap; change in config.py
) -> list[dict]:
    """
    Run inference on a list of texts.

    Plots are only saved for the first `plot_limit` sentences.
    All sentences still get predictions and attention weights returned.
    """
    results  = []
    plotted  = 0

    for i, text in enumerate(texts):
        ids    = torch.tensor(
            [tokenizer.encode(text)], dtype=torch.long
        ).to(device)
        tokens = tokenizer.get_tokens(text)

        with torch.no_grad():
            logits, alpha = model(ids, return_attention=True)

        probs = logits.squeeze(0).cpu().numpy()   # (num_labels,)
        attn  = alpha.squeeze(0).cpu().numpy()    # (T,)
        attn  = attn[: len(tokens)]               # trim padding

        pred_labels = [LABEL_COLUMNS[j] for j, p in enumerate(probs) if p >= threshold]

        result = {
            "text":        text,
            "predictions": {LABEL_COLUMNS[j]: float(probs[j]) for j in range(NUM_LABELS)},
            "pred_labels": pred_labels,
            "attention":   attn.tolist(),
            "tokens":      tokens,
        }
        results.append(result)

        # Console summary (always)
        print(f"\n[{i+1}/{len(texts)}] {text}")
        print(f"  Labels : {pred_labels or ['none']}")
        tok_attn = sorted(zip(tokens, attn.tolist()), key=lambda x: -x[1])
        print("  Top attention tokens:")
        for tok, w in tok_attn[:5]:
            print(f"    {tok:<20} {w:.4f}")

        # Plot only up to the cap
        if visualize and plotted < plot_limit:
            sample_id = f"infer_{i:03d}"
            plot_attention_heatmap(tokens, attn, pred_labels, sample_id, ATTENTION_DIR)
            plot_attention_bar(tokens, attn, pred_labels, sample_id, ATTENTION_DIR)
            plotted += 1
            print(f"  ✓ Plots saved ({plotted}/{plot_limit})")
        elif visualize and plotted >= plot_limit:
            # Only print this message once
            if plotted == plot_limit:
                print(f"\n  [Visualisation] Plot cap of {plot_limit} reached — "
                      "remaining sentences skipped. Adjust MAX_PLOT_SAMPLES in config.py.")
            plotted += 1   # increment so the message doesn't repeat

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(device)

    test_texts = [
        "एमालेले मधेस दमन गर्‍यो।",
        "महिलाहरू नेतृत्व गर्न सक्दैनन्।",
        "आजको मौसम राम्रो छ।",
    ]

    predict(test_texts, model, tokenizer, device, visualize=True)
