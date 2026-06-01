import os, sys, random, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_FILE, CHECKPOINT_DIR, ATTENTION_DIR,
    EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_DIM,
    NUM_LABELS, DROPOUT, SEED, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, MAX_PLOT_SAMPLES,
)
from utils.tokenizer  import NepaliTokenizer
from utils.dataset    import load_data, build_dataloaders
from utils.visualize  import plot_batch_attention, plot_training_curves
from models.bilstm_attention import build_model
from models.trainer          import Trainer



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records   = load_data(DATA_FILE)
    texts     = [r["text"] for r in records]
    tokenizer = NepaliTokenizer()
    tokenizer.build_vocab(texts)

    vocab_save = os.path.join(CHECKPOINT_DIR, "vocab.json")
    tokenizer.save(vocab_save)

    train_loader, val_loader, test_loader = build_dataloaders(
        DATA_FILE, tokenizer,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        seed=SEED,
    )

    model = build_model(
        vocab_size    = tokenizer.vocab_size,
        embedding_dim = EMBEDDING_DIM,
        hidden_dim    = HIDDEN_DIM,
        attention_dim = ATTENTION_DIM,
        num_labels    = NUM_LABELS,
        dropout       = DROPOUT,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(model)

    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        epochs         = EPOCHS,
        lr             = LEARNING_RATE,
        weight_decay   = WEIGHT_DECAY,
        checkpoint_dir = CHECKPOINT_DIR,
    )
    history = trainer.train()

    trainer.evaluate(test_loader)

    plot_training_curves(history, save_dir=ATTENTION_DIR)

    limit = MAX_PLOT_SAMPLES  
    print(f"\n[Visualisation] Generating attention plots for up to {limit} sentences …")

    model.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINT_DIR, "best_model.pt"),
            map_location=device,
        )
    )
    model.eval()
    model.to(device)

    plotted = 0
    for batch in test_loader:
        if plotted >= limit:
            break

        ids    = batch["input_ids"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            logits, alpha = model(ids, return_attention=True)

        remaining = limit - plotted
        n = min(remaining, ids.size(0))

        plot_batch_attention(
            tokens_list = batch["tokens"][:n],
            attentions  = alpha.cpu().numpy()[:n],
            pred_probs  = logits.cpu().numpy()[:n],
            ids         = batch["id"][:n],
            threshold   = 0.5,
            save_dir    = ATTENTION_DIR,
            top_k       = 10,
        )
        plotted += n

    print(f"\nPlotted {plotted} sentence(s). All attention plots saved in: {ATTENTION_DIR}")
    print("Training complete ✓")


if __name__ == "__main__":
    main()
