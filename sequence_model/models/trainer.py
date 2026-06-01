import os, sys, json, time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    CHECKPOINT_DIR, LABEL_COLUMNS, THRESHOLD,
)
from utils.metrics import compute_metrics, classification_report_str


# Multilabel Binary cross Entropy
class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross-Entropy loss — reduces the contribution of easy
    (well-classified) examples so the model focuses on hard ones.

        FL(p_t) = -α · (1 - p_t)^γ · log(p_t)

    Falls back to standard BCE when gamma=0.
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt   = torch.exp(-bce)
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()



# Trainer
class Trainer:
    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        device:         torch.device,
        epochs:         int   = EPOCHS,
        lr:             float = LEARNING_RATE,
        weight_decay:   float = WEIGHT_DECAY,
        checkpoint_dir: str   = CHECKPOINT_DIR,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.epochs         = epochs
        self.checkpoint_dir = checkpoint_dir

        self.criterion = FocalBCELoss(gamma=2.0)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        self.history: Dict[str, List] = {
            "train_loss": [], "val_loss": [],
            "val_f1_micro": [], "val_f1_macro": [],
        }
        self.best_val_f1   = -1.0
        self.best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            ids    = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(ids)
            loss       = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self) -> Dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in self.val_loader:
            ids    = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, _ = self.model(ids)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        y_prob = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        metrics = compute_metrics(y_true, y_prob)
        metrics["val_loss"] = total_loss / len(self.val_loader)
        return metrics

    def train(self) -> Dict:
        print(f"\n{'─'*60}")
        print(f"  Training for {self.epochs} epochs on {self.device}")
        print(f"{'─'*60}")

        for epoch in range(1, self.epochs + 1):
            t0         = time.time()
            train_loss = self._train_epoch()
            val_metrics = self._validate()
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_f1_micro"].append(val_metrics["f1_micro"])
            self.history["val_f1_macro"].append(val_metrics["f1_macro"])

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:>3}/{self.epochs}  "
                f"TrainLoss={train_loss:.4f}  "
                f"ValLoss={val_metrics['val_loss']:.4f}  "
                f"F1-micro={val_metrics['f1_micro']:.4f}  "
                f"F1-macro={val_metrics['f1_macro']:.4f}  "
                f"[{elapsed:.1f}s]"
            )

            # Save best checkpoint
            if val_metrics["f1_micro"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1_micro"]
                torch.save(self.model.state_dict(), self.best_ckpt_path)
                print(f"  ✓ Best model saved  (F1-micro={self.best_val_f1:.4f})")

        # Persist training history
        history_path = os.path.join(self.checkpoint_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nHistory saved → {history_path}")
        return self.history

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> None:
        """Load best checkpoint and run final evaluation."""
        self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.device))
        self.model.eval()

        all_preds, all_labels = [], []
        for batch in test_loader:
            ids    = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits, _ = self.model(ids)
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        y_prob = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        metrics = compute_metrics(y_true, y_prob)

        print("\n" + "═"*60)
        print("  TEST SET RESULTS")
        print("═"*60)
        for k, v in metrics.items():
            print(f"  {k:<20}: {v:.4f}")
        print("\nPer-label Report:")
        print(classification_report_str(y_true, y_prob))
