import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_DIM,
    NUM_LABELS, DROPOUT, MAX_SEQ_LEN,
)

class BahdanauAttention(nn.Module):

    def __init__(self, encoder_dim: int, attention_dim: int):
        super().__init__()
        # Project hidden states
        self.W_h = nn.Linear(encoder_dim, attention_dim, bias=False)
        # Project the learnable query
        self.W_q = nn.Linear(attention_dim, attention_dim, bias=True)
        # Score projection → scalar
        self.v   = nn.Linear(attention_dim, 1, bias=False)
        # Learnable global query vector
        self.query = nn.Parameter(torch.randn(attention_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,         # (B, T, encoder_dim)
        mask: torch.Tensor | None = None,    # (B, T)  — True for padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        context    : (B, encoder_dim)   — attended representation
        alpha      : (B, T)             — attention weight distribution
        # B -> Batch
        # T -> Token length (sequence length)
        """
        B, T, _ = hidden_states.shape

        # Project hidden states
        # (B, T, attention_dim)
        h_proj = self.W_h(hidden_states)

        # Expand & project query
        # query: (attention_dim,) → (B, 1, attention_dim) → (B, T, attention_dim)
        q_proj = self.W_q(self.query)          # (attention_dim,)
        q_proj = q_proj.unsqueeze(0).unsqueeze(0)  # (1, 1, attention_dim)
        q_proj = q_proj.expand(B, T, -1)       # (B, T, attention_dim)

        # Additive score 
        # e: (B, T, 1) → squeeze → (B, T)
        energy = self.v(torch.tanh(h_proj + q_proj)).squeeze(-1)

        #Mask padding tokens (set to -inf before softmax)
        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))

        alpha = F.softmax(energy, dim=-1)      # (B, T)

        # Weighted sum of hidden states
        # (B, 1, T) × (B, T, encoder_dim) → (B, 1, encoder_dim) → (B, encoder_dim)
        context = torch.bmm(alpha.unsqueeze(1), hidden_states).squeeze(1)

        return context, alpha


class BiLSTMAttentionClassifier(nn.Module):

    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim:    int = HIDDEN_DIM,
        attention_dim: int = ATTENTION_DIM,
        num_labels:    int = NUM_LABELS,
        dropout:       float = DROPOUT,
        pad_idx:       int = 0,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.encoder_dim   = hidden_dim * 2   # BiLSTM concatenates both directions

        # Token Embeddings
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )

        # Bidirectional LSTM 
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Bahdanau Attention 
        self.attention = BahdanauAttention(self.encoder_dim, attention_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder_dim, self.encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(self.encoder_dim // 2, num_labels),
        )

        self._init_weights()

    # Weights initialization
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # LSTM forget-gate bias trick: set to 1
                n = param.shape[0]
                param.data[n // 4 : n // 2].fill_(1.0)


    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T)
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns
        logits  : (B, num_labels)   — raw sigmoid scores
        alpha   : (B, T) or None    — attention weights (if return_attention=True)
        """
        # Padding mask: True where token is PAD (index 0)
        pad_mask = (input_ids == 0)   # (B, T)
        x = self.embedding(input_ids)   # (B, T, embedding_dim)

        # hidden_states: (B, T, 2*hidden_dim)
        hidden_states, _ = self.lstm(x)
        context, alpha = self.attention(hidden_states, mask=pad_mask)
        logits = torch.sigmoid(self.classifier(context))   # (B, num_labels)

        if return_attention:
            return logits, alpha
        return logits, None



def build_model(vocab_size: int, **kwargs) -> BiLSTMAttentionClassifier:
    """Instantiate and return the model (moved to device inside trainer)."""
    return BiLSTMAttentionClassifier(vocab_size=vocab_size, **kwargs)
