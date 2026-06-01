import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Tuple
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LABEL_COLUMNS, MAX_SEQ_LEN, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, SEED
from utils.tokenizer import NepaliTokenizer


class NepaliBiasDataset(Dataset):
    """
    PyTorch Dataset for the Nepali multilabel bias corpus.

    Each item returns:
        input_ids  : LongTensor of shape (MAX_SEQ_LEN,)
        labels     : FloatTensor of shape (NUM_LABELS,)
        tokens     : raw token list for attention visualisation
        text       : original text string
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer: NepaliTokenizer,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec  = self.records[idx]
        text = rec["text"]

        input_ids = torch.tensor(
            self.tokenizer.encode(text, max_len=self.max_seq_len),
            dtype=torch.long,
        )
        labels = torch.tensor(
            [float(rec.get(col, 0)) for col in LABEL_COLUMNS],
            dtype=torch.float,
        )
        tokens = self.tokenizer.get_tokens(text)[: self.max_seq_len]

        return {
            "input_ids": input_ids,
            "labels":    labels,
            "tokens":    tokens,
            "text":      text,
            "id":        rec.get("id", ""),
        }



def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels":    torch.stack([b["labels"]    for b in batch]),
        "tokens":    [b["tokens"] for b in batch],
        "text":      [b["text"]   for b in batch],
        "id":        [b["id"]     for b in batch],
    }



def load_data(data_path: str) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    data_path: str,
    tokenizer: NepaliTokenizer,
    batch_size: int  = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT,
    val_split: float   = VAL_SPLIT,
    seed: int          = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Reads JSON, builds Dataset, splits into train/val/test DataLoaders.
    Returns (train_loader, val_loader, test_loader).
    """
    records = load_data(data_path)
    dataset = NepaliBiasDataset(records, tokenizer)

    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val   = int(n_total * val_split)
    n_test  = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    kwargs = dict(collate_fn=collate_fn, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kwargs)

    print(f"[Dataset] Total={n_total}  Train={n_train}  Val={n_val}  Test={n_test}")
    return train_loader, val_loader, test_loader
