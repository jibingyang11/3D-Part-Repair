"""Dataset index management for loading and splitting samples."""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional


class DatasetIndex:
    """Manage dataset index for train/val/test splitting."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.index_path = self.data_dir / "dataset_index.json"
        self.entries = []
        if self.index_path.exists():
            self.load()

    def load(self):
        """Load index from file."""
        with open(self.index_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

    def save(self):
        """Save index to file."""
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)

    @property
    def sample_ids(self) -> List[str]:
        return [e["sample_id"] for e in self.entries]

    def __len__(self):
        return len(self.entries)

    def get_sample_dir(self, sample_id: str) -> str:
        """Get directory path for a sample."""
        for e in self.entries:
            if e["sample_id"] == sample_id:
                return e["dir"]
        # Fallback: construct from data_dir
        return str(self.data_dir / sample_id)

    def split(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
              seed: int = 42) -> Dict[str, List[str]]:
        """Split samples into train/val/test sets."""
        random.seed(seed)
        ids = self.sample_ids.copy()
        random.shuffle(ids)

        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": ids[:n_train],
            "val": ids[n_train:n_train + n_val],
            "test": ids[n_train + n_val:],
        }
        return splits

    def save_splits(self, splits_dir: str, **kwargs):
        """Save train/val/test splits to files."""
        splits = self.split(**kwargs)
        os.makedirs(splits_dir, exist_ok=True)
        for split_name, ids in splits.items():
            path = os.path.join(splits_dir, f"{split_name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ids, f, indent=2)
        return splits
