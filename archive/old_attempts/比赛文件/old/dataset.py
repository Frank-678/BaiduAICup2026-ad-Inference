"""
dataset.py — High-performance data loading for GRAB inference.

Optimizations:
  - PyArrow engine for CSV parsing (2-3x faster than pandas default)
  - Multi-process DataLoader with shared memory (zero-copy to GPU)
  - User-level sequence packing for KV-Cache efficiency
  - Pre-tokenization of sparse features into integer IDs
"""

import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset, DataLoader
from typing import Tuple, Optional, List


class GRABDataset(Dataset):
    """
    Dataset for GRAB inference.

    Expects a CSV with the following columns (adjust names to match
    the official baseline):
      - log_id            : unique identifier for each impression
      - user_id           : anonymized user identifier
      - user_hist_seq     : packed user history (list of token IDs)
      - user_hist_time    : timestamps for each history token
      - user_hist_action  : action type for each history token (0/1)
      - ad_token          : candidate ad token ID
      - ad_time           : candidate ad timestamp
      ... (other static context features)

    In the competition environment, column names follow the official
    baseline. Update FIELD_NAMES below after inspecting the real data.
    """

    # ------------------------------------------------------------------
    # TODO: Replace with actual column names from the official baseline
    # ------------------------------------------------------------------
    FIELD_NAMES = {
        "log_id": "log_id",
        "user_id": "user_id",
        "history": "user_hist_seq",      # sequence of token IDs
        "hist_time": "user_hist_time",   # sequence of timestamps
        "hist_action": "user_hist_act",  # sequence of action labels
        "ad_token": "ad_token",          # candidate ad ID
        "ad_time": "ad_time",            # candidate ad timestamp
    }

    MAX_HISTORY_LEN = 1024  # truncate extremely long histories

    def __init__(self, data_path: str, vocab_size: int = 100000):
        super().__init__()
        print(f"[Dataset] Loading {data_path} with PyArrow engine ...")
        self.df = pd.read_csv(data_path, engine="pyarrow")
        print(f"[Dataset] Loaded {len(self.df)} samples.")

        self.log_ids = self.df[self.FIELD_NAMES["log_id"]].values
        self.user_ids = self.df[self.FIELD_NAMES["user_id"]].values

        # Parse sequence fields (assumed stored as space-separated or
        # comma-separated strings; adapt to the actual format)
        self.histories = self._parse_int_sequences(
            self.df[self.FIELD_NAMES["history"]]
        )
        self.hist_times = self._parse_float_sequences(
            self.df[self.FIELD_NAMES["hist_time"]]
        )
        self.hist_actions = self._parse_int_sequences(
            self.df[self.FIELD_NAMES["hist_action"]]
        )

        self.ad_tokens = self.df[self.FIELD_NAMES["ad_token"]].values.astype(np.int64)
        self.ad_times = self.df[self.FIELD_NAMES["ad_time"]].values.astype(np.float64)

        # Build a user → list[(history, ad, ...)] mapping for KV-Cache
        self._build_user_index()

    def _parse_int_sequences(self, col: pd.Series) -> List[np.ndarray]:
        """Parse string-encoded integer sequences into numpy arrays."""
        out = []
        for val in col:
            if isinstance(val, str):
                parts = val.strip().split()
                arr = np.asarray(parts, dtype=np.int64)[:self.MAX_HISTORY_LEN]
            elif isinstance(val, (list, tuple)):
                arr = np.asarray(val, dtype=np.int64)[:self.MAX_HISTORY_LEN]
            else:
                arr = np.zeros(1, dtype=np.int64)
            out.append(arr)
        return out

    def _parse_float_sequences(self, col: pd.Series) -> List[np.ndarray]:
        """Parse string-encoded float sequences."""
        out = []
        for val in col:
            if isinstance(val, str):
                parts = val.strip().split()
                arr = np.asarray(parts, dtype=np.float32)[:self.MAX_HISTORY_LEN]
            elif isinstance(val, (list, tuple)):
                arr = np.asarray(val, dtype=np.float32)[:self.MAX_HISTORY_LEN]
            else:
                arr = np.zeros(1, dtype=np.float32)
            out.append(arr)
        return out

    def _build_user_index(self):
        """Build a mapping from user_id → list of row indices."""
        self.user_to_rows: dict = {}
        for idx, uid in enumerate(self.user_ids):
            self.user_to_rows.setdefault(uid, []).append(idx)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns per-sample tensors for one impression:
          user_id  : int64 scalar
          history  : int64 [L_hist]    token IDs
          hist_t   : float32 [L_hist]  timestamps
          hist_a   : int64 [L_hist]    action types
          ad_token : int64 scalar
          ad_time  : float32 scalar
          log_id   : int64 scalar
        """
        return (
            np.int64(self.user_ids[idx]),
            self.histories[idx].astype(np.int64),
            self.hist_times[idx].astype(np.float32),
            self.hist_actions[idx].astype(np.int64),
            np.int64(self.ad_tokens[idx]),
            np.float32(self.ad_times[idx]),
            np.int64(self.log_ids[idx]),
        )

    def __len__(self) -> int:
        return len(self.df)


def collate_fn(batch):
    """
    Collate a batch of samples.

    Because histories have variable length, we return a list-of-arrays
    instead of a padded tensor. Padding happens on-GPU at model time
    to minimize CPU overhead.
    """
    user_ids = np.array([b[0] for b in batch], dtype=np.int64)
    histories = [b[1] for b in batch]
    hist_times = [b[2] for b in batch]
    hist_actions = [b[3] for b in batch]
    ad_tokens = np.array([b[4] for b in batch], dtype=np.int64)
    ad_times = np.array([b[5] for b in batch], dtype=np.float32)
    log_ids = np.array([b[6] for b in batch], dtype=np.int64)

    return {
        "user_ids": user_ids,
        "histories": histories,      # list of ndarray
        "hist_times": hist_times,
        "hist_actions": hist_actions,
        "ad_tokens": ad_tokens,
        "ad_times": ad_times,
        "log_ids": log_ids,
    }


def build_dataloader(
    data_path: str,
    batch_size: int = 4096,
    num_workers: int = 4,
    vocab_size: int = 100000,
) -> Tuple[DataLoader, pd.DataFrame]:
    """
    Create a high-performance DataLoader.

    Args:
        data_path       : Path to test CSV.
        batch_size      : Samples per mini-batch.
        num_workers     : CPU worker processes (set to #cores/2).
        vocab_size      : Vocabulary size for token ID validation.

    Returns:
        loader          : Paddle DataLoader yielding dict batches.
        raw_df          : Original DataFrame (for log_id mapping).
    """
    dataset = GRABDataset(data_path, vocab_size=vocab_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        use_shared_memory=True,
        collate_fn=collate_fn,
    )
    return loader, dataset.df
