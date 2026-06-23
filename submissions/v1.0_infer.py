from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - judge environment should provide torch
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


LOGID_KEYS = ("logid", "log_id", "id", "sample_id")
USER_KEYS = ("user_id", "uid", "userid")
SEQUENCE_KEYS = ("user_seq", "history", "hist", "histories", "seq", "features", "feasigns")
ITEM_KEYS = ("item_id", "ad_id", "creative_id", "cid", "aid")
MODEL_KEYS = ("model", "net", "module")
STATE_DICT_KEYS = ("model_state_dict", "state_dict", "net_state_dict")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(f"PyTorch is required by infer.py: {_TORCH_IMPORT_ERROR}")


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            return abs(hash(value)) % 2_147_483_647
    return abs(hash(repr(value))) % 2_147_483_647


def _tokenize_scalar(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{" and text[-1] in "]}":
            try:
                return _flatten_ints(json.loads(text))
            except Exception:
                pass
        for sep in (",", "|", ";"):
            text = text.replace(sep, " ")
        return [_as_int(part) for part in text.split() if part]
    return [_as_int(value)]


def _flatten_ints(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        out: List[int] = []
        for key in sorted(value):
            out.extend(_flatten_ints(value[key]))
        return out
    if isinstance(value, np.ndarray):
        return [_as_int(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple, set)):
        out: List[int] = []
        for item in value:
            out.extend(_flatten_ints(item))
        return out
    return _tokenize_scalar(value)


def _first_key(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    lowered = {str(k).lower(): k for k in record}
    for key in keys:
        if key in lowered:
            return record[lowered[key]]
    return None


def _records_from_object(obj: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        rows = obj.get("data") or obj.get("records") or obj.get("samples")
        if rows is not None:
            yield from _records_from_object(rows)
            return
        if all(hasattr(v, "__len__") and not isinstance(v, (str, bytes, Mapping)) for v in obj.values()):
            keys = list(obj.keys())
            try:
                n = len(obj[keys[0]])
            except Exception:
                yield obj
                return
            for i in range(n):
                yield {k: obj[k][i] for k in keys}
        else:
            yield obj
    elif isinstance(obj, np.ndarray):
        if obj.dtype.names:
            for row in obj:
                yield {name: row[name] for name in obj.dtype.names}
        else:
            for row in obj.tolist():
                if isinstance(row, Mapping):
                    yield row
    elif isinstance(obj, (list, tuple)):
        for row in obj:
            if isinstance(row, Mapping):
                yield row


def _iter_records(path: Path) -> Iterable[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jl"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, Mapping):
                        yield obj
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            yield from _records_from_object(json.load(f))
        return
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f, delimiter=delimiter)
        return
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            yield from _records_from_object(pickle.load(f))
        return
    if suffix == ".npz":
        obj = np.load(path, allow_pickle=True)
        keys = list(obj.keys())
        if keys:
            for i in range(len(obj[keys[0]])):
                yield {k: obj[k][i].item() if getattr(obj[k][i], "shape", None) == () else obj[k][i] for k in keys}
        return
    if suffix == ".npy":
        yield from _records_from_object(np.load(path, allow_pickle=True))
        return
    if suffix == ".pt" and torch is not None:
        yield from _records_from_object(torch.load(path, map_location="cpu"))
        return
    if suffix == ".parquet":
        try:
            import pandas as pd
        except Exception:
            return
        for row in pd.read_parquet(path).to_dict("records"):
            yield row


def _features_from_record(record: Mapping[str, Any], max_ctx_len: int | None = None) -> List[int]:
    features: List[int] = []
    for key in SEQUENCE_KEYS:
        value = _first_key(record, (key,))
        if value is not None:
            features.extend(_flatten_ints(value))
    item = _first_key(record, ITEM_KEYS)
    if item is not None:
        features.extend(_flatten_ints(item))
    if not features:
        ignored = set(LOGID_KEYS) | set(USER_KEYS)
        for key, value in record.items():
            if str(key).lower() not in ignored:
                features.extend(_flatten_ints(value))
    features = [abs(int(x)) for x in features if int(x) >= 0]
    if max_ctx_len is not None and max_ctx_len > 0:
        features = features[-max_ctx_len:]
    return features or [0]


def _load_cached_batches_from_files(files: Sequence[Path]) -> List[Mapping[str, Any]]:
    _require_torch()
    batches: List[Mapping[str, Any]] = []
    for file_path in sorted(Path(p) for p in files):
        obj = torch.load(file_path, map_location="cpu", weights_only=False)
        if isinstance(obj, Mapping):
            if {"logid", "pred_mask"}.issubset(obj.keys()):
                batches.append(obj)
            else:
                nested = obj.get("batches") or obj.get("data")
                if isinstance(nested, (list, tuple)):
                    batches.extend(x for x in nested if isinstance(x, Mapping))
        elif isinstance(obj, (list, tuple)):
            batches.extend(x for x in obj if isinstance(x, Mapping))
    return batches


def _looks_like_cached_batch_file(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".pt" and ("shard" in name or "batch" in name or "cached" in str(path).lower())


def _detect_has_clk(file_path: Path) -> bool:
    with Path(file_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 5:
                return ":" not in parts[4]
            return False
    return False


def _load_official_csv_sample_files(sample_files: Sequence[Path]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[int]]]:
    item_dict: Dict[int, Dict[str, Any]] = {}
    user_logs: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for sample_file in sorted(Path(f) for f in sample_files):
        has_clk = _detect_has_clk(sample_file)
        min_parts = 5 if has_clk else 4
        with sample_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < min_parts:
                    continue
                try:
                    logid = int(parts[0])
                    userid = int(parts[1])
                    adid = int(parts[2])
                    if has_clk:
                        clk = int(parts[3])
                        timestamp = int(parts[4])
                        feat_start = 5
                    else:
                        clk = 0
                        timestamp = int(parts[3])
                        feat_start = 4
                except ValueError:
                    continue
                signs = []
                slots = []
                for pair in parts[feat_start:]:
                    if ":" not in pair:
                        continue
                    sign, slot = pair.split(":", 1)
                    try:
                        signs.append(int(sign))
                        slots.append(int(slot))
                    except ValueError:
                        continue
                item_dict[logid] = {
                    "logid": logid,
                    "userid": userid,
                    "adid": adid,
                    "clk": clk,
                    "timestamp": timestamp,
                    "signs": np.asarray(signs, dtype=np.int64),
                    "slots": np.asarray(slots, dtype=np.int64),
                }
                user_logs.setdefault(userid, []).append((timestamp, logid))
    user_seq: Dict[int, List[int]] = {}
    for userid, logs in user_logs.items():
        logs.sort(key=lambda x: x[0])
        user_seq[userid] = [logid for _, logid in logs]
    return item_dict, user_seq


def load_sample_files(sample_files_list: List[Path]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[int]]]:
    paths: List[Path] = []
    for raw_path in sample_files_list:
        path = Path(raw_path)
        if path.is_dir():
            paths.extend(p for p in sorted(path.rglob("*.pt")) if _looks_like_cached_batch_file(p))
        elif _looks_like_cached_batch_file(path):
            paths.append(path)
    if paths:
        return {i: {"cached_batch": batch} for i, batch in enumerate(_load_cached_batches_from_files(paths))}, {"__cached_batches__": []}

    csv_files: List[Path] = []
    for raw_path in sample_files_list:
        path = Path(raw_path)
        files = [p for p in sorted(path.rglob("*.csv")) if p.is_file()] if path.is_dir() else [path]
        csv_files.extend(p for p in files if p.suffix.lower() == ".csv")
    if csv_files:
        return _load_official_csv_sample_files(csv_files)

    item_dict: Dict[int, Dict[str, Any]] = {}
    user_seq: Dict[int, List[int]] = {}
    for raw_path in sample_files_list:
        path = Path(raw_path)
        files = [p for p in sorted(path.rglob("*")) if p.is_file()] if path.is_dir() else [path]
        for file_path in files:
            for record in _iter_records(file_path):
                logid = _as_int(_first_key(record, LOGID_KEYS), default=len(item_dict))
                user_id = _as_int(_first_key(record, USER_KEYS), default=logid)
                features = _features_from_record(record)
                item_dict[logid] = {"logid": logid, "user_id": user_id, "features": features, "raw": record}
                user_seq.setdefault(user_id, []).extend(features)
    return item_dict, user_seq


class CTRTestSeqDataset:
    def __init__(self, test_logids_ordered, item_dict, user_seq, max_feasign_per_slot, max_ctx_len):
        self.item_dict = item_dict or {}
        self.user_seq = user_seq or {}
        self.cached_batches = [v["cached_batch"] for _, v in sorted(self.item_dict.items()) if isinstance(v, Mapping) and "cached_batch" in v]
        self.test_logids_ordered = [_as_int(x) for x in test_logids_ordered]
        if self.cached_batches and not self.test_logids_ordered:
            self.test_logids_ordered = list(range(len(self.cached_batches)))
        self.pred_logids = set(self.test_logids_ordered)
        self.max_feasign_per_slot = max_feasign_per_slot
        self.max_ctx_len = int(max_ctx_len or 1)
        self.official_csv = bool(self.item_dict) and all(
            isinstance(v, Mapping) and {"userid", "signs", "slots"}.issubset(v.keys())
            for v in self.item_dict.values()
        )
        self.user_items: Dict[int, List[Tuple[int, Dict[int, List[int]], float]]] = {}
        if self.official_csv:
            user_items: Dict[int, List[Tuple[int, Dict[int, List[int]], float]]] = defaultdict(list)
            for raw_logid, rec in self.item_dict.items():
                logid = int(rec.get("logid", raw_logid))
                userid = int(rec["userid"])
                feasign: Dict[int, List[int]] = defaultdict(list)
                signs = rec["signs"].tolist() if isinstance(rec["signs"], np.ndarray) else list(rec["signs"])
                slots = rec["slots"].tolist() if isinstance(rec["slots"], np.ndarray) else list(rec["slots"])
                for slot, sign in zip(slots, signs):
                    feasign[int(slot)].append(int(sign))
                if isinstance(max_feasign_per_slot, Mapping):
                    feasign = {
                        slot: signs[:max_feasign_per_slot[slot]]
                        if max_feasign_per_slot.get(slot, -1) != -1 else signs
                        for slot, signs in feasign.items()
                    }
                user_items[userid].append((logid, dict(feasign), float(rec.get("clk", 0))))
            self.user_items = dict(user_items)
            self.user_ids = sorted(self.user_items.keys())
            self.num_users = len(self.user_ids)
            self.total_samples = len(self.item_dict)
            self.max_slot_id = 28
        else:
            self.user_ids = []
            self.num_users = 0
            self.total_samples = len(self.item_dict)
            self.max_slot_id = 28 if self.cached_batches else max(0, self.max_ctx_len - 1)

    def __len__(self) -> int:
        if self.cached_batches:
            return len(self.cached_batches)
        if self.official_csv:
            return len(self.user_ids)
        return len(self.test_logids_ordered)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cached_batches:
            return self.cached_batches[idx]
        if self.official_csv:
            userid = self.user_ids[idx]
            items = list(self.user_items[userid])
            if userid in self.user_seq:
                seq_order = {logid: i for i, logid in enumerate(self.user_seq[userid])}
                items.sort(key=lambda x: seq_order.get(x[0], x[0]))
            else:
                items.sort(key=lambda x: x[0])
            logids = [logid for logid, _, _ in items]
            return {
                "userid": userid,
                "logids": logids,
                "feasigns": [feasign for _, feasign, _ in items],
                "labels": [label for _, _, label in items],
                "pred_mask": [1 if not self.pred_logids or logid in self.pred_logids else 0 for logid in logids],
            }
        logid = self.test_logids_ordered[idx]
        record = self.item_dict.get(logid, {"logid": logid, "user_id": logid, "features": [0]})
        user_id = _as_int(record.get("user_id"), default=logid)
        history = list(self.user_seq.get(user_id, []))
        features = list(record.get("features") or [])
        seq = (history + features)[-self.max_ctx_len:] or [0]
        return {"logid": logid, "user_id": user_id, "features": seq, "pred_mask": True}


def _is_cached_batch(row: Mapping[str, Any]) -> bool:
    return {"logid", "pred_mask"}.issubset(row.keys()) and any(isinstance(k, int) for k in row.keys())


def _merge_offsets(parts: Sequence[Any]) -> Any:
    if not parts:
        return parts
    if torch is not None and all(torch.is_tensor(p) for p in parts):
        out = [parts[0]]
        base = int(parts[0][-1].item())
        for offsets in parts[1:]:
            out.append(offsets[1:] + base)
            base += int(offsets[-1].item())
        return torch.cat(out, dim=0)
    arrays = [np.asarray(p) for p in parts]
    out = [arrays[0]]
    base = int(arrays[0][-1])
    for offsets in arrays[1:]:
        out.append(offsets[1:] + base)
        base += int(offsets[-1])
    return np.concatenate(out, axis=0)


def _concat_values(parts: Sequence[Any]) -> Any:
    if torch is not None and all(torch.is_tensor(p) for p in parts):
        return torch.cat(list(parts), dim=0)
    return np.concatenate([np.asarray(p) for p in parts], axis=0)


def _merge_cached_batches(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 1:
        return dict(batch[0])
    merged: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [row[key] for row in batch if key in row]
        if not values:
            continue
        first = values[0]
        if isinstance(key, int) and isinstance(first, tuple) and len(first) == 2:
            slot_values = [value[0] for value in values]
            slot_offsets = [value[1] for value in values]
            merged[key] = (_concat_values(slot_values), _merge_offsets(slot_offsets))
        elif key == "user_offsets":
            row_counts = [int(row["logid"].shape[0]) for row in batch]
            out = [values[0]]
            base = row_counts[0]
            for offsets, rows in zip(values[1:], row_counts[1:]):
                out.append(offsets[1:] + base)
                base += rows
            merged[key] = torch.cat(out, dim=0) if torch is not None and all(torch.is_tensor(v) for v in values) else np.concatenate([np.asarray(x) for x in out], axis=0)
        elif torch is not None and all(torch.is_tensor(v) for v in values):
            merged[key] = torch.cat(values, dim=0) if first.ndim > 0 else torch.stack(values, dim=0)
        elif all(isinstance(v, np.ndarray) for v in values):
            merged[key] = np.concatenate(values, axis=0) if first.ndim > 0 else np.stack(values, axis=0)
        else:
            merged[key] = values
    return merged


def make_collate_fn(max_slot_id) -> Callable:
    def collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if batch and all(isinstance(row, Mapping) and _is_cached_batch(row) for row in batch):
            return _merge_cached_batches(batch)
        if batch and all(isinstance(row, Mapping) and {"logids", "feasigns", "labels"}.issubset(row.keys()) for row in batch):
            all_userids = []
            all_logids = []
            all_labels = []
            all_pred_masks = []
            all_feasigns = []
            user_offsets = [0]
            for item in batch:
                for i, logid in enumerate(item["logids"]):
                    all_userids.append(item["userid"])
                    all_logids.append(logid)
                    all_labels.append(item["labels"][i])
                    all_pred_masks.append(item["pred_mask"][i])
                    all_feasigns.append(item["feasigns"][i])
                user_offsets.append(len(all_labels))
            slot_data = {}
            for slot in range(1, int(max_slot_id) + 1):
                values = []
                offsets = [0]
                for feasign in all_feasigns:
                    if slot in feasign:
                        values.extend(feasign[slot])
                    offsets.append(len(values))
                slot_data[slot] = (
                    torch.tensor(values, dtype=torch.long),
                    torch.tensor(offsets, dtype=torch.long),
                )
            result = {
                "userid": torch.tensor(all_userids, dtype=torch.long),
                "logid": torch.tensor(all_logids, dtype=torch.long),
                "label": torch.tensor(all_labels, dtype=torch.float32),
                "pred_mask": torch.tensor(all_pred_masks, dtype=torch.bool),
                "user_offsets": torch.tensor(user_offsets, dtype=torch.long),
            }
            result.update(slot_data)
            return result
        batch_size = len(batch)
        max_len = max((len(row.get("features", [])) for row in batch), default=1)
        feature_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.bool_)
        for i, row in enumerate(batch):
            vals = [abs(_as_int(x)) for x in row.get("features", [])][-max_len:]
            if vals:
                feature_ids[i, -len(vals):] = vals
                attention_mask[i, -len(vals):] = True
        return {
            "logid": np.asarray([_as_int(row.get("logid")) for row in batch], dtype=np.int64),
            "user_id": np.asarray([_as_int(row.get("user_id")) for row in batch], dtype=np.int64),
            "feature_ids": feature_ids,
            "attention_mask": attention_mask,
            "pred_mask": np.ones((batch_size,), dtype=np.bool_),
        }
    return collate


class RepEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx=0, slot_num=0, d_model=0):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        self.emb_dim = emb_dim
        self.slot_num = slot_num
        self.input_norm = nn.LayerNorm(slot_num * emb_dim)
        self.linear = nn.Linear(in_features=slot_num * emb_dim, out_features=d_model)

    def forward(self, batch):
        pooled_embs = []
        max_idx = self.emb.num_embeddings - 1
        for i in range(self.slot_num):
            values, offsets = batch[i + 1]
            offsets = offsets.to(values.device)
            values = values.clamp(0, max_idx)
            sign_emb = self.emb(values)
            res = torch.segment_reduce(sign_emb, reduce="sum", offsets=offsets, initial=0)
            pooled_embs.append(res)
        fused_embs = torch.cat(pooled_embs, dim=1)
        norm_emb = self.input_norm(fused_embs)
        return self.linear(norm_emb)


def scaled_dot_product(q, k, v, extension):
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if extension is not None and "mask" in extension:
        mask = extension["mask"]
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


class Expert(nn.Module):
    def __init__(self, d_model, dim_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TopKGate(nn.Module):
    def __init__(self, d_model, num_experts, k=2, noisy_gating=True):
        super().__init__()
        self.w_g = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

    def forward(self, x):
        logits = self.w_g(x)
        if self.noisy_gating and self.training:
            logits = logits + torch.randn_like(logits) * 0.1
        probs = torch.softmax(logits, dim=-1)
        topk_score, topk_idx = torch.topk(probs, self.k, dim=-1)
        return topk_idx, topk_score, probs


class SMoE(nn.Module):
    def __init__(self, d_model, dim_ff, num_experts, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([Expert(d_model, dim_ff) for _ in range(num_experts)])
        self.gate = TopKGate(d_model, num_experts, k=k)

    def forward(self, x):
        B, S, D = x.shape
        topk_idx, topk_score, probs = self.gate(x)
        out = torch.zeros_like(x)
        x_flat = x.reshape(-1, D)
        idx_flat = topk_idx.reshape(-1, self.k)
        score_flat = topk_score.reshape(-1, self.k)
        out_flat = out.reshape(-1, D)
        for i in range(self.num_experts):
            mask = (idx_flat == i)
            if not mask.any():
                continue
            token_idx, k_idx = mask.nonzero(as_tuple=True)
            expert_out = self.experts[i](x_flat[token_idx])
            weight = score_flat[token_idx, k_idx].unsqueeze(-1)
            out_flat[token_idx] += expert_out * weight
        importance = probs.sum(dim=(0, 1))
        moe_loss = importance.std() / (importance.mean() + 1e-6)
        return out, moe_loss


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_ff, act="relu", attention_fn=scaled_dot_product):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_layers = num_layers
        assert d_model % n_heads == 0
        self.qkv_proj = nn.ModuleList([nn.Linear(d_model, 3 * d_model) for _ in range(num_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.ffn1 = nn.ModuleList([nn.Linear(d_model, dim_ff) for _ in range(num_layers)])
        self.ffn2 = nn.ModuleList([nn.Linear(dim_ff, d_model) for _ in range(num_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.act = getattr(F, act)
        self.attention_fn = attention_fn
        self.moe = nn.ModuleList([SMoE(d_model, dim_ff, num_experts=8, k=2) for _ in range(num_layers)])

    def forward(self, x, extension):
        x = x.unsqueeze(0)
        B, S, D = x.shape
        moe_loss_total = 0.0
        for i in range(self.num_layers):
            residual = x
            x = self.norm1[i](x)
            qkv = self.qkv_proj[i](x)
            qkv = qkv.view(B, S, self.n_heads, 3 * self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)
            q, k, v = torch.split(qkv, self.head_dim, dim=-1)
            attn_out = self.attention_fn(q, k, v, extension)
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
            x = residual + self.out_proj[i](attn_out)
            residual = x
            x = self.norm2[i](x)
            moe_out, moe_loss = self.moe[i](x)
            x = residual + moe_out
            moe_loss_total = moe_loss_total + moe_loss
        return x, moe_loss_total


class CTRModel(nn.Module):
    def __init__(self, rep_encoder, seq_encoder, d_model):
        super().__init__()
        self.rep_encoder = rep_encoder
        self.seq_encoder = seq_encoder
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

    def get_sequence_causal_mask(self, seq_info):
        lengths = seq_info[1:] - seq_info[:-1]
        lengths = lengths.view(-1)
        indices = torch.cumsum(torch.ones_like(lengths), dim=0) - 1
        result = torch.repeat_interleave(indices, lengths)
        a = result.view(1, -1) - result.view(-1, 1)
        return torch.tril((a == 0).to(torch.int32)).bool()

    def forward(self, batch):
        seq_input = self.rep_encoder(batch)
        seq_mask = self.get_sequence_causal_mask(batch["user_offsets"])
        encoder_output, moe_loss = self.seq_encoder(
            x=seq_input,
            extension={"mask": seq_mask.unsqueeze(0).unsqueeze(0)},
        )
        encoder_output_dim = encoder_output.shape[-1]
        encoder_output = encoder_output.reshape(1, -1, encoder_output_dim).squeeze(0)
        pred = self.linear(encoder_output)
        pred_logits = torch.clamp(pred, min=-15.0, max=15.0)
        return pred_logits, moe_loss


def _build_official_model() -> nn.Module:
    emb_dim = 512
    slot_num = 28
    vocab_size = 5_000_000
    d_model = 512
    n_heads = 8
    num_layers = 8
    dim_ff = 1024
    rep_encoder = RepEncoder(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        padding_idx=0,
        slot_num=slot_num,
        d_model=d_model,
    )
    seq_encoder = TransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_ff=dim_ff,
        act="relu",
    )
    return CTRModel(rep_encoder, seq_encoder, d_model=d_model)


def _extract_state_dict(ckpt: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in STATE_DICT_KEYS:
        value = ckpt.get(key)
        if isinstance(value, Mapping):
            return value
    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    return None


def _load_state_dict_strict(model: nn.Module, state_dict: Mapping[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        cleaned = _clean_state_dict_keys(state_dict)
        original_keys = [str(k) for k in state_dict.keys()]
        if list(cleaned.keys()) == original_keys:
            raise
        model.load_state_dict(cleaned)


class ModelAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch: Mapping[str, Any]):
        out = self.model(batch)
        if isinstance(out, tuple):
            if len(out) >= 2:
                return out[0], out[1]
            return out[0], torch.zeros((), device=_object_device(batch))
        if isinstance(out, Mapping):
            logits = None
            for key in ("logits", "pred", "prediction", "output"):
                if key in out:
                    logits = out[key]
                    break
            if logits is None:
                raise RuntimeError("model output mapping does not contain logits/pred/prediction/output")
            return logits, out.get("moe_loss", torch.zeros((), device=_object_device(batch)))
        return out, torch.zeros((), device=_object_device(batch))


def _object_device(obj: Any):
    if torch is not None and torch.is_tensor(obj):
        return obj.device
    if isinstance(obj, Mapping):
        for value in obj.values():
            device = _object_device(value)
            if device is not None:
                return device
    if isinstance(obj, (list, tuple)):
        for value in obj:
            device = _object_device(value)
            if device is not None:
                return device
    return torch.device("cpu") if torch is not None else None


def _clean_state_dict_keys(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in state_dict.items():
        key = str(key)
        for prefix in ("module.", "model.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def _try_load_state_dict_into_existing(model: nn.Module, ckpt: Mapping[str, Any]) -> bool:
    state_dict = _extract_state_dict(ckpt)
    if state_dict is None:
        return False
    _load_state_dict_strict(model, state_dict)
    return True


def load_model(ckpt_path: Path):
    _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_path = Path(ckpt_path)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    model = None
    epoch = "?"
    if isinstance(ckpt, nn.Module):
        model = ckpt
    elif isinstance(ckpt, Mapping):
        epoch = ckpt.get("epoch", "?")
        for key in MODEL_KEYS:
            value = ckpt.get(key)
            if isinstance(value, nn.Module):
                model = value
                _try_load_state_dict_into_existing(model, ckpt)
                break
        if model is None:
            state_dict = _extract_state_dict(ckpt)
            if state_dict is not None:
                model = _build_official_model()
                _load_state_dict_strict(model, state_dict)

    if model is None:
        raise RuntimeError(
            "ckpt.pt did not contain a loadable torch.nn.Module or official model_state_dict. "
            "infer.py intentionally does not fall back to a random replacement model."
        )

    print(f"[INFO] Loaded checkpoint from {ckpt_path} (epoch={epoch})")
    model.eval().to(device)
    return ModelAdapter(model), device


def _to_tensor_if_plain_numeric_sequence(value: Any):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, (int, bool, np.integer, np.bool_)) for x in value):
        return torch.tensor(value)
    return None


def move_batch_to_device(batch, device):
    _require_torch()

    def move(obj: Any, key: Any = None):
        if torch.is_tensor(obj):
            tensor = obj
        else:
            tensor = _to_tensor_if_plain_numeric_sequence(obj)
        if tensor is not None:
            if key in {"pred_mask", "attention_mask"}:
                tensor = tensor.bool()
            elif key in {"logid", "userid", "user_id", "feature_ids", "user_offsets"} or isinstance(key, int):
                tensor = tensor.long()
            return tensor.to(device, non_blocking=True)
        if isinstance(obj, Mapping):
            return {k: move(v, k) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(move(v, key) for v in obj)
        if isinstance(obj, list):
            return [move(v, key) for v in obj]
        return obj

    return move(batch)


def _find_cached_batch_files(code_root: Path) -> List[Path]:
    candidates = [
        code_root / "dataset" / "cached_batches",
        code_root / "cached_batches",
        code_root / "official" / "dataset_cached_probe" / "dataset" / "cached_batches",
    ]
    for directory in candidates:
        if directory.exists():
            files = sorted(directory.glob("shard_*.pt"))
            if files:
                return files
    return []


def _run_model(model: nn.Module, batch: Mapping[str, Any]):
    with torch.inference_mode():
        logits, _ = model(batch)
    if logits.ndim > 1 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    return logits


def _write_prediction_values(preds: Sequence[float], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        for pred in preds:
            f.write(f"{float(pred):.8f}\n")


def _read_test_logids_in_order(test_file: Path) -> List[int]:
    logids: List[int] = []
    with test_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logids.append(int(line.split(",", 1)[0]))
    return logids


def _order_prediction_values(predictions: Sequence[Tuple[int, float]], code_root: Path) -> List[float]:
    test_file = code_root / "dataset" / "test.csv"
    if not test_file.exists():
        return [float(prob) for _, prob in predictions]
    preds_by_logid: Dict[int, List[float]] = defaultdict(list)
    for logid, prob in predictions:
        preds_by_logid[int(logid)].append(float(prob))
    ordered: List[float] = []
    for logid in _read_test_logids_in_order(test_file):
        values = preds_by_logid.get(logid)
        if not values:
            raise RuntimeError(f"missing prediction for logid {logid}")
        ordered.append(values.pop(0))
    return ordered


def run_inference(code_root: Path) -> Path:
    _require_torch()
    code_root = Path(code_root)
    shard_files = _find_cached_batch_files(code_root)
    if not shard_files:
        raise FileNotFoundError(f"cached batch shards not found under {code_root / 'dataset' / 'cached_batches'}")
    ckpt_path = code_root / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print(f"[INFO] loading cached batch shards from {shard_files[0].parent}")
    model, device = load_model(ckpt_path)
    print(f"[INFO] Model ready. Device: {device}")

    predictions: List[Tuple[int, float]] = []
    for shard_path in shard_files:
        batches = _load_cached_batches_from_files([shard_path])
        print(f"[INFO] loaded {len(batches)} batches from {shard_path.name}")
        for batch in batches:
            moved = move_batch_to_device(batch, device)
            logits = _run_model(model, moved)
            pred_mask = moved["pred_mask"].bool()
            probs = torch.sigmoid(logits[pred_mask]).detach().cpu().float().tolist()
            logids = moved["logid"][pred_mask].detach().cpu().long().tolist()
            predictions.extend(zip(logids, probs))

    output_path = code_root / "predict.txt"
    ordered_values = _order_prediction_values(predictions, code_root)
    _write_prediction_values(ordered_values, output_path)
    print(f"[INFO] predictions written to {output_path.name}, total: {len(ordered_values)}")
    return output_path


def self_test(code_root: Path) -> None:
    _require_torch()
    shard_files = _find_cached_batch_files(Path(code_root))
    if not shard_files:
        raise FileNotFoundError("self-test requires at least one cached shard; expected official/dataset_cached_probe/... or dataset/cached_batches")
    batches = _load_cached_batches_from_files([shard_files[0]])
    if not batches:
        raise RuntimeError(f"no batches loaded from {shard_files[0]}")
    total_rows = sum(int(batch["logid"].shape[0]) for batch in batches)
    total_preds = sum(int(batch["pred_mask"].sum().item()) for batch in batches)
    moved = move_batch_to_device(batches[0], torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    assert isinstance(moved, dict)
    assert "logid" in moved and "pred_mask" in moved and "user_offsets" in moved
    assert all(slot in moved and isinstance(moved[slot], tuple) and len(moved[slot]) == 2 for slot in range(1, 29))
    assert moved["logid"].shape[0] == moved["pred_mask"].shape[0]
    print(f"[SELF-TEST] shard={shard_files[0].name} batches={len(batches)} rows={total_rows} preds={total_preds}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("code_root", nargs="?", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test(Path(args.code_root))
    else:
        run_inference(Path(args.code_root))


if __name__ == "__main__":
    main()
