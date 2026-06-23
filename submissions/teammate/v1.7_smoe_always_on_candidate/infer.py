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


# ---------------------------------------------------------------------------
# Try TensorRT (optional — the script still works without it)
# ---------------------------------------------------------------------------
try:
    import tensorrt as trt
    _HAS_TENSORRT = True
except Exception:
    trt = None
    _HAS_TENSORRT = False

try:
    import torch_tensorrt
    _HAS_TORCH_TRT = True
except Exception:
    torch_tensorrt = None
    _HAS_TORCH_TRT = False


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


def _make_offsets_like(reference: Any, lengths: Sequence[int]) -> Any:
    offsets = [0]
    total = 0
    for length in lengths:
        total += int(length)
        offsets.append(total)
    if torch is not None and torch.is_tensor(reference):
        return torch.tensor(offsets, dtype=reference.dtype, device=reference.device)
    if isinstance(reference, np.ndarray):
        return np.asarray(offsets, dtype=reference.dtype)
    return offsets


def _slice_rows_first_dim(value: Any, ranges: Sequence[Tuple[int, int]], total_rows: int) -> Any:
    if torch is not None and torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == total_rows:
        return torch.cat([value[start:end] for start, end in ranges], dim=0) if ranges else value[:0]
    if isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == total_rows:
        return np.concatenate([value[start:end] for start, end in ranges], axis=0) if ranges else value[:0]
    if isinstance(value, list) and len(value) == total_rows:
        out: List[Any] = []
        for start, end in ranges:
            out.extend(value[start:end])
        return out
    if isinstance(value, tuple) and len(value) == total_rows:
        out = []
        for start, end in ranges:
            out.extend(value[start:end])
        return tuple(out)
    return value


def _slice_slot_offsets(slot_value: Any, ranges: Sequence[Tuple[int, int]], total_rows: int) -> Any:
    if not (isinstance(slot_value, tuple) and len(slot_value) == 2):
        return slot_value
    values, offsets = slot_value
    if torch is not None and torch.is_tensor(offsets) and offsets.ndim == 1 and int(offsets.numel()) == total_rows + 1:
        parts = []
        offset_parts = [offsets.new_zeros((1,))]
        base = 0
        for start, end in ranges:
            start_value = int(offsets[start].item())
            end_value = int(offsets[end].item())
            if end_value > start_value:
                parts.append(values[start_value:end_value])
            rel = offsets[start + 1:end + 1] - offsets[start] + base
            offset_parts.append(rel)
            base += end_value - start_value
        new_values = torch.cat(parts, dim=0) if parts else values[:0]
        new_offsets = torch.cat(offset_parts, dim=0)
        return new_values, new_offsets
    if isinstance(offsets, np.ndarray) and offsets.ndim == 1 and int(offsets.shape[0]) == total_rows + 1:
        parts = []
        offset_parts = [np.zeros((1,), dtype=offsets.dtype)]
        base = 0
        for start, end in ranges:
            start_value = int(offsets[start])
            end_value = int(offsets[end])
            if end_value > start_value:
                parts.append(values[start_value:end_value])
            rel = offsets[start + 1:end + 1] - offsets[start] + base
            offset_parts.append(rel)
            base += end_value - start_value
        new_values = np.concatenate(parts, axis=0) if parts else values[:0]
        new_offsets = np.concatenate(offset_parts, axis=0)
        return new_values, new_offsets
    return slot_value


def _prune_batch_for_predictions(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    """Drop rows that cannot affect any requested prediction.

    This is exact for the official causal sequence model: users with no
    ``pred_mask`` rows produce no output, and rows after a user's final
    prediction are future context that cannot affect earlier logits.
    """
    if "pred_mask" not in batch or "user_offsets" not in batch:
        return batch
    if torch is None:
        return batch
    pred_mask = batch["pred_mask"]
    user_offsets = batch["user_offsets"]
    pred_cpu = pred_mask.detach().cpu().bool() if torch.is_tensor(pred_mask) else torch.as_tensor(pred_mask, dtype=torch.bool)
    offsets_cpu = user_offsets.detach().cpu().long() if torch.is_tensor(user_offsets) else torch.as_tensor(user_offsets, dtype=torch.long)
    total_rows = int(pred_cpu.numel())
    if total_rows == 0 or offsets_cpu.numel() < 2 or int(offsets_cpu[-1].item()) != total_rows:
        return batch
    if not bool(pred_cpu.any().item()):
        no_pred = dict(batch)
        no_pred["_has_predictions"] = False
        return no_pred

    ranges: List[Tuple[int, int]] = []
    for user_idx in range(int(offsets_cpu.numel()) - 1):
        start = int(offsets_cpu[user_idx].item())
        end = int(offsets_cpu[user_idx + 1].item())
        if end <= start:
            continue
        pred_positions = torch.nonzero(pred_cpu[start:end], as_tuple=False).view(-1)
        if pred_positions.numel() == 0:
            continue
        last_pred = start + int(pred_positions[-1].item())
        ranges.append((start, last_pred + 1))
    if not ranges:
        return batch
    kept_rows = sum(end - start for start, end in ranges)
    if kept_rows == total_rows and len(ranges) == int(offsets_cpu.numel()) - 1:
        unchanged = dict(batch)
        unchanged["_has_predictions"] = True
        return unchanged

    pruned: Dict[Any, Any] = {"_has_predictions": True}
    lengths = [end - start for start, end in ranges]
    for key, value in batch.items():
        if key == "user_offsets":
            pruned[key] = _make_offsets_like(value, lengths)
        elif isinstance(key, int):
            pruned[key] = _slice_slot_offsets(value, ranges, total_rows)
        else:
            pruned[key] = _slice_rows_first_dim(value, ranges, total_rows)
    return pruned


def _slot_offsets_to_cpu(offsets: Any) -> torch.Tensor | None:
    if torch is None:
        return None
    if torch.is_tensor(offsets):
        if offsets.ndim != 1:
            return None
        return offsets.detach().cpu().long()
    try:
        tensor = torch.as_tensor(offsets, dtype=torch.long)
    except Exception:
        return None
    return tensor if tensor.ndim == 1 else None


def _annotate_fast_batch_metadata(batch: Mapping[str, Any], max_slot_id: int | None = None) -> Mapping[str, Any]:
    if torch is None or "pred_mask" not in batch:
        return batch
    annotated = dict(batch)
    pred_mask = batch["pred_mask"]
    pred_cpu = pred_mask.detach().cpu().bool() if torch.is_tensor(pred_mask) else torch.as_tensor(pred_mask, dtype=torch.bool)
    pred_index = torch.nonzero(pred_cpu, as_tuple=False).view(-1).long()
    num_predictions = int(pred_index.numel())
    annotated["_has_predictions"] = bool(num_predictions)
    annotated["_num_predictions"] = num_predictions
    annotated["_pred_index"] = pred_index

    total_rows = int(pred_cpu.numel())
    if max_slot_id is None:
        slot_ids = sorted(key for key in batch.keys() if isinstance(key, int))
    else:
        slot_ids = list(range(1, int(max_slot_id) + 1))
    active: List[int] = []
    empty: List[int] = []
    single_value: List[int] = []
    variable: List[int] = []
    for slot_id in slot_ids:
        slot_value = batch.get(slot_id)
        if not (isinstance(slot_value, tuple) and len(slot_value) == 2):
            continue
        offsets = _slot_offsets_to_cpu(slot_value[1])
        if offsets is None or int(offsets.numel()) != total_rows + 1:
            continue
        counts = offsets[1:] - offsets[:-1]
        slot_idx = int(slot_id) - 1
        if int(offsets[-1].item()) == 0:
            empty.append(slot_idx)
        else:
            active.append(slot_idx)
            if bool(torch.all(counts == 1).item()):
                single_value.append(slot_idx)
            else:
                variable.append(slot_idx)
    annotated["_active_slot_indices"] = active
    annotated["_empty_slot_indices"] = empty
    annotated["_single_value_slot_indices"] = single_value
    annotated["_variable_slot_indices"] = variable
    return annotated


def _prepare_official_batch(batch: Mapping[str, Any], max_slot_id: int | None = None) -> Mapping[str, Any]:
    return _annotate_fast_batch_metadata(_prune_batch_for_predictions(batch), max_slot_id=max_slot_id)


def _sequence_causal_mask(seq_info):
    lengths = seq_info[1:] - seq_info[:-1]
    lengths = lengths.view(-1)
    indices = torch.cumsum(torch.ones_like(lengths), dim=0) - 1
    result = torch.repeat_interleave(indices, lengths)
    a = result.view(1, -1) - result.view(-1, 1)
    return torch.tril((a == 0).to(torch.int32)).bool()


def make_collate_fn(max_slot_id) -> Callable:
    def collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if batch and all(isinstance(row, Mapping) and _is_cached_batch(row) for row in batch):
            return dict(_prepare_official_batch(_merge_cached_batches(batch), max_slot_id=max_slot_id))
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
            return dict(_prepare_official_batch(result, max_slot_id=max_slot_id))
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

    def _pool_slot(self, values, offsets, is_single_value: bool):
        offsets = offsets.to(values.device, dtype=torch.long, non_blocking=True)
        rows = max(int(offsets.numel()) - 1, 0)
        if rows == 0:
            return self.emb.weight.new_zeros((0, self.emb_dim))
        max_idx = self.emb.num_embeddings - 1
        if int(values.numel()) == 0:
            return self.emb.weight.new_zeros((rows, self.emb_dim))
        values = values.to(dtype=torch.long).clamp(0, max_idx)
        if is_single_value and int(values.numel()) == rows:
            return self.emb(values)
        return F.embedding_bag(
            values,
            self.emb.weight,
            offsets,
            mode="sum",
            include_last_offset=True,
        )

    def forward(self, batch):
        pooled_embs = []
        single_value_slots = set(batch.get("_single_value_slot_indices", ())) if isinstance(batch, Mapping) else set()
        empty_slots = set(batch.get("_empty_slot_indices", ())) if isinstance(batch, Mapping) else set()
        rows = None
        pred_mask = batch.get("pred_mask") if isinstance(batch, Mapping) else None
        if pred_mask is not None and hasattr(pred_mask, "shape"):
            rows = int(pred_mask.shape[0])
        for i in range(self.slot_num):
            values, offsets = batch[i + 1]
            if i in empty_slots:
                if rows is None:
                    rows = max(int(offsets.numel()) - 1, 0)
                res = self.emb.weight.new_zeros((rows, self.emb_dim))
            else:
                res = self._pool_slot(values, offsets, i in single_value_slots)
                if rows is None:
                    rows = int(res.shape[0])
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
        topk_logits, topk_idx = torch.topk(logits, self.k, dim=-1)
        log_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
        topk_score = torch.exp(topk_logits - log_denom)
        return topk_idx, topk_score


class SMoE(nn.Module):
    """Sparse Mixture of Experts with configurable gate threshold.

    ``gate_threshold`` controls the latency/accuracy trade-off:
      - None (default): always use top-k experts.
      - 0.0: always top-1 expert (maximum speed).
      - 0.5–0.99: top-1 when the gate's softmax score >= threshold,
        otherwise top-k. Higher threshold = more tokens use top-2.

    Applied in ``_forward_sparse_dispatch`` by zeroing out scores for
    skipped experts, so they are never computed.  NOT pruning — all
    weights intact.
    """

    def __init__(self, d_model, dim_ff, num_experts, k=2, gate_threshold=None):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate_threshold = gate_threshold
        self.experts = nn.ModuleList([Expert(d_model, dim_ff) for _ in range(num_experts)])
        self.gate = TopKGate(d_model, num_experts, k=k)

    def set_threshold(self, threshold):
        self.gate_threshold = threshold

    def _apply_threshold(self, score_flat):
        """Zero out expert scores below gate_threshold (in-place)."""
        if self.gate_threshold is None:
            return
        if self.gate_threshold <= 0.0:
            score_flat[:, 1:] = 0.0          # always top-1
        else:
            confident = score_flat[:, 0] >= self.gate_threshold
            if confident.any():
                score_flat[confident, 1:] = 0.0

    def _forward_sparse_dispatch(self, x, topk_idx, topk_score):
        B, S, D = x.shape
        out = torch.zeros_like(x)
        x_flat = x.reshape(-1, D)
        idx_flat = topk_idx.reshape(-1, self.k)
        score_flat = topk_score.reshape(-1, self.k)
        self._apply_threshold(score_flat)
        out_flat = out.reshape(-1, D)
        for i in range(self.num_experts):
            mask = (idx_flat == i) & (score_flat > 0.0)
            if not mask.any():
                continue
            token_idx, k_idx = mask.nonzero(as_tuple=True)
            expert_out = self.experts[i](x_flat[token_idx])
            weight = score_flat[token_idx, k_idx].unsqueeze(-1)
            out_flat.index_add_(0, token_idx, expert_out * weight)
        return out

    def _forward_batched_sparse(self, x, topk_idx, topk_score):
        """Zero-Python-loop dispatch using stacked weights + batched matmul.

        Stacks all expert weights once, then for each expert grabs its
        tokens with a single index_select.  Avoids calling expert.forward()
        in a loop — uses ``F.linear`` with pre-stacked weights.
        """
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        idx_flat = topk_idx.reshape(-1, self.k)
        score_flat = topk_score.reshape(-1, self.k)
        self._apply_threshold(score_flat)
        N = x_flat.shape[0]

        # Stack expert weights once (cached by _stacked_expert_params)
        w1, b1, w2, b2 = self._stacked_expert_params(x_flat)
        # w1: [E, F, D]  b1: [E, F]  w2: [E, D, F]  b2: [E, D]

        out = torch.zeros(N, D, dtype=x_flat.dtype, device=x_flat.device)

        for e in range(self.num_experts):
            mask = (idx_flat == e) & (score_flat > 0.0)
            token_idx, k_pos = mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue

            x_e = x_flat[token_idx]                            # [n_e, D]
            h = F.linear(x_e, w1[e], b1[e])                   # [n_e, F]
            h = F.relu(h)
            out_e = F.linear(h, w2[e], b2[e])                  # [n_e, D]
            weights = score_flat[token_idx, k_pos].unsqueeze(-1)
            out.index_add_(0, token_idx, out_e * weights)

        return out.view(B, S, D)

    def _stacked_expert_params(self, x):
        """Cache stacked [E, F, D]/[E, D, F] weight tensors."""
        key = (x.device, x.dtype)
        if hasattr(self, "_stacked_cache") and self._stacked_cache is not None:
            cached = self._stacked_cache
            if cached[0] == key:
                return cached[1]
        w1 = torch.stack([e.fc1.weight.to(dtype=x.dtype) for e in self.experts], dim=0).contiguous()
        b1 = torch.stack([e.fc1.bias.to(dtype=x.dtype) for e in self.experts], dim=0).contiguous()
        w2 = torch.stack([e.fc2.weight.to(dtype=x.dtype) for e in self.experts], dim=0).contiguous()
        b2 = torch.stack([e.fc2.bias.to(dtype=x.dtype) for e in self.experts], dim=0).contiguous()
        params = (w1, b1, w2, b2)
        self._stacked_cache = (key, params)
        return params

    def forward(self, x):
        topk_idx, topk_score = self.gate(x)
        return self._forward_batched_sparse(x, topk_idx, topk_score), x.new_zeros(())


# =========================================================================
#  Transformer Encoder
# =========================================================================

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_ff, act="relu",
                 attention_fn=scaled_dot_product, gate_threshold=None):
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
        self.moe = nn.ModuleList([
            SMoE(d_model, dim_ff, num_experts=8, k=2, gate_threshold=gate_threshold)
            for _ in range(num_layers)
        ])

    def set_gate_threshold(self, threshold):
        for moe_layer in self.moe:
            moe_layer.set_threshold(threshold)

    def _dense_attention(self, q, k, v, extension):
        return self.attention_fn(q, k, v, extension)

    def _block_causal_attention(self, q, k, v, user_offsets):
        B, H, S, HD = q.shape
        if B != 1 or user_offsets is None:
            return None
        if torch.is_tensor(user_offsets):
            offsets = user_offsets.detach().cpu().tolist()
        else:
            offsets = list(user_offsets)
        if len(offsets) < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != S:
            return None
        out = q.new_empty(q.shape)
        for start, end in zip(offsets[:-1], offsets[1:]):
            start = int(start)
            end = int(end)
            if end <= start:
                continue
            q_u = q[:, :, start:end, :]
            k_u = k[:, :, start:end, :]
            v_u = v[:, :, start:end, :]
            try:
                out[:, :, start:end, :] = F.scaled_dot_product_attention(
                    q_u, k_u, v_u,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )
            except Exception:
                length = end - start
                mask = torch.ones((length, length), dtype=torch.bool, device=q.device).tril()
                scores = torch.matmul(q_u, k_u.transpose(-2, -1)) / math.sqrt(HD)
                scores = scores.masked_fill(mask.view(1, 1, length, length) == 0, float("-inf"))
                attn = torch.softmax(scores, dim=-1)
                out[:, :, start:end, :] = torch.matmul(attn, v_u)
        return out

    def forward(self, x, extension=None, user_offsets=None):
        x = x.unsqueeze(0)
        B, S, D = x.shape
        if user_offsets is not None:
            if torch.is_tensor(user_offsets):
                block_offsets = user_offsets.detach().cpu().tolist()
            else:
                block_offsets = list(user_offsets)
        else:
            block_offsets = None
        moe_loss_total = x.new_zeros(())
        for i in range(self.num_layers):
            residual = x
            x = self.norm1[i](x)
            qkv = self.qkv_proj[i](x)
            qkv = qkv.view(B, S, self.n_heads, 3 * self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)
            q, k, v = torch.split(qkv, self.head_dim, dim=-1)
            attn_out = self._block_causal_attention(q, k, v, block_offsets)
            if attn_out is None:
                attn_out = self._dense_attention(q, k, v, extension)
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
            x = residual + self.out_proj[i](attn_out)
            residual = x
            x = self.norm2[i](x)
            moe_out, moe_loss = self.moe[i](x)
            x = residual + moe_out
            moe_loss_total = moe_loss_total + moe_loss
        return x, moe_loss_total


# =========================================================================
#  CTR Model
# =========================================================================

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
        encoder_output, moe_loss = self.seq_encoder(
            x=seq_input,
            user_offsets=batch.get("user_offsets"),
        )
        encoder_output_dim = encoder_output.shape[-1]
        encoder_output = encoder_output.reshape(1, -1, encoder_output_dim).squeeze(0)
        pred_index = batch.get("_pred_index") if isinstance(batch, Mapping) else None
        num_predictions = batch.get("_num_predictions") if isinstance(batch, Mapping) else None
        if pred_index is not None and num_predictions is not None:
            rows = int(encoder_output.shape[0])
            num_predictions = int(num_predictions)
            if 0 < num_predictions < rows:
                if not torch.is_tensor(pred_index):
                    pred_index = torch.as_tensor(pred_index, dtype=torch.long, device=encoder_output.device)
                elif pred_index.device != encoder_output.device:
                    pred_index = pred_index.to(encoder_output.device, non_blocking=True)
                pred_logits = encoder_output.new_zeros((rows, 1))
                selected = torch.clamp(self.linear(encoder_output.index_select(0, pred_index.long())), min=-15.0, max=15.0)
                pred_logits.index_copy_(0, pred_index.long(), selected)
                return pred_logits, moe_loss
        pred = self.linear(encoder_output)
        pred_logits = torch.clamp(pred, min=-15.0, max=15.0)
        return pred_logits, moe_loss


# =========================================================================
#  Model construction helpers
# =========================================================================

def _build_official_model(gate_threshold=None) -> nn.Module:
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
        gate_threshold=gate_threshold,
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
        with torch.inference_mode():
            pred_mask = batch.get("pred_mask") if isinstance(batch, Mapping) else None
            logid = batch.get("logid") if isinstance(batch, Mapping) else None
            has_predictions = batch.get("_has_predictions") if isinstance(batch, Mapping) else None
            if has_predictions is False or (has_predictions is None and pred_mask is not None and logid is not None):
                no_predictions = has_predictions is False
                if not no_predictions:
                    mask_tensor = pred_mask if torch.is_tensor(pred_mask) else torch.as_tensor(pred_mask)
                    no_predictions = mask_tensor.numel() > 0 and not bool(mask_tensor.bool().any().item())
                if no_predictions:
                    rows = int(logid.shape[0]) if hasattr(logid, "shape") else len(logid)
                    device = logid.device if torch.is_tensor(logid) else _object_device(batch)
                    return torch.zeros((rows, 1), dtype=torch.float32, device=device), torch.zeros((), device=device)
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


# =========================================================================
#  TensorRT helpers
# =========================================================================

def _try_compile_with_torch_compile(model: nn.Module) -> nn.Module:
    """Wrap model with torch.compile if available (PyTorch >= 2.0)."""
    if not hasattr(torch, "compile"):
        print("[INFO] torch.compile not available (requires PyTorch >= 2.0)")
        return model
    try:
        print("[INFO] applying torch.compile(mode='reduce-overhead') ...")
        compiled = torch.compile(model, mode="reduce-overhead")
        # warm-up
        return compiled
    except Exception as exc:
        print(f"[INFO] torch.compile failed: {exc}")
        return model


def _try_compile_with_tensorrt(model: nn.Module, device: torch.device) -> nn.Module | None:
    """Try torch-tensorrt compilation for the SMoE layers.

    Returns the optimized model, or None if TensorRT is not available.
    """
    if not _HAS_TORCH_TRT or device.type != "cuda":
        return None
    try:
        print("[INFO] attempting torch-tensorrt compilation ...")
        # Compile each SMoE module individually (they're the heaviest)
        if hasattr(model, "model") and hasattr(model.model, "seq_encoder"):
            seq = model.model.seq_encoder
            for moe_layer in seq.moe:
                try:
                    # Only compile the expert sub-modules — dynamic routing
                    # doesn't trace well through TensorRT
                    for j, expert in enumerate(moe_layer.experts):
                        compiled = torch_tensorrt.compile(
                            expert,
                            inputs=[
                                torch_tensorrt.Input(
                                    min_shape=(1, 512),
                                    opt_shape=(256, 512),
                                    max_shape=(8192, 512),
                                    dtype=torch.float32,
                                )
                            ],
                            enabled_precisions={torch.float32},
                        )
                        moe_layer.experts[j] = compiled
                except Exception:
                    pass  # If one expert fails, keep the original
            print("[INFO] torch-tensorrt compilation complete for expert modules")
        return model
    except Exception as exc:
        print(f"[INFO] torch-tensorrt compilation failed: {exc}")
        return None


# =========================================================================
#  Model loading
# =========================================================================

def load_model(
    ckpt_path: Path,
    gate_threshold=None,
    use_compile: bool = False,
    use_tensorrt: bool = False,
):
    """Load model from checkpoint with optional optimizations."""
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
                model = _build_official_model(gate_threshold=gate_threshold)
                _load_state_dict_strict(model, state_dict)

    if model is None:
        raise RuntimeError(
            "ckpt.pt did not contain a loadable torch.nn.Module or official model_state_dict. "
            "infer.py intentionally does not fall back to a random replacement model."
        )

    if gate_threshold is not None and hasattr(model, "seq_encoder"):
        if hasattr(model.seq_encoder, "set_gate_threshold"):
            model.seq_encoder.set_gate_threshold(gate_threshold)
            print(f"[INFO] SMoE gate_threshold = {gate_threshold}")

    print(f"[INFO] Loaded checkpoint from {ckpt_path} (epoch={epoch})")
    model.eval().to(device)
    adapter = ModelAdapter(model)

    if use_tensorrt and device.type == "cuda":
        trt_adapter = _try_compile_with_tensorrt(adapter, device)
        if trt_adapter is not None:
            adapter = trt_adapter

    if use_compile:
        adapter = _try_compile_with_torch_compile(adapter)

    return adapter, device


def _to_tensor_if_plain_numeric_sequence(value: Any):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, (int, bool, np.integer, np.bool_)) for x in value):
        return torch.tensor(value)
    return None


def move_batch_to_device(batch, device):
    _require_torch()
    list_metadata_keys = {
        "_active_slot_indices",
        "_empty_slot_indices",
        "_single_value_slot_indices",
        "_variable_slot_indices",
    }

    def move(obj: Any, key: Any = None):
        if key == "_has_predictions":
            return bool(obj)
        if key == "_num_predictions":
            return int(obj)
        if key in list_metadata_keys:
            return list(obj) if isinstance(obj, (list, tuple)) else obj
        if torch.is_tensor(obj):
            tensor = obj
        else:
            tensor = _to_tensor_if_plain_numeric_sequence(obj)
        if tensor is not None:
            if key in {"pred_mask", "attention_mask"}:
                tensor = tensor.bool()
            elif key in {"logid", "userid", "user_id", "feature_ids", "user_offsets", "_pred_index"} or isinstance(key, int):
                tensor = tensor.long()
            if key == "user_offsets":
                return tensor.cpu()
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


# =========================================================================
#  Inference entry point
# =========================================================================

def run_inference(
    code_root: Path,
    gate_threshold=None,
    use_compile: bool = False,
    use_tensorrt: bool = False,
    use_fp16: bool = False,
    use_pipeline: bool = True,
) -> Path:
    """Run full inference pipeline.

    When ``use_pipeline=True`` and CUDA is available, batches are processed
    through a *triple-chain* CUDA stream pipeline (三链DNA):

      Chain 1 (H2D stream) — prefetch next batch CPU→GPU
      Chain 2 (compute stream) — model forward
      Chain 3 (D2H stream) — extract predictions GPU→CPU

    The three chains overlap, keeping the GPU copy engines and compute
    engines busy simultaneously while the CPU prepares the next batch.
    """
    _require_torch()
    code_root = Path(code_root)
    shard_files = _find_cached_batch_files(code_root)
    if not shard_files:
        raise FileNotFoundError(f"cached batch shards not found under {code_root / 'dataset' / 'cached_batches'}")
    ckpt_path = code_root / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print(f"[INFO] loading cached batch shards from {shard_files[0].parent}")
    model, device = load_model(
        ckpt_path,
        gate_threshold=gate_threshold,
        use_compile=use_compile,
        use_tensorrt=use_tensorrt,
    )

    if use_fp16 and device.type == "cuda":
        model = model.half()
        print("[INFO] Model converted to FP16")

    print(f"[INFO] Model ready. Device: {device}")

    # Collect all valid batches first
    all_batches: List[Mapping[str, Any]] = []
    for shard_path in shard_files:
        batches = _load_cached_batches_from_files([shard_path])
        print(f"[INFO] loaded {len(batches)} batches from {shard_path.name}")
        for batch in batches:
            pred_mask_cpu = batch["pred_mask"].bool() if torch.is_tensor(batch["pred_mask"]) else torch.as_tensor(batch["pred_mask"], dtype=torch.bool)
            if not bool(pred_mask_cpu.any().item()):
                continue
            batch = dict(_prepare_official_batch(batch, max_slot_id=28))
            if batch.get("_has_predictions", True) is False:
                continue
            all_batches.append(batch)

    if not all_batches:
        print("[WARN] No batches with predictions found — writing empty output")
        output_path = code_root / "predict.txt"
        _write_prediction_values([], output_path)
        return output_path

    # Choose pipeline vs sequential
    if use_pipeline and device.type == "cuda" and len(all_batches) >= 2:
        predictions = _run_triple_pipeline(model, device, all_batches, use_fp16)
    else:
        if use_pipeline and device.type != "cuda":
            print("[INFO] Pipeline disabled — CUDA not available")
        predictions = _run_sequential(model, device, all_batches, use_fp16)

    output_path = code_root / "predict.txt"
    ordered_values = _order_prediction_values(predictions, code_root)
    _write_prediction_values(ordered_values, output_path)
    print(f"[INFO] predictions written to {output_path.name}, total: {len(ordered_values)}")
    return output_path


def _run_sequential(model, device, batches, use_fp16):
    """Process batches one at a time (fallback path)."""
    predictions: List[Tuple[int, float]] = []
    for batch in batches:
        moved = move_batch_to_device(batch, device)
        if use_fp16:
            moved = _to_half(moved)
        logits = _run_model(model, moved)
        pred_mask = moved["pred_mask"].bool()
        probs = torch.sigmoid(logits[pred_mask]).detach().cpu().float().tolist()
        logids = moved["logid"][pred_mask].detach().cpu().long().tolist()
        predictions.extend(zip(logids, probs))
    return predictions


def _run_triple_pipeline(model, device, batches, use_fp16):
    """三链DNA triple-CUDA-stream pipeline.

    Launches ALL GPU work (H2D → compute → D2H) for every batch using
    3 streams synchronised with CUDA events, then collects results once.

    The GPU hardware pipelines the three stages automatically:

      stream_h2d     : CPU→GPU  (copy engine)
      stream_compute : forward  (compute engine)
      stream_d2h     : GPU→CPU  (copy engine)

    Because modern GPUs have independent copy + compute engines, batches
    flow through the pipeline with data movement hidden behind compute.
    """
    n = len(batches)
    s_h2d = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()
    s_d2h = torch.cuda.Stream()

    gpu_batches: List[Any] = [None] * n
    logits_list: List[Any] = [None] * n
    cpu_probs: List[Any] = [None] * n
    cpu_logids: List[Any] = [None] * n

    # ---- Launch all GPU operations (non-blocking) -----------------------
    for i, batch in enumerate(batches):
        # Chain-1: H2D (copy engine)
        with torch.cuda.stream(s_h2d):
            gpu_batches[i] = move_batch_to_device(batch, device)
            if use_fp16:
                gpu_batches[i] = _to_half(gpu_batches[i])
        ev_h2d = s_h2d.record_event()

        # Chain-2: Compute (compute engine)
        s_comp.wait_event(ev_h2d)
        with torch.cuda.stream(s_comp):
            logits_list[i] = _run_model(model, gpu_batches[i])
        ev_comp = s_comp.record_event()

        # Chain-3: D2H (copy engine) — non_blocking so CPU is not stalled
        s_d2h.wait_event(ev_comp)
        with torch.cuda.stream(s_d2h):
            moved = gpu_batches[i]
            pred_mask = moved["pred_mask"].bool()
            cpu_probs[i] = torch.sigmoid(logits_list[i][pred_mask]).to(
                "cpu", non_blocking=True
            ).float()
            cpu_logids[i] = moved["logid"][pred_mask].to(
                "cpu", non_blocking=True
            ).long()

    # ---- Single sync point — all D2H transfers complete ----------------
    torch.cuda.synchronize()
    predictions: List[Tuple[int, float]] = []
    for i in range(n):
        if cpu_probs[i] is not None:
            predictions.extend(zip(cpu_logids[i].tolist(), cpu_probs[i].tolist()))
    return predictions


def _to_half(batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively convert float tensors to fp16."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dtype in (torch.float32, torch.float64):
            out[k] = v.half()
        elif isinstance(v, tuple):
            out[k] = tuple(
                x.half() if torch.is_tensor(x) and x.dtype in (torch.float32, torch.float64) else x
                for x in v
            )
        elif isinstance(v, Mapping):
            out[k] = _to_half(v)
        else:
            out[k] = v
    return out


# =========================================================================
#  Self-test
# =========================================================================

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


# =========================================================================
#  CLI
# =========================================================================

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run CTR inference with SMoE threshold & TensorRT optimizations."
    )
    parser.add_argument("code_root", nargs="?", default=str(Path(__file__).resolve().parent),
                        help="Root directory containing dataset/cached_batches/ and ckpt.pt")
    parser.add_argument("--self-test", action="store_true",
                        help="Run a quick self-test on the first cached shard")
    parser.add_argument("--gate-threshold", type=float, default=None,
                        help="SMoE threshold: 0.0=top-1 always, 0.5-0.99=adaptive, None=top-k")
    parser.add_argument("--compile", action="store_true", dest="use_compile",
                        help="Apply torch.compile for graph-level optimization")
    parser.add_argument("--tensorrt", action="store_true", dest="use_tensorrt",
                        help="Apply TensorRT to expert FFN layers (requires torch-tensorrt)")
    parser.add_argument("--fp16", action="store_true", dest="use_fp16",
                        help="Run inference in FP16 precision")
    parser.add_argument("--no-pipeline", action="store_true", dest="no_pipeline",
                        help="Disable triple-chain CUDA pipeline")
    parser.add_argument("--profile", action="store_true",
                        help="Print timing breakdown for each stage")
    args = parser.parse_args(argv)

    code_root = Path(args.code_root)

    if args.profile:
        _run_with_profiling(code_root, args)
    elif args.self_test:
        self_test(code_root)
    else:
        run_inference(
            code_root,
            gate_threshold=args.gate_threshold,
            use_compile=args.use_compile,
            use_tensorrt=args.use_tensorrt,
            use_fp16=args.use_fp16,
            use_pipeline=not args.no_pipeline,
        )


def _run_with_profiling(code_root: Path, args) -> None:
    """Run inference with per-stage timing."""
    import time

    _require_torch()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    shard_files = _find_cached_batch_files(code_root)
    print(f"[PROFILE] Found {len(shard_files)} shard files")

    ckpt_path = code_root / "ckpt.pt"
    t1 = time.perf_counter()
    model, device = load_model(
        ckpt_path,
        gate_threshold=args.gate_threshold,
        use_compile=args.use_compile,
        use_tensorrt=args.use_tensorrt,
    )
    if args.use_fp16 and device.type == "cuda":
        model = model.half()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t2 = time.perf_counter()
    print(f"[PROFILE] Model load: {t2 - t1:.3f}s")

    # Warm-up
    batches = _load_cached_batches_from_files([shard_files[0]])
    for batch in batches[:1]:
        pred_mask_cpu = batch["pred_mask"].bool() if torch.is_tensor(batch["pred_mask"]) else torch.as_tensor(batch["pred_mask"], dtype=torch.bool)
        if not bool(pred_mask_cpu.any().item()):
            continue
        batch = dict(_prepare_official_batch(batch, max_slot_id=28))
        moved = move_batch_to_device(batch, device)
        if args.use_fp16 and device.type == "cuda":
            moved = _to_half(moved)
        _ = _run_model(model, moved)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t3 = time.perf_counter()
    print(f"[PROFILE] Warm-up: {t3 - t2:.3f}s")

    # Timed run
    predictions: List[Tuple[int, float]] = []
    total_batches = 0
    infer_times = []
    for shard_path in shard_files:
        batches = _load_cached_batches_from_files([shard_path])
        for batch in batches:
            pred_mask_cpu = batch["pred_mask"].bool() if torch.is_tensor(batch["pred_mask"]) else torch.as_tensor(batch["pred_mask"], dtype=torch.bool)
            if not bool(pred_mask_cpu.any().item()):
                continue
            batch = dict(_prepare_official_batch(batch, max_slot_id=28))
            if batch.get("_has_predictions", True) is False:
                continue
            moved = move_batch_to_device(batch, device)
            if args.use_fp16 and device.type == "cuda":
                moved = _to_half(moved)

            t_b0 = time.perf_counter()
            logits = _run_model(model, moved)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_b1 = time.perf_counter()
            infer_times.append(t_b1 - t_b0)

            pred_mask = moved["pred_mask"].bool()
            probs = torch.sigmoid(logits[pred_mask]).detach().cpu().float().tolist()
            logids = moved["logid"][pred_mask].detach().cpu().long().tolist()
            predictions.extend(zip(logids, probs))
            total_batches += 1

    t4 = time.perf_counter()

    output_path = code_root / "predict.txt"
    ordered_values = _order_prediction_values(predictions, code_root)
    _write_prediction_values(ordered_values, output_path)

    t5 = time.perf_counter()

    print(f"\n[PROFILE] ========================================")
    print(f"[PROFILE]  Total batches:       {total_batches}")
    print(f"[PROFILE]  Total predictions:   {len(ordered_values)}")
    print(f"[PROFILE]  Model load:          {t2 - t1:.3f}s")
    print(f"[PROFILE]  Warm-up:             {t3 - t2:.3f}s")
    print(f"[PROFILE]  Inference (total):   {t4 - t3:.3f}s")
    if infer_times:
        print(f"[PROFILE]  Per-batch (mean):    {np.mean(infer_times) * 1000:.1f}ms")
        print(f"[PROFILE]  Per-batch (std):     {np.std(infer_times) * 1000:.1f}ms")
        print(f"[PROFILE]  Per-batch (min):     {np.min(infer_times) * 1000:.1f}ms")
        print(f"[PROFILE]  Per-batch (max):     {np.max(infer_times) * 1000:.1f}ms")
    print(f"[PROFILE]  Write output:        {t5 - t4:.3f}s")
    print(f"[PROFILE]  Total wall time:     {t5 - t0:.3f}s")
    print(f"[PROFILE] ========================================")


if __name__ == "__main__":
    main()
