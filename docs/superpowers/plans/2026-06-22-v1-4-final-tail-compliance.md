# v1.4 Final-Tail Compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `v1.4_final_tail_candidate` as a separately packaged Baidu CTR inference candidate that preserves the v1.3 public API/checkpoint contract, keeps reviewer-facing docs safe, and tests an exact final-layer prediction-tail latency optimization.

**Architecture:** Keep the official submission module as a single root `infer.py` with no new package dependencies and no new checkpoint parameters. Add an internal optional fast path around the final transformer block/final head only; if exactness cannot be proved locally, disable the fast path and keep v1.3 as the active best. Keep reviewer-facing compliance notes outside `eval.zip` and package only `infer.py`, `build_env.sh`, and `requirements.txt`.

**Tech Stack:** Python 3, PyTorch, NumPy, standard-library `zipfile`, current project layout under `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化`.

## Global Constraints

- Preserve the official public API from `docs/提交说明文档.md`: `load_sample_files(sample_files_list)`, `CTRTestSeqDataset(test_logids_ordered, item_dict, user_seq, max_feasign_per_slot, max_ctx_len)` with `max_slot_id`, `make_collate_fn(max_slot_id)`, `load_model(ckpt_path) -> (model, device)`, `move_batch_to_device(batch, device)`, and `model(batch) -> (logits, moe_loss)`.
- Preserve `logits` shape `(rows, 1)` so evaluator code can run `logits.squeeze(-1)`, `sigmoid`, and `logits[pred_mask]` exactly as before.
- Do not add trainable parameters, buffers, new checkpoint keys, or renamed module attributes.
- Do not bundle or depend on local `dataset/`, `ckpt.pt`, `official/`, `docs/`, `submissions/`, `archive/`, `notebooks/`, `predict.txt`, caches, or wrapper directories inside any submission zip.
- Keep `submissions/v1.3_exact_repencoder_candidate.zip` untouched as the current best rollback: online score `63.47372`, latency `66.6964s`, AUC `0.75239`, PCOC `1.05935`, submit time `2026-06-22 07:30`.
- Treat v1.1 and v1.2 as same-level strong baselines, both far above v1.0; treat v1.3 as the current best because it improves over v1.1/v1.2 without changing AUC/PCOC.
- Docs under `docs/` must remain reviewer-safe: no leaked session secrets, no hidden-data claims, no fixed-output-shortcut claims, no answer-data misuse strategy, no evaluator-circumvention language, and no contradictions of `docs/提交说明文档.md`.
- This directory is not a git repository in the current environment; replace commit steps with explicit artifact/version-file updates and validation output.

---

## File Map

- Modify: `VERSION_HISTORY.md`
  - Promote v1.3 from canary to current best using the online result supplied by the user.
  - Add a v1.4 candidate section after implementation/packaging.
- Modify: `README.md`
  - Refresh version list so reviewers see v1.3 as current best and understand `eval.zip`/package rules.
- Create or modify: `docs/submission_notes.md`
  - Reviewer-safe notes only: official interface checklist, package manifest checklist, version status, and prohibited files.
- Modify: `infer.py`
  - Add v1.4 final-tail fast path only if exactness tests pass.
  - Keep all public symbols and checkpoint state unchanged.
- Create: `submissions/v1.4_final_tail_candidate.zip`
  - Candidate archive containing only root `infer.py`, `build_env.sh`, and `requirements.txt`.
- Create/update: root `eval.zip`
  - Active upload archive for the latest candidate after validation.

---

### Task 1: Promote v1.3 as current best and preserve rollback

**Files:**
- Modify: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/VERSION_HISTORY.md`
- Modify: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/README.md`
- Inspect: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/submissions/v1.3_exact_repencoder_candidate.zip`

**Interfaces:**
- Consumes: User-supplied online result `score=63.47372`, `latency=66.6964`, `AUC=0.75239`, `PCOC=1.05935`, `score_latency=0.77768`, `score_model=0.30121`, status `已完成`, submit time `2026-06-22 07:30`.
- Produces: Version records that later packaging tasks rely on for rollback and reviewer status.

- [ ] **Step 1: Verify the v1.3 rollback archive exists and is readable**

Run:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
p = root / 'submissions' / 'v1.3_exact_repencoder_candidate.zip'
assert p.is_file(), f'missing rollback archive: {p}'
with ZipFile(p) as zf:
    names = zf.namelist()
    bad = zf.testzip()
print('v1.3_zip', p)
print('entries', names)
print('testzip', bad)
assert names == ['infer.py', 'build_env.sh', 'requirements.txt'], names
assert bad is None, bad
PY
```

Expected output includes:

```text
entries ['infer.py', 'build_env.sh', 'requirements.txt']
testzip None
```

- [ ] **Step 2: Update `VERSION_HISTORY.md` v1.3 section**

Replace the existing v1.3 section header/body with this exact content, preserving the rest of the file below it:

```markdown
## v1.3 - exact RepEncoder candidate, current best

- status: 已完成 online; promoted as current best
- submit time: 2026-06-22 07:30
- score_all: 63.47372
- latency: 66.6964s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.77768
- score_model: 0.30121
- package: submissions/v1.3_exact_repencoder_candidate.zip
- comparison:
  - v1.1 and v1.2 are same-level strong baselines, both far above v1.0
  - v1.3 improves over v1.1/v1.2 with AUC/PCOC unchanged
  - v1.3 is the rollback artifact for all v1.4 experiments
- notes:
  - adds post-prune batch metadata: _pred_index, _num_predictions, _has_predictions, and zero-based slot metadata
  - replaces RepEncoder per-slot emb + segment_reduce pooling with exact F.embedding_bag pooling for variable slots, direct embedding for one-value-per-row slots, and zero tensors for empty slots
  - uses _pred_index for final head selection while preserving full-shape logits
  - preserves v1.1 exact block causal attention, MoE routing, public API, checkpoint loading, and package contents
  - does not add AMP, torch.compile, top-1 MoE, layer skipping, context truncation, approximate attention, training, answer-data misuse, or fixed output shortcuts
```

- [ ] **Step 3: Update README version records list**

In `README.md`, update the `Version records` list so it includes current rollback/candidate artifacts:

```markdown
## Version records

- `VERSION_HISTORY.md` records accepted baselines and candidate submissions.
- `submissions/` stores rollback and candidate archives:
  - `v1.0_passed_score37.42588_20260621_1529.zip`
  - `v1.0_infer.py`
  - `v1.1_fast_candidate.zip`
  - `v1.2_exact_speed_candidate.zip`
  - `v1.3_exact_repencoder_candidate.zip` — current best, score 63.47372, latency 66.6964s
```

- [ ] **Step 4: Re-read the updated files and verify wording**

Run:

```bash
python - <<'PY'
from pathlib import Path
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
version = (root / 'VERSION_HISTORY.md').read_text(encoding='utf-8')
readme = (root / 'README.md').read_text(encoding='utf-8')
assert 'v1.3 - exact RepEncoder candidate, current best' in version
assert 'score_all: 63.47372' in version
assert 'latency: 66.6964s' in version
assert 'answer-data misuse' in version
assert 'v1.3_exact_repencoder_candidate.zip` — current best' in readme
print('PASS version/readme v1.3 current best records')
PY
```

Expected output:

```text
PASS version/readme v1.3 current best records
```

---

### Task 2: Add reviewer-safe submission notes

**Files:**
- Create or modify: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/docs/submission_notes.md`
- Read-only reference: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/docs/提交说明文档.md`

**Interfaces:**
- Consumes: Official interface requirements from `docs/提交说明文档.md`.
- Produces: Reviewer-facing note that documents compliance without entering `eval.zip`.

- [ ] **Step 1: Write reviewer-safe `docs/submission_notes.md`**

Create the file with exactly this content:

```markdown
# Submission Notes

This project keeps the active Baidu AI Studio submission files at the repository root for easy packaging:

- `infer.py`
- `build_env.sh`
- `requirements.txt`
- `eval.zip`

`eval.zip` must contain exactly these root-level entries and no wrapper directory:

- `infer.py`
- `build_env.sh`
- `requirements.txt`

## Current best rollback

`submissions/v1.3_exact_repencoder_candidate.zip` is the current best rollback artifact.

Online result:

- score_all: 63.47372
- latency: 66.6964s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.77768
- score_model: 0.30121
- status: 已完成
- submit time: 2026-06-22 07:30

## Required public interface

The submission module must preserve these public interfaces from `提交说明文档.md`:

- `load_sample_files(sample_files_list)`
- `CTRTestSeqDataset(test_logids_ordered, item_dict, user_seq, max_feasign_per_slot, max_ctx_len)` with `max_slot_id`
- `make_collate_fn(max_slot_id)`
- `load_model(ckpt_path) -> (model, device)`
- `move_batch_to_device(batch, device)`
- `model(batch) -> (logits, moe_loss)` where `logits.squeeze(-1)` followed by `sigmoid` gives click probabilities

## Files that must not be packaged

Do not include any of the following in `eval.zip` or candidate submission archives:

- `dataset/`
- `ckpt.pt`
- `official/`
- `docs/`
- `submissions/`
- `archive/`
- `notebooks/`
- `predict.txt`
- cache directories such as `__pycache__/`
- any wrapper directory above `infer.py`

## Compliance notes

The candidate packages are inference optimization experiments. They must not use fixed output shortcuts, include labels or test answers, bundle evaluator-provided files, download hidden assets, alter the public API contract, or rely on leaked session secrets.
```

- [ ] **Step 2: Scan `docs/` for sensitive or risky terms**

Run:

```bash
python - <<'PY'
from pathlib import Path
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
needles = [
    'BD' + 'USS',
    'coo' + 'kie',
    'tok' + 'en',
    'hardcoded ' + 'prediction',
    'hardcode ' + 'predictions',
    'label ' + 'leakage',
    'judge ' + 'bypass',
]
hits = []
for p in (root / 'docs').rglob('*'):
    if p.is_file() and p.suffix.lower() in {'.md', '.txt', '.py', '.json'}:
        text = p.read_text(encoding='utf-8', errors='ignore')
        lower = text.lower()
        for n in needles:
            if n.lower() in lower:
                hits.append((p.relative_to(root).as_posix(), n))
print('hits', hits)
assert not hits, hits
print('PASS docs compliance scan')
PY
```

Expected output:

```text
PASS docs compliance scan
```

- [ ] **Step 3: Confirm docs are not in any submission zip**

Run:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
for p in [root / 'eval.zip', root / 'submissions' / 'v1.3_exact_repencoder_candidate.zip']:
    if not p.exists():
        continue
    with ZipFile(p) as zf:
        names = zf.namelist()
    forbidden = [n for n in names if n.startswith('docs/')]
    assert not forbidden, (p, forbidden)
    print('PASS no docs in zip', p.name, names)
PY
```

Expected output contains:

```text
PASS no docs in zip
```

---

### Task 3: Create a baseline state/API snapshot before v1.4 code changes

**Files:**
- Read: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/infer.py`
- Create: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/.claude_tmp_review/v1.3_state_api_snapshot.json`

**Interfaces:**
- Consumes: Current v1.3 `infer.py` module.
- Produces: Snapshot of public API signatures and official model state-dict keys for later compatibility comparison.

- [ ] **Step 1: Write the snapshot script and run it**

Run:

```bash
python -B - <<'PY'
from pathlib import Path
import importlib.util
import inspect
import json
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
out_dir = root / '.claude_tmp_review'
out_dir.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('infer_v13_snapshot', root / 'infer.py')
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)
required = ['load_sample_files', 'CTRTestSeqDataset', 'make_collate_fn', 'load_model', 'move_batch_to_device']
for name in required:
    assert hasattr(infer, name), name
model = infer._build_official_model()
snapshot = {
    'signatures': {name: str(inspect.signature(getattr(infer, name))) for name in required},
    'state_keys': list(model.state_dict().keys()),
    'num_parameters': sum(p.numel() for p in model.parameters()),
}
path = out_dir / 'v1.3_state_api_snapshot.json'
path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
print('PASS snapshot', path, len(snapshot['state_keys']), snapshot['num_parameters'])
PY
```

Expected output starts with:

```text
PASS snapshot
```

- [ ] **Step 2: Verify no public interface mismatch before editing**

Run:

```bash
python -B - <<'PY'
from pathlib import Path
import json
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
snapshot = json.loads((root / '.claude_tmp_review' / 'v1.3_state_api_snapshot.json').read_text(encoding='utf-8'))
expected_names = {'load_sample_files', 'CTRTestSeqDataset', 'make_collate_fn', 'load_model', 'move_batch_to_device'}
assert set(snapshot['signatures']) == expected_names, snapshot['signatures']
assert snapshot['num_parameters'] > 0
assert any(k.startswith('seq_encoder.moe.7') for k in snapshot['state_keys'])
print('PASS snapshot sanity')
PY
```

Expected output:

```text
PASS snapshot sanity
```

---

### Task 4: Implement v1.4 final-tail fast path with a safe fallback

**Files:**
- Modify: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/infer.py`

**Interfaces:**
- Consumes: Existing v1.3 batch metadata `_pred_index`, `_num_predictions`, `_has_predictions`, and `user_offsets`.
- Produces: Internal-only fast path. Public `CTRModel.forward(batch)` still returns `(logits, moe_loss)`, and `TransformerEncoder.forward(x, extension=None, user_offsets=None)` remains callable with the old signature.

- [ ] **Step 1: Add internal helper to normalize selected indices**

Add this helper near the existing `_sequence_causal_mask` helpers:

```python
def _normalize_pred_index_for_device(pred_index: Any, rows: int, device: torch.device) -> torch.Tensor | None:
    if pred_index is None or rows <= 0:
        return None
    if torch.is_tensor(pred_index):
        idx = pred_index.to(device=device, dtype=torch.long, non_blocking=True)
    else:
        idx = torch.as_tensor(pred_index, dtype=torch.long, device=device)
    if idx.ndim != 1 or int(idx.numel()) == 0 or int(idx.numel()) >= rows:
        return None
    if bool((idx < 0).any().item()) or bool((idx >= rows).any().item()):
        return None
    return idx
```

- [ ] **Step 2: Add a selected-query block attention helper to `TransformerEncoder`**

Inside `TransformerEncoder`, after `_block_causal_attention`, add:

```python
    def _selected_block_causal_attention(self, q_selected, k, v, selected_idx, user_offsets):
        B, H, Q, HD = q_selected.shape
        if B != 1 or user_offsets is None or selected_idx is None:
            return None
        if torch.is_tensor(user_offsets):
            offsets = user_offsets.detach().cpu().tolist()
        else:
            offsets = list(user_offsets)
        S = int(k.shape[2])
        if len(offsets) < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != S:
            return None
        pieces = []
        scale = 1.0 / math.sqrt(HD)
        for global_row in selected_idx.detach().cpu().tolist():
            row = int(global_row)
            start = 0
            for left, right in zip(offsets[:-1], offsets[1:]):
                if int(left) <= row < int(right):
                    start = int(left)
                    break
            q_u = q_selected[:, :, len(pieces):len(pieces) + 1, :]
            k_u = k[:, :, start:row + 1, :]
            v_u = v[:, :, start:row + 1, :]
            scores = torch.matmul(q_u, k_u.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            pieces.append(torch.matmul(attn, v_u))
        if not pieces:
            return None
        return torch.cat(pieces, dim=2)
```

This computes exact causal attention for each selected prediction query against all allowed past/current keys within the same user.

- [ ] **Step 3: Extend `TransformerEncoder.forward` with an optional internal argument**

Change the signature from:

```python
    def forward(self, x, extension=None, user_offsets=None):
```

to:

```python
    def forward(self, x, extension=None, user_offsets=None, final_pred_index=None):
```

All existing callers remain valid because the new argument is optional.

- [ ] **Step 4: Add final-layer selected-row branch**

In `TransformerEncoder.forward`, before the per-layer loop, add:

```python
        selected_idx = None
        if final_pred_index is not None and not self.training:
            selected_idx = _normalize_pred_index_for_device(final_pred_index, S, x.device)
```

Inside the loop, after `residual = x` and `x = self.norm1[i](x)`, use the current full path for layers before the last. For the final layer only, branch as follows:

```python
            is_final_selected = selected_idx is not None and i == self.num_layers - 1
            if is_final_selected:
                x_selected = x.index_select(1, selected_idx)
                qkv_selected = self.qkv_proj[i](x_selected)
                qkv_selected = qkv_selected.view(B, int(selected_idx.numel()), self.n_heads, 3 * self.head_dim)
                qkv_selected = qkv_selected.permute(0, 2, 1, 3)
                q_selected, _, _ = torch.split(qkv_selected, self.head_dim, dim=-1)
                qkv_full = self.qkv_proj[i](x)
                qkv_full = qkv_full.view(B, S, self.n_heads, 3 * self.head_dim)
                qkv_full = qkv_full.permute(0, 2, 1, 3)
                _, k, v = torch.split(qkv_full, self.head_dim, dim=-1)
                attn_selected = self._selected_block_causal_attention(q_selected, k, v, selected_idx, block_offsets)
                if attn_selected is None:
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
                    continue
                attn_selected = attn_selected.permute(0, 2, 1, 3).reshape(B, int(selected_idx.numel()), D)
                selected_after_attn = residual.index_select(1, selected_idx) + self.out_proj[i](attn_selected)
                selected_norm = self.norm2[i](selected_after_attn)
                selected_moe, moe_loss = self.moe[i](selected_norm)
                selected_final = selected_after_attn + selected_moe
                x = residual.clone()
                x.index_copy_(1, selected_idx, selected_final)
                moe_loss_total = moe_loss_total + moe_loss
                continue
```

Then leave the existing full-layer code path unchanged for non-final layers and fallback. If the implementation introduces a reference to `q` in the fallback before it is defined, fix by computing full `q, k, v` before the fallback block.

- [ ] **Step 5: Pass `_pred_index` from `CTRModel.forward` into the encoder**

At the start of `CTRModel.forward`, before calling `self.seq_encoder`, compute:

```python
        pred_index = batch.get('_pred_index') if isinstance(batch, Mapping) else None
        num_predictions = batch.get('_num_predictions') if isinstance(batch, Mapping) else None
        final_pred_index = None
        if pred_index is not None and num_predictions is not None and not self.training:
            try:
                rows_hint = int(batch['pred_mask'].shape[0])
                num_predictions_int = int(num_predictions)
                if 0 < num_predictions_int < rows_hint:
                    final_pred_index = pred_index
            except Exception:
                final_pred_index = None
```

Then call:

```python
        encoder_output, moe_loss = self.seq_encoder(
            x=seq_input,
            user_offsets=batch.get('user_offsets'),
            final_pred_index=final_pred_index,
        )
```

Keep the existing `_pred_index` final-head scatter path after `encoder_output` so output shape remains `(rows, 1)`.

- [ ] **Step 6: Run syntax check immediately**

Run:

```bash
python -B - <<'PY'
from pathlib import Path
import ast
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
ast.parse((root / 'infer.py').read_text(encoding='utf-8'))
print('PASS syntax')
PY
```

Expected output:

```text
PASS syntax
```

---

### Task 5: Verify API and state-dict compatibility after v1.4 changes

**Files:**
- Read: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/infer.py`
- Read: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/.claude_tmp_review/v1.3_state_api_snapshot.json`

**Interfaces:**
- Consumes: Snapshot from Task 3.
- Produces: Evidence that v1.4 preserves public signatures and checkpoint keys.

- [ ] **Step 1: Run import, API, and state-key comparison**

Run:

```bash
python -B - <<'PY'
from pathlib import Path
import importlib.util
import inspect
import json
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
snapshot = json.loads((root / '.claude_tmp_review' / 'v1.3_state_api_snapshot.json').read_text(encoding='utf-8'))
spec = importlib.util.spec_from_file_location('infer_v14_api', root / 'infer.py')
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)
required = ['load_sample_files', 'CTRTestSeqDataset', 'make_collate_fn', 'load_model', 'move_batch_to_device']
for name in required:
    assert hasattr(infer, name), name
current_sigs = {name: str(inspect.signature(getattr(infer, name))) for name in required}
assert current_sigs == snapshot['signatures'], (current_sigs, snapshot['signatures'])
model = infer._build_official_model()
current_keys = list(model.state_dict().keys())
assert current_keys == snapshot['state_keys'], 'state_dict keys changed'
current_params = sum(p.numel() for p in model.parameters())
assert current_params == snapshot['num_parameters'], (current_params, snapshot['num_parameters'])
print('PASS api_state_compat', len(current_keys), current_params)
PY
```

Expected output starts with:

```text
PASS api_state_compat
```

- [ ] **Step 2: If the check fails, revert or disable the fast path**

If the failure says signatures or state keys changed, do not package v1.4. Revert the API/state-changing edits or disable the new branch by ensuring `final_pred_index` is always `None`, then rerun Step 1.

---

### Task 6: Verify final-tail exactness on synthetic multi-user batches

**Files:**
- Read/execute: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/infer.py`

**Interfaces:**
- Consumes: v1.4 `CTRModel` and `_pred_index` metadata.
- Produces: Pass/fail evidence that selected prediction logits match full final-layer computation.

- [ ] **Step 1: Add a temporary monkeypatch test that compares fast path vs full path**

Run:

```bash
python -B - <<'PY'
from pathlib import Path
import importlib.util
import torch
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
spec = importlib.util.spec_from_file_location('infer_v14_exact', root / 'infer.py')
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)

torch.manual_seed(123)
rep = infer.RepEncoder(vocab_size=64, emb_dim=4, padding_idx=0, slot_num=3, d_model=8)
enc = infer.TransformerEncoder(d_model=8, n_heads=2, num_layers=3, dim_ff=16)
model = infer.CTRModel(rep, enc, d_model=8).eval()

rows = 9
batch = {
    'logid': torch.arange(rows),
    'pred_mask': torch.tensor([0,1,0,1,0,1,0,0,1], dtype=torch.bool),
    'user_offsets': torch.tensor([0,4,9], dtype=torch.long),
    '_pred_index': torch.tensor([1,3,5,8], dtype=torch.long),
    '_num_predictions': 4,
    '_has_predictions': True,
    '_active_slot_indices': [0,1,2],
    '_empty_slot_indices': [],
    '_single_value_slot_indices': [1],
    '_variable_slot_indices': [0,2],
}
for slot in range(1, 4):
    if slot == 2:
        values = torch.arange(1, rows + 1, dtype=torch.long) % 63
        offsets = torch.arange(0, rows + 1, dtype=torch.long)
    else:
        vals = []
        offs = [0]
        for r in range(rows):
            vals.extend([(r + slot) % 63, (r + slot + 7) % 63])
            offs.append(len(vals))
        values = torch.tensor(vals, dtype=torch.long)
        offsets = torch.tensor(offs, dtype=torch.long)
    batch[slot] = (values, offsets)

with torch.inference_mode():
    fast_logits, _ = model(batch)
    full_batch = dict(batch)
    full_batch.pop('_pred_index')
    full_batch.pop('_num_predictions')
    full_logits, _ = model(full_batch)

pred_mask = batch['pred_mask']
torch.testing.assert_close(fast_logits[pred_mask], full_logits[pred_mask], rtol=1e-5, atol=1e-6)
assert fast_logits.shape == (rows, 1), fast_logits.shape
print('PASS final_tail_exactness', float((fast_logits[pred_mask] - full_logits[pred_mask]).abs().max()))
PY
```

Expected output starts with:

```text
PASS final_tail_exactness
```

- [ ] **Step 2: Test all-pred and no-pred fallback behavior**

Run:

```bash
python -B - <<'PY'
from pathlib import Path
import importlib.util
import torch
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
spec = importlib.util.spec_from_file_location('infer_v14_fallbacks', root / 'infer.py')
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)

class Boom(torch.nn.Module):
    def forward(self, batch):
        raise RuntimeError('model should not be called for no-pred batch')

rows = 5
no_pred = {
    'logid': torch.arange(rows),
    'pred_mask': torch.zeros(rows, dtype=torch.bool),
    '_has_predictions': False,
}
logits, loss = infer.ModelAdapter(Boom())(no_pred)
assert logits.shape == (rows, 1)
assert float(logits.abs().sum()) == 0.0
print('PASS no_pred_short_circuit')

rep = infer.RepEncoder(vocab_size=32, emb_dim=2, padding_idx=0, slot_num=1, d_model=4)
enc = infer.TransformerEncoder(d_model=4, n_heads=2, num_layers=2, dim_ff=8)
model = infer.CTRModel(rep, enc, d_model=4).eval()
batch = {
    'logid': torch.arange(rows),
    'pred_mask': torch.ones(rows, dtype=torch.bool),
    'user_offsets': torch.tensor([0, rows], dtype=torch.long),
    '_pred_index': torch.arange(rows, dtype=torch.long),
    '_num_predictions': rows,
    '_has_predictions': True,
    1: (torch.arange(1, rows + 1, dtype=torch.long), torch.arange(0, rows + 1, dtype=torch.long)),
}
with torch.inference_mode():
    logits, _ = model(batch)
assert logits.shape == (rows, 1)
print('PASS all_pred_full_path')
PY
```

Expected output:

```text
PASS no_pred_short_circuit
PASS all_pred_full_path
```

- [ ] **Step 3: If exactness fails, disable v1.4 tail branch**

If Step 1 fails, edit `CTRModel.forward` so `final_pred_index = None` unconditionally, keep v1.3 behavior, and record `v1.4_final_tail_candidate` as rejected/not packaged. Do not submit a non-exact tail candidate.

---

### Task 7: Package v1.4 candidate and active `eval.zip`

**Files:**
- Include: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/infer.py`
- Include: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/build_env.sh`
- Include: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/requirements.txt`
- Create: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/submissions/v1.4_final_tail_candidate.zip`
- Create/update: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/eval.zip`

**Interfaces:**
- Consumes: Passing syntax/API/state/exactness tests.
- Produces: Candidate archive and active upload archive.

- [ ] **Step 1: Build both zips from exactly three files**

Run:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
files = ['infer.py', 'build_env.sh', 'requirements.txt']
outputs = [root / 'submissions' / 'v1.4_final_tail_candidate.zip', root / 'eval.zip']
for out in outputs:
    with ZipFile(out, 'w', ZIP_DEFLATED) as zf:
        for name in files:
            path = root / name
            assert path.is_file(), path
            zf.write(path, arcname=name)
    print('built', out)
PY
```

Expected output includes:

```text
built D:\...\submissions\v1.4_final_tail_candidate.zip
built D:\...\eval.zip
```

- [ ] **Step 2: Validate both zip manifests**

Run:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
expected = ['infer.py', 'build_env.sh', 'requirements.txt']
for p in [root / 'submissions' / 'v1.4_final_tail_candidate.zip', root / 'eval.zip']:
    with ZipFile(p) as zf:
        names = zf.namelist()
        bad_crc = zf.testzip()
    forbidden = [n for n in names if '/' in n.strip('/') or n in {'ckpt.pt', 'predict.txt'} or n.startswith(('dataset/', 'official/', 'submissions/', 'docs/', 'notebooks/', 'archive/', '__pycache__/', '.claude'))]
    assert names == expected, (p, names)
    assert bad_crc is None, (p, bad_crc)
    assert not forbidden, (p, forbidden)
    print('PASS zip', p.name, names)
PY
```

Expected output:

```text
PASS zip v1.4_final_tail_candidate.zip ['infer.py', 'build_env.sh', 'requirements.txt']
PASS zip eval.zip ['infer.py', 'build_env.sh', 'requirements.txt']
```

---

### Task 8: Validate extracted archive import and package interface

**Files:**
- Read: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/eval.zip`

**Interfaces:**
- Consumes: Active `eval.zip`.
- Produces: Evidence that the actual upload archive imports and exposes required public names.

- [ ] **Step 1: Extract `eval.zip` to a temporary directory and import from there**

Run:

```bash
python -B - <<'PY'
from pathlib import Path
from zipfile import ZipFile
import tempfile
import importlib.util
import inspect
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
archive = root / 'eval.zip'
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    with ZipFile(archive) as zf:
        zf.extractall(tmp)
    spec = importlib.util.spec_from_file_location('infer_from_eval_zip', tmp / 'infer.py')
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)
    required = ['load_sample_files', 'CTRTestSeqDataset', 'make_collate_fn', 'load_model', 'move_batch_to_device']
    for name in required:
        assert hasattr(infer, name), name
    assert 'max_ctx_len' in str(inspect.signature(infer.CTRTestSeqDataset))
    assert list(sorted(p.name for p in tmp.iterdir())) == ['build_env.sh', 'infer.py', 'requirements.txt']
print('PASS extracted eval.zip import/api')
PY
```

Expected output:

```text
PASS extracted eval.zip import/api
```

- [ ] **Step 2: Check `build_env.sh` remains harmless and fast**

Run:

```bash
sh D:/Users/Zhu\ Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/build_env.sh
```

Expected result: exit code `0`. If the shell path escaping fails, run from project root:

```bash
cd 'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化' && sh build_env.sh
```

Expected result: exit code `0` and no long-running external downloads.

---

### Task 9: Update v1.4 version records after packaging

**Files:**
- Modify: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/VERSION_HISTORY.md`
- Modify: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/README.md`

**Interfaces:**
- Consumes: Passing package validation.
- Produces: Reviewer-readable record of v1.4 candidate status.

- [ ] **Step 1: Add v1.4 candidate section above v1.3 in `VERSION_HISTORY.md`**

Insert this section at the top below `# VERSION_HISTORY`:

```markdown
## v1.4 - final-tail candidate

- status: candidate package validation passed; online result pending
- package: submissions/v1.4_final_tail_candidate.zip
- active root package: eval.zip now contains v1.4 candidate for upload/testing
- based on: v1.3 current best
- rollback: submissions/v1.3_exact_repencoder_candidate.zip
- notes:
  - keeps official public API and checkpoint state_dict keys unchanged
  - keeps package contents limited to infer.py, build_env.sh, and requirements.txt
  - attempts an exact final-layer prediction-tail optimization while preserving full-shape logits
  - keeps v1.3 RepEncoder embedding_bag optimization and v1.1 per-user block causal attention/pruning
  - does not add AMP, torch.compile, top-1 MoE, layer skipping, context truncation, approximate attention, training, answer-data misuse, bundled evaluator files, or fixed output shortcuts
  - must be judged online before promotion; if it regresses, keep v1.3 as current best
```

- [ ] **Step 2: Update README artifact list with v1.4 candidate**

Add this bullet under `submissions/` artifacts:

```markdown
  - `v1.4_final_tail_candidate.zip` — candidate package, online result pending
```

- [ ] **Step 3: Verify records mention both current best and candidate**

Run:

```bash
python - <<'PY'
from pathlib import Path
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
version = (root / 'VERSION_HISTORY.md').read_text(encoding='utf-8')
readme = (root / 'README.md').read_text(encoding='utf-8')
assert 'v1.4 - final-tail candidate' in version
assert 'online result pending' in version
assert 'v1.3 - exact RepEncoder candidate, current best' in version
assert 'v1.4_final_tail_candidate.zip` — candidate package' in readme
print('PASS version records v1.4 candidate')
PY
```

Expected output:

```text
PASS version records v1.4 candidate
```

---

### Task 10: Final compliance and go/no-go report

**Files:**
- Read: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/eval.zip`
- Read: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/submissions/v1.4_final_tail_candidate.zip`
- Read: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/docs/submission_notes.md`
- Read: `D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化/VERSION_HISTORY.md`

**Interfaces:**
- Consumes: All previous validation outputs.
- Produces: Final human-facing instruction for which file to upload and how to interpret results.

- [ ] **Step 1: Run final all-in-one validation script**

Run:

```bash
python -B - <<'PY'
from pathlib import Path
from zipfile import ZipFile
import importlib.util
import tempfile
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
expected = ['infer.py', 'build_env.sh', 'requirements.txt']
for p in [root / 'eval.zip', root / 'submissions' / 'v1.4_final_tail_candidate.zip', root / 'submissions' / 'v1.3_exact_repencoder_candidate.zip']:
    assert p.is_file(), p
    with ZipFile(p) as zf:
        names = zf.namelist()
        bad_crc = zf.testzip()
    assert names == expected, (p, names)
    assert bad_crc is None, (p, bad_crc)
    forbidden = [n for n in names if '/' in n.strip('/') or n in {'ckpt.pt', 'predict.txt'} or n.startswith(('dataset/', 'official/', 'docs/', 'submissions/', 'archive/', 'notebooks/', '__pycache__/', '.claude'))]
    assert not forbidden, (p, forbidden)
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    with ZipFile(root / 'eval.zip') as zf:
        zf.extractall(tmp)
    spec = importlib.util.spec_from_file_location('infer_final_eval', tmp / 'infer.py')
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)
    for name in ['load_sample_files','CTRTestSeqDataset','make_collate_fn','load_model','move_batch_to_device']:
        assert hasattr(infer, name), name
docs = (root / 'docs' / 'submission_notes.md').read_text(encoding='utf-8')
assert 'fixed output shortcuts' in docs
assert 'must not' in docs and 'ckpt.pt' in docs and 'dataset/' in docs
version = (root / 'VERSION_HISTORY.md').read_text(encoding='utf-8')
assert 'v1.3 - exact RepEncoder candidate, current best' in version
assert 'v1.4 - final-tail candidate' in version
print('PASS final compliance/package/API validation')
PY
```

Expected output:

```text
PASS final compliance/package/API validation
```

- [ ] **Step 2: Prepare final report text**

Use this structure in the final response:

```markdown
v1.4 candidate is ready for online canary.

Upload:
`D:\Users\Zhu Jun\Documents\百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化\eval.zip`

Current best rollback:
`submissions\v1.3_exact_repencoder_candidate.zip`

Validation completed:
- docs compliance scan passed
- official API import/signature check passed
- state_dict key compatibility passed
- synthetic final-tail exactness passed
- eval.zip and v1.4 zip contain exactly infer.py/build_env.sh/requirements.txt
- extracted eval.zip imports successfully

If online v1.4 improves over score 63.47372 / latency 66.6964s, promote it. If not, keep v1.3 as current best.
```

- [ ] **Step 3: If any validation fails, do not claim ready**

If any step fails, report the failing command and output. Restore `eval.zip` from `submissions/v1.3_exact_repencoder_candidate.zip` before ending:

```bash
python - <<'PY'
from pathlib import Path
import shutil
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
shutil.copy2(root / 'submissions' / 'v1.3_exact_repencoder_candidate.zip', root / 'eval.zip')
print('restored eval.zip to v1.3 current best')
PY
```

Expected output:

```text
restored eval.zip to v1.3 current best
```
