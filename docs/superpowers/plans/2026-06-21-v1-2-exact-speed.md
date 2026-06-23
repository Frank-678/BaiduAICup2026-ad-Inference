# v1.2 Exact-Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce online model-forward latency beyond v1.1 while preserving AUC/PCOC and competition legality.

**Architecture:** Keep v1.1's successful exact optimizations as the base: per-user block causal SDPA, exact future-row pruning, MoE gate logsumexp, and inference-mode adapter. v1.2 adds low-risk exact micro-optimizations only: final linear on prediction rows with full-shape scatter where possible, pruning marker cleanup, version/package records, and validation. Avoid top-1 MoE, layer skipping, fixed-window attention, feature truncation, or logit calibration by default.

**Tech Stack:** Python, PyTorch, NumPy, standard-library zipfile. No new dependencies.

## Global Constraints

- Current accepted v1.0 rollback remains `submissions/v1.0_passed_score37.42588_20260621_1529.zip`.
- Current v1.1 online result: `score=60.77022`, `latency=78.28285`, `auc=0.75239`, `pcoc=1.05935`.
- Final submission archive must contain exactly `infer.py`, `build_env.sh`, `requirements.txt` at archive root.
- Final submission archive must not contain `dataset/`, `ckpt.pt`, `official/`, `submissions/`, `docs/`, `notebooks/`, `archive/`, `predict.txt`, `__pycache__/`, or a wrapper directory.
- Do not train on or modify test data.
- Do not use fixed output shortcuts.
- Do not download or assemble local `ckpt.pt` unless the user explicitly changes strategy.
- Do not change default model semantics with top-1 MoE, layer skipping, windowed attention, context truncation, slot-value budgets, or logit calibration in v1.2.

---

## File Structure

- Modify `infer.py` only for runtime code.
  - `CTRModel.forward`: compute final linear only for prediction rows when `pred_mask` exists, scatter logits back to full row shape.
  - `ModelAdapter.forward`: preserve no-prediction short-circuit and inference mode.
  - `move_batch_to_device`: preserve `_has_predictions` bool marker, keep `user_offsets` CPU.
- Modify `VERSION_HISTORY.md` after packaging.
  - Add v1.2 candidate metadata, package path, and known validation status.
- Create/update `submissions/v1.2_exact_speed_candidate.zip` after packaging.
- Keep `build_env.sh` and `requirements.txt` unchanged.

---

### Task 1: Final-Linear Pred-Only Fast Path

**Files:**
- Modify: `infer.py`

**Interfaces:**
- Consumes: `batch["pred_mask"]` as bool tensor/list and `encoder_output` shaped `[rows, d_model]`.
- Produces: `CTRModel.forward(batch) -> (pred_logits, moe_loss)` with `pred_logits` still shaped `[rows, 1]` so evaluator behavior remains unchanged.

- [ ] **Step 1: Add helper inside `CTRModel.forward` after `encoder_output` is flattened**

Replace current final head block:

```python
pred = self.linear(encoder_output)
pred_logits = torch.clamp(pred, min=-15.0, max=15.0)
return pred_logits, moe_loss
```

with:

```python
pred_mask = batch.get("pred_mask") if isinstance(batch, Mapping) else None
if pred_mask is not None:
    mask = pred_mask.bool() if torch.is_tensor(pred_mask) else torch.as_tensor(pred_mask, dtype=torch.bool, device=encoder_output.device)
    if mask.device != encoder_output.device:
        mask = mask.to(encoder_output.device, non_blocking=True)
    if mask.ndim == 1 and int(mask.numel()) == int(encoder_output.shape[0]) and bool(mask.any().item()) and int(mask.sum().item()) < int(mask.numel()):
        pred_logits = encoder_output.new_zeros((encoder_output.shape[0], 1))
        pred_logits[mask] = torch.clamp(self.linear(encoder_output[mask]), min=-15.0, max=15.0)
        return pred_logits, moe_loss
pred = self.linear(encoder_output)
pred_logits = torch.clamp(pred, min=-15.0, max=15.0)
return pred_logits, moe_loss
```

Rationale: evaluator later indexes `logits[pred_mask]`; non-pred logits are irrelevant. The full output shape is preserved.

- [ ] **Step 2: Validate no-pred and mixed-pred shapes with a tiny synthetic module**

Run:

```bash
python -B - <<'PY'
from pathlib import Path
import importlib.util, torch
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
spec = importlib.util.spec_from_file_location('infer_v12', root/'infer.py')
infer = importlib.util.module_from_spec(spec); spec.loader.exec_module(infer)
# Full imported API still present
for name in ['load_sample_files','CTRTestSeqDataset','make_collate_fn','load_model','move_batch_to_device']:
    assert hasattr(infer, name), name
print('api_ok')
PY
```

Expected output includes `api_ok`.

---

### Task 2: Lightweight Regression Validation

**Files:**
- Read/execute: `infer.py`

**Interfaces:**
- Consumes: current `infer.py` required functions.
- Produces: terminal evidence that v1.2 keeps v1.1 invariants for synthetic data.

- [ ] **Step 1: Run synthetic invariant validation**

Run:

```bash
python -B - <<'PY'
from pathlib import Path
import ast, importlib.util, inspect
import torch
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
ast.parse((root/'infer.py').read_text(encoding='utf-8'))
spec = importlib.util.spec_from_file_location('infer_v12_check', root/'infer.py')
infer = importlib.util.module_from_spec(spec); spec.loader.exec_module(infer)
print('syntax_import_ok')
print('load_model_signature', inspect.signature(infer.load_model))
# no-pred marker survives collate and adapter short-circuits
batch0 = {'logid': torch.arange(3), 'pred_mask': torch.zeros(3, dtype=torch.bool), 'user_offsets': torch.tensor([0,3])}
for slot in range(1,29): batch0[slot] = (torch.arange(3), torch.arange(4))
coll0 = infer.make_collate_fn(28)([batch0])
assert coll0.get('_has_predictions') is False
moved0 = infer.move_batch_to_device(coll0, torch.device('cpu'))
class Boom(torch.nn.Module):
    def forward(self, batch): raise RuntimeError('should short-circuit')
logits, loss = infer.ModelAdapter(Boom())(moved0)
assert logits.shape == (3,1) and float(logits.abs().sum()) == 0.0
print('no_prediction_short_circuit_ok')
# cached prune/collate
total = 10
batch = {'userid': torch.arange(total), 'logid': torch.arange(100,110), 'label': torch.zeros(total), 'pred_mask': torch.tensor([0,1,0,0, 0,0,1, 0,0,0], dtype=torch.bool), 'user_offsets': torch.tensor([0,4,7,10])}
for slot in range(1,29): batch[slot] = (torch.arange(slot*100, slot*100+total), torch.arange(0,total+1))
collated = infer.make_collate_fn(28)([batch])
assert collated.get('_has_predictions') is True
assert collated['logid'].tolist() == [100,101,104,105,106]
assert collated['pred_mask'].tolist() == [False, True, False, False, True]
assert collated['user_offsets'].tolist() == [0,2,5]
print('cached_collate_prune_ok')
# block attention equivalence on tiny tensors
torch.manual_seed(0)
enc = infer.TransformerEncoder(d_model=8, n_heads=2, num_layers=1, dim_ff=16).eval()
q = torch.randn(1,2,7,4); k = torch.randn(1,2,7,4); v = torch.randn(1,2,7,4)
offsets = torch.tensor([0,3,7])
mask = infer._sequence_causal_mask(offsets).unsqueeze(0).unsqueeze(0)
dense = infer.scaled_dot_product(q,k,v, {'mask': mask})
block = enc._block_causal_attention(q,k,v, offsets.tolist())
assert (dense-block).abs().max().item() < 1e-5
print('block_attention_equivalence_ok')
# gate equivalence
gate = infer.TopKGate(8,8,2).eval(); x = torch.randn(2,5,8)
with torch.no_grad():
    logits = gate.w_g(x); probs = torch.softmax(logits, dim=-1); rs, ri = torch.topk(probs, 2, dim=-1); i,s = gate(x)
assert torch.equal(i, ri)
assert (s-rs).abs().max().item() < 1e-6
print('gate_equivalence_ok')
print('v1_2_lightweight_validation_ok')
PY
```

Expected output includes `v1_2_lightweight_validation_ok`.

---

### Task 3: Package and Version v1.2 Candidate

**Files:**
- Modify: `VERSION_HISTORY.md`
- Create/update: `eval.zip`
- Create/update: `submissions/v1.2_exact_speed_candidate.zip`

**Interfaces:**
- Consumes: root `infer.py`, `build_env.sh`, `requirements.txt`.
- Produces: root-flat `eval.zip` ready for upload and versioned copy.

- [ ] **Step 1: Rebuild `eval.zip` from exactly three files**

Run:

```bash
python - <<'PY'
from pathlib import Path
import zipfile, hashlib, shutil
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
required = ['infer.py','build_env.sh','requirements.txt']
archive = root/'eval.zip'
if archive.exists(): archive.unlink()
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for name in required:
        zf.write(root/name, arcname=name)
with zipfile.ZipFile(archive) as zf:
    names = zf.namelist(); bad = zf.testzip()
assert names == required, names
assert bad is None, bad
sub = root/'submissions'; sub.mkdir(exist_ok=True)
copy = sub/'v1.2_exact_speed_candidate.zip'
shutil.copy2(archive, copy)
print('eval_zip_sha256', hashlib.sha256(archive.read_bytes()).hexdigest())
print('v1_2_copy', copy)
print('entries', names)
print('testzip', bad)
PY
```

Expected output includes `entries ['infer.py', 'build_env.sh', 'requirements.txt']` and `testzip None`.

- [ ] **Step 2: Update `VERSION_HISTORY.md`**

Prepend:

```markdown
## v1.2 - exact speed candidate

- status: candidate package validation passed
- package: submissions/v1.2_exact_speed_candidate.zip
- based on: v1.1 online accepted candidate
- v1.1 online metrics: score=60.77022, latency=78.28285s, AUC=0.75239, PCOC=1.05935
- notes:
  - preserves v1.1 exact per-user block causal attention
  - preserves exact prediction pruning and no-prediction short-circuit
  - adds pred-mask-only final linear fast path with full-shape scatter
  - no local ckpt/full-shard runtime validation
```

- [ ] **Step 3: Final archive validation**

Run:

```bash
python - <<'PY'
from pathlib import Path
import zipfile
root = Path(r'D:/Users/Zhu Jun/Documents/百度商业AI技术创新大赛-生成式推荐广告排序推理性能优化')
with zipfile.ZipFile(root/'eval.zip') as zf:
    names = zf.namelist(); bad = zf.testzip()
forbidden = [n for n in names if '/' in n.strip('/') or n in {'ckpt.pt','predict.txt'} or n.startswith(('dataset/','official/','submissions/','docs/','notebooks/','archive/','__pycache__/'))]
print('names', names)
print('bad', bad)
print('forbidden', forbidden)
assert names == ['infer.py','build_env.sh','requirements.txt']
assert bad is None
assert not forbidden
print('final_archive_ok')
PY
```

Expected output includes `final_archive_ok`.

---

## Self-Review

- Spec coverage: v1.2 only includes exact/near-exact legal optimizations; risky approximate changes are explicitly excluded.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: public required interfaces remain unchanged; `CTRModel.forward` still returns full-shape logits.
- Risk note: full online latency/AUC/PCOC still requires judge environment because local `ckpt.pt` is intentionally absent.
