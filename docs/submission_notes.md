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
