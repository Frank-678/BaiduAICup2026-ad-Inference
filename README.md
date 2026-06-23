# 生成式推荐广告排序推理性能优化

## Root submission package

Keep only the official submission entry files at project root:

- `infer.py`
- `build_env.sh`
- `requirements.txt`
- `eval.zip`

`eval.zip` must contain exactly these three files and nothing else:

- `infer.py`
- `build_env.sh`
- `requirements.txt`

## Version records

- `VERSION_HISTORY.md` records accepted baselines and candidate submissions.
- `docs/version_results.md` records online score/latency/AUC/PCOC results for each submitted version.
- `submissions/` stores rollback and candidate archives:
  - `v1.0_passed_score37.42588_20260621_1529.zip`
  - `v1.0_infer.py`
  - `v1.1_fast_candidate.zip`
  - `v1.2_exact_speed_candidate.zip`
  - `v1.3_exact_repencoder_candidate.zip` — conservative rollback, score 63.47372, latency 66.6964s
  - `v1.4_sync_metadata_candidate.zip` — metadata/sync cleanup, not promoted
  - `v1.5_bf16_canary.zip` — precision-risk canary, not promoted
  - `v1.6_tf32_sync_metadata_candidate.zip` — TF32/sync candidate, not promoted
  - `v1.7_smoe_all_experts_candidate.zip` — SMoE dispatch mainline, score 64.10920, latency 63.97291s
  - `v1.7_smoe_threshold_4096_candidate.zip` — previous online best, score 65.10414, latency 59.70853s
  - `v1.7_smoe_threshold_6144_candidate.zip` — packaged threshold sweep candidate
  - `v1.7_smoe_threshold_8192_candidate.zip` — previous online best, score 66.19630, latency 55.02794s
  - `v1.7_smoe_threshold_12288_candidate.zip` — packaged threshold sweep candidate
  - `v1.7_smoe_always_on_candidate.zip` — previous online best, score 66.34358, latency 54.39675s
  - `v1.7_smoe_always_on_prewarm_candidate.zip` — previous online best, score 66.35171, latency 54.36190s
  - `v1.9_smoe_prewarm_guarded_attention_min8_cap128_candidate.zip` — previous online best, score 66.37894, latency 54.24521s
  - `v1.9b_smoe_prewarm_guarded_attention_min4_cap128_candidate.zip` — previous online best, score 66.44197, latency 53.97509s
  - `v1.9c_smoe_prewarm_guarded_attention_min8_cap256_candidate.zip` — packaged aggressive attention cap sweep
  - `v1.9d_smoe_prewarm_guarded_attention_min4_cap256_candidate.zip` — not promoted, score 66.41995, latency 54.06944s
  - `v1.9e_smoe_prewarm_guarded_attention_min2_cap128_candidate.zip` — current online best and root `eval.zip`, score 66.44569, latency 53.95916s
  - `v1.9f_smoe_prewarm_guarded_attention_selective_bf16_candidate.zip` — packaged precision-risk canary, not first choice
  - `v2.2_cuda_varlen_flash_attention_final_candidate.zip` — CUDA/FlashAttention varlen canary, not promoted, score 63.77724, latency 65.39537s

## Project tree

- `docs/`: submission notes and competition documents.
- `docs/references/`: reference PDFs.
- `notebooks/`: exploratory notebooks.
- `archive/old_attempts/`: older attempt files not used by the current package.
- `official/`: official reference tree, left in place.

## Zip safety

Before submitting, verify `eval.zip` contains only `infer.py`, `build_env.sh`, and `requirements.txt`; do not include notebooks, docs, caches, checkpoints, shards, or archive files.
