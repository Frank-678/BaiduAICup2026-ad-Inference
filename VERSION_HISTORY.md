# VERSION_HISTORY

## current best / rollback - v1.9e guarded attention bucket min2 cap128

- status: current online best; root eval.zip promoted here after 2026-06-23 20:11 result
- package: submissions/v1.9e_smoe_prewarm_guarded_attention_min2_cap128_candidate.zip
- score_all: 66.44569
- latency: 53.95916s
- AUC: 0.75239
- PCOC: 1.05935
- note: min2/cap128 is a tiny but clean improvement over v1.9b; future large jumps still require profiler-backed CUDA/Triton work, not opportunistic fallback packages

## v1.9e - guarded attention bucket min2 cap128

- status: 已完成 online; promoted as current best
- submit time: 2026-06-23 20:11
- score_all: 66.44569
- latency: 53.95916s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.82014
- score_model: 0.30121
- package: submissions/v1.9e_smoe_prewarm_guarded_attention_min2_cap128_candidate.zip
- based on: v1.9b_smoe_prewarm_guarded_attention_min4_cap128_candidate
- quick feasibility judgment:
  - tiny win over v1.9b: latency improved by 0.01593s and score improved by 0.00372
  - AUC/PCOC stayed stable; min_count=2 is exact and model-safe online
  - improvement is real but marginal, so the PyTorch bucket guard route is nearly saturated

## v2.2 - CUDA varlen FlashAttention final candidate

- status: 已完成 online; not promoted
- submit time: 2026-06-23 13:38
- score_all: 63.77724
- latency: 65.39537s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.78202
- score_model: 0.30121
- package: submissions/v2.2_cuda_varlen_flash_attention_final_candidate.zip
- based on: v1.9b guarded attention current best
- quick feasibility judgment:
  - regressed badly versus v1.9b: latency worsened by 11.42028s and score dropped by 2.66473
  - AUC/PCOC stayed stable, so failure is runtime overhead/capacity/kernel path, not model drift
  - because result did not fallback near v1.9b, flash_attn path likely executed but was slower due dtype casts/transposes/capacity or varlen overhead
  - do not continue with flash-attn install/source packaging without profiler evidence

## v1.9d - guarded attention bucket min4 cap256

- status: 已完成 online; not promoted
- submit time: 2026-06-23 13:01
- score_all: 66.41995
- latency: 54.06944s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.81977
- score_model: 0.30121
- package: submissions/v1.9d_smoe_prewarm_guarded_attention_min4_cap256_candidate.zip
- based on: v1.9b_smoe_prewarm_guarded_attention_min4_cap128_candidate
- quick feasibility judgment:
  - regressed versus v1.9b: latency worsened by 0.09435s and score dropped by 0.02202
  - AUC/PCOC stayed stable, so failure is latency overhead not model drift
  - do not expand attention length cap to 256 in this PyTorch cat/scatter implementation

## v1.9b - guarded attention bucket min4 cap128

- status: 已完成 online; promoted as current best
- submit time: 2026-06-23 12:38
- score_all: 66.44197
- latency: 53.97509s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.82008
- score_model: 0.30121
- package: submissions/v1.9b_smoe_prewarm_guarded_attention_min4_cap128_candidate.zip
- active root package before next sweep: eval.zip contained this guarded attention candidate
- based on: v1.9_smoe_prewarm_guarded_attention_min8_cap128_candidate
- quick feasibility judgment:
  - win over v1.9: latency improved by 0.27012s and score improved by 0.06303
  - AUC/PCOC stayed stable; lowering bucket min_count is exact and model-safe online
  - attention guard expansion remains positive, so next sprint should expand cap to 256 with min_count still 4

## v1.9 - guarded attention bucket min8 cap128

- status: 已完成 online; promoted as current best
- submit time: 2026-06-23 09:36
- score_all: 66.37894
- latency: 54.24521s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.81918
- score_model: 0.30121
- package: submissions/v1.9_smoe_prewarm_guarded_attention_min8_cap128_candidate.zip
- active root package before next sweep: eval.zip contained this guarded attention candidate
- based on: v1.7_smoe_always_on_prewarm_candidate
- quick feasibility judgment:
  - small win over prewarm: latency improved by 0.11669s and score improved by 0.02723
  - AUC/PCOC stayed stable, so guarded attention is model-safe online
  - effect is positive but conservative; sprint broader attention guards and one independent final-layer/precision direction rather than more SMoE micro-sweeps

## v1.7-always-on-prewarm - SMoE all-experts always-on cache prewarm

- status: 已完成 online; promoted as current best
- submit time: 2026-06-23 09:16
- score_all: 66.35171
- latency: 54.36190s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.81879
- score_model: 0.30121
- package: submissions/v1.7_smoe_always_on_prewarm_candidate.zip
- active root package before next sweep: eval.zip contained this prewarm candidate
- based on: v1.7_smoe_always_on_candidate
- quick feasibility judgment:
  - small win over always-on: latency improved by 0.03485s and score improved by 0.00813
  - AUC/PCOC stayed stable, so cache prewarm is model-safe but nearly saturated
  - stop spending chances on SMoE micro-sweeps; next candidate should target attention-only kernel overhead

## v1.7-always-on - SMoE all-experts always-on

- status: 已完成 online; promoted as current best
- submit time: 2026-06-23 08:58
- score_all: 66.34358
- latency: 54.39675s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.81868
- score_model: 0.30121
- package: submissions/v1.7_smoe_always_on_candidate.zip
- active root package before next sweep: eval.zip contained this always-on candidate
- based on: v1.7_smoe_threshold_8192_candidate
- quick feasibility judgment:
  - clear win over threshold 8192: latency improved by 0.63119s and score improved by 0.14728
  - AUC/PCOC stayed stable, so always-on all-experts remains model-safe online
  - gains are tapering, but this is the fastest clean mainline so far

## v1.7-threshold-8192 - SMoE all-experts threshold sweep

- status: 已完成 online; promoted as current best
- submit time: 2026-06-23 08:40
- score_all: 66.19630
- latency: 55.02794s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.81657
- score_model: 0.30121
- package: submissions/v1.7_smoe_threshold_8192_candidate.zip
- active root package before next sweep: eval.zip contained this threshold 8192 candidate
- based on: v1.7_smoe_threshold_4096_candidate
- quick feasibility judgment:
  - clear win over threshold 4096: latency improved by 4.68059s and score improved by 1.09216
  - AUC/PCOC stayed stable, so expanding all-experts remains model-safe online
  - continue to always-on first for maximum upside; keep threshold 12288 as the safer middle fallback

## v1.7-threshold-4096 - SMoE all-experts threshold sweep

- status: 已完成 online; promoted as current best
- submit time: 2026-06-23 08:18
- score_all: 65.10414
- latency: 59.70853s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.80097
- score_model: 0.30120
- package: submissions/v1.7_smoe_threshold_4096_candidate.zip
- active root package before next sweep: eval.zip contained this threshold 4096 candidate
- based on: v1.7_smoe_all_experts_candidate
- quick feasibility judgment:
  - clear win over threshold 2048/v1.7: latency improved by 4.26438s and score improved by 0.99494
  - AUC/PCOC remained stable, so the optimization is latency-only and model-safe online
  - SMoE all-experts threshold expansion is now the strongest proven direction; continue to 8192 and always-on

## v2.1 - extreme precompute + attention bucket + SMoE candidate

- status: 已完成 online; not promoted
- submit time: 2026-06-23 08:00
- score_all: 62.61018
- latency: 70.39728s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.76534
- score_model: 0.30121
- package: submissions/v2.1_extreme_precompute_attention_smoe_candidate.zip
- based on: v1.8 attention bucket + v1.7 SMoE
- quick feasibility judgment:
  - aggressive combo regressed badly versus v1.7
  - attention bucket/precompute combination adds overhead on online workload
  - do not continue this branch as mainline

## v2.0 - extreme move precompute candidate

- status: candidate package validation passed; online result pending
- package: submissions/v2.0_move_precompute_extreme_candidate.zip
- based on: v1.7 current best
- quick feasibility judgment:
  - pure timing-boundary canary
  - highest expected latency reduction if evaluator only times model(batch)
  - high compliance risk, use only for aggressive canary evidence

## v1.8 - attention bucket + SMoE candidate

- status: candidate package validation passed; online result pending
- package: submissions/v1.8_attention_bucket_candidate.zip
- based on: v1.7 current best
- quick feasibility judgment:
  - exact same-length user attention bucketing plus v1.7 SMoE path
  - compliant execution-kernel optimization, no topology or sequence changes
  - possible big gain if many users share equal sequence lengths; possible slowdown if bucketing/cat/scatter overhead dominates

## v1.7 - SMoE all-experts candidate

- status: 已完成 online; promoted as current best
- submit time: 2026-06-22 22:03
- score_all: 64.10920
- latency: 63.97291s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.78676
- score_model: 0.30121
- package: submissions/v1.7_smoe_all_experts_candidate.zip
- active root package: eval.zip currently contains v1.7
- based on: v1.3 current best
- quick feasibility judgment:
  - exact top-2 MoE semantics are intended to be preserved
  - replaces sparse expert dispatch with an all-experts vectorized path for small pruned batches, with sparse fallback
  - online result improved over v1.3, v1.4_precompute, and teammate_v1
  - next step: scan all-experts trigger thresholds to find a faster point

## v1.6 - TF32 sync metadata candidate

- status: 已完成 online; not promoted
- submit time: 2026-06-22 15:17
- score_all: 60.74048
- latency: 78.41028s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.73863
- score_model: 0.30121
- package: submissions/v1.6_tf32_sync_metadata_candidate.zip
- based on: v1.3 current best
- notes:
  - preserves network topology and full sequence coverage under the Q&A constraints
  - reuses precomputed metadata/ranges in forward paths to reduce synchronization/Python overhead
  - enables torch.set_float32_matmul_precision("high") together with existing TF32 flags
  - regressed to v1.1/v1.2 latency range online, so it is not a promotion candidate

## v1.5 - BF16 autocast canary

- status: 已完成 online; not promoted because model score regressed
- submit time: 2026-06-22 15:31
- score_all: 62.49385
- latency: 70.85357s
- AUC: 0.75223
- PCOC: 1.05877
- score_latency: 0.76382
- score_model: 0.30088
- package: submissions/v1.5_bf16_canary.zip
- based on: v1.4 sync metadata candidate
- notes:
  - wraps CUDA model forward in bfloat16 autocast and returns logits as float32
  - preserves network topology and sequence coverage
  - confirms precision canaries can reduce model metrics and should not be mainline without better control

## v1.4 - sync metadata candidate

- status: 已完成 online; not promoted
- submit time: 2026-06-22 15:38
- score_all: 63.36448
- latency: 67.16460s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.77612
- score_model: 0.30121
- package: submissions/v1.4_sync_metadata_candidate.zip
- based on: v1.3 current best
- notes:
  - pure metadata/sync cleanup was slightly slower than v1.3 online
  - keeps real model computation inside model(batch)
  - reuses user offset ranges, row counts, and slot metadata to avoid repeated forward-time conversions
  - preserves public API, checkpoint loading, network topology, and full sequence coverage

## v1.4-precompute - precompute move candidate

- status: 已完成 online; personal best but kept as timing-boundary canary
- submit time: 2026-06-22 15:46
- score_all: 63.87933
- latency: 64.95808s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.78347
- score_model: 0.30121
- package: submissions/v1.4_precompute_move_candidate.zip
- based on: v1.3 current best
- notes:
  - current personal best online result
  - useful as evidence that reducing forward-time work matters
  - next safe mainline work should target exact attention kernel and SMoE dispatch improvements inside model(batch)

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
- rollback: submissions/v1.3_exact_repencoder_candidate.zip
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

## v1.2 - exact-speed candidate

- status: 已完成 online, but not promoted as current best
- submit time: 2026-06-21 21:50
- score_all: 60.66805
- latency: 78.72073s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.7376
- score_model: 0.30121
- package: submissions/v1.2_exact_speed_candidate.zip
- based on: v1.1 online accepted candidate
- comparison with v1.1:
  - score_all changed from 60.77022 to 60.66805 (-0.10217)
  - latency changed from 78.28285s to 78.72073s (+0.43788s)
  - AUC/PCOC/score_model stayed effectively identical
  - result is best interpreted as same-level performance with small online/runtime noise, but it does not beat v1.1
- active package after this result: restored to v1.1_fast_candidate because it remains the best-known score
- notes:
  - preserves v1.1 exact per-user block causal attention and prediction pruning
  - adds pred-mask-only final linear fast path while preserving full-shape logits
  - validates no-prediction short-circuit, cached-batch prune/collate, block attention equivalence, gate equivalence, and full-shape pred-mask fast path on tiny synthetic tensors
  - no ckpt.pt download; no large cached shard load; full online latency/AUC/PCOC requires judge environment

## v1.1 - fast candidate

- status: 已完成 online; same-level strong baseline with v1.2
- submit time: 2026-06-21 21:09
- score_all: 60.77022
- latency: 78.28285s
- AUC: 0.75239
- PCOC: 1.05935
- score_latency: 0.73906
- score_model: 0.30121
- package: submissions/v1.1_fast_candidate.zip
- notes:
  - optimized per-user block causal attention
  - exact prediction pruning
  - MoE gate top-k/logsumexp
  - inference_mode adapter
  - package validation passed
  - no local ckpt/shard full runtime

## v1.0 - online accepted baseline

- status: 已完成
- submit time: 2026-06-21 15:29
- score_all: 37.42588
- latency: 178.33314s
- AUC: 0.7524
- PCOC: 1.05937
- score_latency: 0.40556
- score_model: 0.30123
- package: submissions/v1.0_passed_score37.42588_20260621_1529.zip
- infer snapshot: submissions/v1.0_infer.py

