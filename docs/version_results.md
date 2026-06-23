# Version Results

本文档记录各版本线上提交结果，用于对比推理性能优化收益。

| 版本 | 团队成员用户名 | score | pcoc | score_latency | score_model | latency | auc | 提交状态 | 提交时间 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| v1 / v1.0 | 与君同行 | 37.42588 | 1.05937 | 0.40556 | 0.30123 | 178.33314 | 0.75240 | 已完成 | 2026-06-21 15:29 | 首个通过版，作为有效基线 |
| v1.1 | 与君同行 | 60.77022 | 1.05935 | 0.73906 | 0.30121 | 78.28285 | 0.75239 | 已完成 | 2026-06-21 21:09 | 第一轮大幅提速，v1.1/v1.2 同水平强基线 |
| v1.2 | 与君同行 | 60.66805 | 1.05935 | 0.73760 | 0.30121 | 78.72073 | 0.75239 | 已完成 | 2026-06-21 21:50 | 与 v1.1 基本同水平，差异可视为线上波动/小幅开销 |
| v1.3 | 与君同行 | 63.47372 | 1.05935 | 0.77768 | 0.30121 | 66.69640 | 0.75239 | 已完成 | 2026-06-22 07:30 | 当前稳定主线最佳 |
| v1.6_tf32_sync_metadata_candidate | 与君同行 | 60.74048 | 1.05935 | 0.73863 | 0.30121 | 78.41028 | 0.75239 | 已完成 | 2026-06-22 15:17 | 回退到 v1.1/v1.2 水平，未提升 |
| v1.5_bf16_canary | 与君同行 | 62.49385 | 1.05877 | 0.76382 | 0.30088 | 70.85357 | 0.75223 | 已完成 | 2026-06-22 15:31 | BF16 有速度收益但模型分下降，不作为主线 |
| v1.4_sync_metadata_candidate | 与君同行 | 63.36448 | 1.05935 | 0.77612 | 0.30121 | 67.16460 | 0.75239 | 已完成 | 2026-06-22 15:38 | 与 v1.3 接近但略差 |
| v1.4_precompute_move_candidate | 与君同行 | 63.87933 | 1.05935 | 0.78347 | 0.30121 | 64.95808 | 0.75239 | 已完成 | 2026-06-22 15:46 | 当前个人最高线上分，但需注意计时边界合规风险 |
| teammate_v1_based_on_v1.3 | 麦克饲喂 | 64.01352 | 1.05935 | 0.78539 | 0.30121 | 64.38298 | 0.75239 | 已完成 | 2026-06-22 16:35 | 队友基于 v1.3 的强参考 |
| v1.7_smoe_all_experts_candidate | 与君同行 | 64.10920 | 1.05935 | 0.78676 | 0.30121 | 63.97291 | 0.75239 | 已完成 | 2026-06-22 22:03 | SMoE dispatch 方向有效，后续阈值 sweep 继续放大收益 |
| v2.1_extreme_precompute_attention_smoe_candidate | 与君同行 | 62.61018 | 1.05935 | 0.76534 | 0.30121 | 70.39728 | 0.75239 | 已完成 | 2026-06-23 08:00 | 激进组合变慢，attention bucket/precompute 组合不应继续主推 |
| v1.7_smoe_threshold_4096_candidate | 与君同行 | 65.10414 | 1.05935 | 0.80097 | 0.30120 | 59.70853 | 0.75239 | 已完成 | 2026-06-23 08:18 | 阈值扩大明显有效，继续测 8192/always-on |
| v1.7_smoe_threshold_8192_candidate | 与君同行 | 66.19630 | 1.05935 | 0.81657 | 0.30121 | 55.02794 | 0.75239 | 已完成 | 2026-06-23 08:40 | SMoE 阈值继续扩大仍然有效，下一发 always-on |
| v1.7_smoe_always_on_candidate | 与君同行 | 66.34358 | 1.05935 | 0.81868 | 0.30121 | 54.39675 | 0.75239 | 已完成 | 2026-06-23 08:58 | always-on 继续小幅提升，下一发 prewarm |
| v1.7_smoe_always_on_prewarm_candidate | 与君同行 | 66.35171 | 1.05935 | 0.81879 | 0.30121 | 54.36190 | 0.75239 | 已完成 | 2026-06-23 09:16 | prewarm 微弱提升，SMoE 微优化基本见顶，下一发 guarded attention |
| v1.9_smoe_prewarm_guarded_attention_min8_cap128_candidate | 与君同行 | 66.37894 | 1.05935 | 0.81918 | 0.30121 | 54.24521 | 0.75239 | 已完成 | 2026-06-23 09:36 | attention-only 小幅正收益，下一步放宽 guard 冲更大覆盖 |
| v1.9b_smoe_prewarm_guarded_attention_min4_cap128_candidate | 与君同行 | 66.44197 | 1.05935 | 0.82008 | 0.30121 | 53.97509 | 0.75239 | 已完成 | 2026-06-23 12:38 | 前线上最高；降低 min_count 明显有效，cap128 是当前甜点 |
| v1.9e_smoe_prewarm_guarded_attention_min2_cap128_candidate | 与君同行 | 66.44569 | 1.05935 | 0.82014 | 0.30121 | 53.95916 | 0.75239 | 已完成 | 2026-06-23 20:11 | 当前线上最高；min2/cap128 比 v1.9b 小幅提升，AUC/PCOC 不变 |
| v1.9d_smoe_prewarm_guarded_attention_min4_cap256_candidate | 与君同行 | 66.41995 | 1.05935 | 0.81977 | 0.30121 | 54.06944 | 0.75239 | 已完成 | 2026-06-23 13:01 | cap256 回退，不晋升；中长序列 cat/scatter 开销大于收益 |
| v2.2_cuda_varlen_flash_attention_final_candidate | 与君同行 | 63.77724 | 1.05935 | 0.78202 | 0.30121 | 65.39537 | 0.75239 | 已完成 | 2026-06-23 13:38 | CUDA/FlashAttention varlen 回退严重，不晋升；推测 flash path 执行但 overhead/capacity 过大 |
| teammate_v2_attempt | 麦克饲喂 | 0.00000 | 1.05935 | 0.00000 | 0.30121 | 365.84964 | 0.75239 | 已完成 | 2026-06-22 19:11 | 延迟严重超限，计 0 分 |
| teammate_v2_modified | 麦克饲喂 | 60.15959 | 1.05936 | 0.73034 | 0.30119 | 80.89820 | 0.75239 | 已完成 | 2026-06-22 19:44 | 回到 v1.1/v1.2 附近但更慢，未提升 |

## 关键对比

- v1.0 → v1.1：latency 从 178.33314s 降至 78.28285s，总分从 37.42588 升至 60.77022。
- v1.1 ↔ v1.2：AUC、PCOC、score_model 基本一致，latency 仅相差 0.43788s，视为同水平。
- v1.2 → v1.3：latency 从 78.72073s 降至 66.69640s，总分从 60.66805 升至 63.47372。
- v1.4_sync_metadata_candidate 没有超过 v1.3，说明纯 Python metadata cleanup 收益不足。
- v1.5_bf16_canary 的 score_model 从 0.30121 降至 0.30088，说明精度 canary 存在模型指标风险。
- v1.6_tf32_sync_metadata_candidate 回退到 v1.1/v1.2 级别，不应晋升。
- v1.7_smoe_all_experts_candidate 超过 v1.4_precompute_move_candidate 与 teammate_v1，说明 SMoE dispatch 方向有效。
- v1.7_smoe_threshold_4096_candidate 继续把 latency 从 63.97291s 降到 59.70853s，总分升到 65.10414，证明扩大 all-experts 触发阈值是当前最有效方向。
- v1.7_smoe_threshold_8192_candidate 再把 latency 从 59.70853s 降到 55.02794s，总分升到 66.19630，说明 all-experts fast path 在更大 batch 上仍然收益大于额外计算。
- v1.7_smoe_always_on_candidate 继续把 latency 从 55.02794s 降到 54.39675s，总分升到 66.34358，说明 always-on 没有过度计算，当前收益开始收窄但仍为正。
- v1.7_smoe_always_on_prewarm_candidate 微降 latency 到 54.36190s，总分升到 66.35171，说明 SMoE 参数缓存微优化安全但收益基本见顶。
- v1.9_smoe_prewarm_guarded_attention_min8_cap128_candidate 继续把 latency 降到 54.24521s，总分升到 66.37894，说明 attention-only 方向成立但 min8/cap128 仍偏保守。
- v1.9b_smoe_prewarm_guarded_attention_min4_cap128_candidate 把 latency 降到 53.97509s，总分升到 66.44197，说明降低 bucket min_count 明显有效。
- v1.9e_smoe_prewarm_guarded_attention_min2_cap128_candidate 继续把 latency 微降到 53.95916s，总分升到 66.44569，说明 min2/cap128 仍有极小正收益但已接近饱和。
- v1.9d_smoe_prewarm_guarded_attention_min4_cap256_candidate 回退到 54.06944s，说明 cap256 的中长序列 cat/scatter 开销大于 attention 合并收益。
- v2.2_cuda_varlen_flash_attention_final_candidate 回退到 65.39537s，说明 opportunistic flash_attn varlen 不是可直接上线的大跳路线；需要 profiler/接口/shape 证据后再做。

## 当前版本定位

- 当前线上最高：v1.9e_smoe_prewarm_guarded_attention_min2_cap128_candidate，score=66.44569，latency=53.95916s。
- 当前稳定主线：v1.9e_smoe_prewarm_guarded_attention_min2_cap128_candidate。
- 当前保底回滚包：`submissions/v1.9e_smoe_prewarm_guarded_attention_min2_cap128_candidate.zip`；更保守回滚包为 `submissions/v1.9b_smoe_prewarm_guarded_attention_min4_cap128_candidate.zip` / `submissions/v1.9_smoe_prewarm_guarded_attention_min8_cap128_candidate.zip` / `submissions/v1.7_smoe_always_on_prewarm_candidate.zip` / `submissions/v1.7_smoe_always_on_candidate.zip` / `submissions/v1.7_smoe_threshold_8192_candidate.zip` / `submissions/v1.7_smoe_threshold_4096_candidate.zip` / `submissions/v1.7_smoe_all_experts_candidate.zip` / `submissions/v1.3_exact_repencoder_candidate.zip`。
- 当前 root eval.zip 为 v1.9e 最佳包；后续大跳必须基于 profiler/形状统计/明确假设，不再进行无证据探索。
