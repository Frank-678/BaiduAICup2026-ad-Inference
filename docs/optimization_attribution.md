# Optimization Attribution

本文档按线上单项数据归因：每个版本解决了什么、失利在哪里，以及下一步该怎么试。

## Baseline progression

| 版本 | 核心变化 | score | latency | AUC | PCOC | score_model | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| v1.0 | 首个通过基线 | 37.42588 | 178.33314 | 0.75240 | 1.05937 | 0.30123 | 可用但慢 |
| v1.1 | per-user block causal attention、预测裁剪、基础加速 | 60.77022 | 78.28285 | 0.75239 | 1.05935 | 0.30121 | 大幅解决主要慢点 |
| v1.2 | exact-speed 小改 | 60.66805 | 78.72073 | 0.75239 | 1.05935 | 0.30121 | 与 v1.1 同水平，未解决新瓶颈 |
| v1.3 | RepEncoder embedding_bag、final-head pred rows | 63.47372 | 66.69640 | 0.75239 | 1.05935 | 0.30121 | 解决 embedding/slot 与 final head 开销，是稳定主线 |

## Candidate attribution

| 版本 | 目标优化点 | 相对 v1.3 latency | 相对 v1.3 score | 成功点 | 失利点 | 决策 |
|---|---|---:|---:|---|---|---|
| v1.4_sync_metadata | forward 元数据/offset/set 复用 | +0.46820s | -0.10924 | AUC/PCOC 完全不变，说明语义安全 | 纯 Python metadata cleanup 不是瓶颈，甚至略慢 | 放弃主线 |
| v1.4_precompute_move | 将更多工作前移到 move 阶段 | -1.73832s | +0.40561 | 验证“减少 model(batch) 内工作”确实涨分 | 计时边界风险，不适合作为长期正攻 | 作为 canary 参考 |
| v1.5_bf16 | BF16 autocast | -5.84283s | -0.97987 | 速度有明显收益 | AUC/PCOC/score_model 下降，模型指标受损 | 不作为主线，除非做更精细混合精度 |
| v1.6_tf32_metadata | TF32/high + metadata | +11.71388s | -2.73324 | AUC/PCOC 不变 | 线上回退到 v1.1/v1.2 水平，说明该组合破坏了 v1.3 优势或引入开销 | 放弃 |
| teammate_v1 | 队友基于 v1.3 工程小改 | -2.31342s | +0.53980 | 保持 AUC/PCOC，说明 v1.3 仍有小开销可榨 | 仍只到 64.38s，无法大跳 | 作为稳定参考 |
| v1.7_smoe_all_experts | SMoE all-experts vectorized path，减少 sparse dispatch 开销 | -2.72349s | +0.63548 | AUC/PCOC 不变；证明 SMoE dispatch 是有效瓶颈 | 收益仍小，未达到 20 分级别 | 继续围绕 SMoE 阈值扫 |
| v1.7_smoe_threshold_4096 | 扩大 all-experts fast path 触发阈值到 4096 | -6.98787s | +1.63042 | 比 v1.7 继续降 4.26438s；AUC/PCOC 不变 | 仍未达到 20 分飞跃，但这是目前最干净的正向增益 | 已被 8192 超过 |
| v1.7_smoe_threshold_8192 | 扩大 all-experts fast path 触发阈值到 8192 | -11.66846s | +2.72258 | 比 4096 再降 4.68059s；AUC/PCOC 不变 | 仍需验证 always-on 是否过度计算 | 已被 always-on 超过 |
| v1.7_smoe_always_on | 全 batch 启用 all-experts vectorized path | -12.29965s | +2.86986 | 比 8192 再降 0.63119s；AUC/PCOC 不变 | 收益开始收窄，后续主要靠微优化或新瓶颈 | 已被 prewarm 小幅超过 |
| v1.7_smoe_always_on_prewarm | 在 load_model 阶段预构建 SMoE stacked expert 缓存 | -12.33450s | +2.87799 | 比 always-on 再降 0.03485s；AUC/PCOC 不变 | 收益极小，SMoE 微优化基本见顶 | 已被 v1.9 attention 超过 |
| v1.9_guarded_attention_min8_cap128 | same-length attention bucket，count>=8 且 length<=128 | -12.45119s | +2.90522 | 比 prewarm 再降 0.11669s；AUC/PCOC 不变 | 收益小，guard 偏保守或 equal-length 分布有限 | 已被 v1.9b 超过 |
| v1.9b_guarded_attention_min4_cap128 | same-length attention bucket，count>=4 且 length<=128 | -12.72131s | +2.96825 | 前最高；比 v1.9 再降 0.27012s；AUC/PCOC 不变 | 仍未到大跳，cap128 是当前甜点 | 已被 v1.9e 微幅超过 |
| v1.9e_guarded_attention_min2_cap128 | same-length attention bucket，count>=2 且 length<=128 | -12.73724s | +2.97197 | 当前最高；比 v1.9b 再降 0.01593s；AUC/PCOC 不变 | 收益极小，说明 PyTorch bucket guard 路线基本饱和 | 晋升主线；后续需 profiler-backed CUDA/Triton |
| v1.9d_guarded_attention_min4_cap256 | same-length attention bucket，count>=4 且 length<=256 | -12.62696s | +2.94623 | AUC/PCOC 不变，说明语义安全 | 比 v1.9b 慢 0.09435s；cap256 的 cat/scatter/memory 开销超过收益 | 不晋升；不要继续扩大 cap |
| v2.2_cuda_varlen_flash_attention | opportunistic flash_attn_varlen_func packed attention | -1.30103s | +0.30352 | AUC/PCOC 不变，语义未漂；线上 latency 65.39537s | 比 v1.9b 慢 11.42028s；flash path/转置/dtype/capacity overhead 严重 | 不晋升；不要继续安装/源码路线，除非 profiler 证明 |
| v2.1_extreme_combo | attention bucket + SMoE + move precompute | +3.70088s | -0.86354 | AUC/PCOC 不变，语义大概率没漂 | attention bucket/cat/scatter/precompute 组合开销大于收益 | 放弃该组合；不要把 attention bucket 和 precompute 叠加主推 |

## Metric-level observations

1. **AUC/PCOC 基本稳定的版本**
   - v1.1/v1.2/v1.3/v1.4_sync/v1.4_precompute/v1.6/v1.7/v2.1 都保持 AUC≈0.75239、PCOC≈1.05935。
   - 这些版本的差异主要是 latency，不是模型质量。

2. **模型指标受损的版本**
   - v1.5 BF16：AUC 0.75223、PCOC 1.05877、score_model 0.30088。
   - 说明粗粒度 BF16 不能作为主线；如继续精度路线，必须只包 attention/linear 部分并保留关键归一化/输出 fp32。

3. **稳定有效瓶颈**
   - RepEncoder/slot pooling：v1.3 证明有效。
   - SMoE dispatch：v1.7 证明有效。
   - SMoE 阈值扩大/always-on：4096、8192、always-on 连续证明有效，是当前最强方向。

4. **目前失败的方向**
   - metadata-only：太小。
   - TF32/high：线上回退。
   - attention bucket naive cat/scatter：v2.1 组合中明显拖慢，至少不能这样叠。
   - torch.compile：队友 v2 出现 365s，极高风险。

## Next experiments

### Priority 1: SMoE threshold sweep

当前线上最好是 v1.9e guarded attention min2/cap128：score 66.44569，latency 53.95916s；v1.9d cap256 回退到 54.06944s。

已生成：

- `submissions/v1.7_smoe_threshold_512_candidate.zip`
- `submissions/v1.7_smoe_threshold_1024_candidate.zip`
- `submissions/v1.7_smoe_threshold_4096_candidate.zip`
- `submissions/v1.7_smoe_threshold_8192_candidate.zip`

4096、8192、always-on、prewarm 连续证明更快，但 prewarm 只省 0.03485s，因此 SMoE all-experts 路线已推到当前极限；下一步必须切到 attention-only。

### Priority 2: SMoE all-experts always-on / finer sweep

继续做，均已打包：

- `submissions/v1.7_smoe_threshold_8192_candidate.zip`，当前 root `eval.zip`
- `submissions/v1.7_smoe_always_on_candidate.zip`
- `submissions/v1.7_smoe_threshold_6144_candidate.zip`
- `submissions/v1.7_smoe_threshold_12288_candidate.zip`

v2.2 已证明 opportunistic fallback 思路不够好：它成功跑完但慢，消耗了提交次数。由于运行错误不扣机会、运行成功才扣机会，明日 CUDA 探测应改为 fail-fast gate：环境/shape/kernel 条件不满足时主动报错退出，满足高置信条件才完整跑。

### Priority 3: Attention separate test only

attention bucket 不要和 precompute 组合。若再测 attention，应从 v1.7 稳定基线单独做：

- only attention bucket, no precompute
- bucket length cap 更小：64/128
- 只对同长度数量 >=4 或 >=8 的 bucket 合并，减少 cat/scatter 开销

### Priority 4: Avoid for now

- torch.compile
- broad BF16
- TF32/high 组合
- naive attention + precompute combo
