# Tomorrow Fail-Fast Submission Strategy

当前线上最佳：`v1.9b_smoe_prewarm_guarded_attention_min4_cap128_candidate.zip`

- score: 66.44197
- latency: 53.97509s
- AUC: 0.75239
- PCOC: 1.05935

今天的关键教训：运行错误不扣机会，运行成功才扣机会。因此明天不应提交“可能成功但很慢”的探测包；CUDA/FlashAttention/Triton 探测必须 fail-fast：不满足高置信条件时主动报错退出，避免消耗成功测评次数。

## 已证实结论

1. SMoE all-experts 是有效方向，但已接近饱和。
   - 2048/v1.7: 63.97291s
   - 4096: 59.70853s
   - 8192: 55.02794s
   - always-on: 54.39675s
   - prewarm: 54.36190s

2. PyTorch same-length attention bucket 有小收益，但不是 30s 路线。
   - min8/cap128: 54.24521s
   - min4/cap128: 53.97509s, current best
   - min4/cap256: 54.06944s, regressed

3. Opportunistic flash-attn fallback 不是好策略。
   - v2.2: 65.39537s, score 63.77724
   - AUC/PCOC 不变，说明不是语义漂移，而是 runtime overhead/capacity/kernel path 问题。
   - 因为它成功跑完但很慢，消耗了机会。

## 明日提交原则

### Principle 1: best package stays safe

Root `eval.zip` 当前应保持 current best：

`submissions/v1.9b_smoe_prewarm_guarded_attention_min4_cap128_candidate.zip`

除非已经准备好更高置信的新包，否则不要覆盖它。

### Principle 2: CUDA probe must fail fast

CUDA/FlashAttention/Triton 包必须在 `load_model` 或第一次 `model(batch)` 前半段做 gate：

- 必须确认 CUDA 可用。
- 必须确认目标库存在且接口匹配。
- 必须确认 shape/head_dim/dtype 满足 kernel 条件。
- 必须进行极小 dry-run 或 capability check。
- 若条件不满足，主动 raise RuntimeError，使评测失败而不是成功慢跑。

示例判断：

```python
if not torch.cuda.is_available():
    raise RuntimeError("FAIL_FAST_CUDA_REQUIRED")
try:
    from flash_attn import flash_attn_varlen_func
except Exception as exc:
    raise RuntimeError("FAIL_FAST_FLASH_ATTN_UNAVAILABLE") from exc
```

注意：不要 fallback 到慢路径作为提交包主逻辑；fallback 适合本地开发，不适合“成功才扣次数”的线上提交策略。

### Principle 3: submit only profiler-backed or strong hypothesis packages

明天优先级：

1. 在 A800/AI Studio 有 ckpt 的环境跑 profiler/shape statistics。
2. 如果 profiler 证明 attention launch 是主瓶颈，再做 CUDA/varlen/fused attention。
3. 如果 profiler 证明 SMoE matmul 是主瓶颈，再做 Triton/fused all-experts。
4. 如果 profiler 证明 embedding/slot 是主瓶颈，再做 slot pooling/fusion。

不要再做：

- cap256 PyTorch bucket 继续扫。
- flash-attn opportunistic fallback。
- torch.compile 全模型。
- broad BF16。
- 没有指标验证的 precision canary。

## 明日候选方向

### Candidate A: fail-fast FlashAttention varlen

仅在以下条件满足时完整跑：

- `flash_attn` 已安装。
- `flash_attn_varlen_func` 接口可调用。
- head_dim 支持。
- dtype 为 fp16/bf16，且 dry-run 成功。
- shape/capacity 不超限。

不满足则 raise，让评测失败不扣机会。

### Candidate B: fail-fast Triton fused SMoE

仅在确认：

- triton 可 import。
- JIT warmup 可在不计时阶段完成。
- fused kernel 对小 synthetic shape dry-run 成功。

否则 raise。

### Candidate C: conservative final best

如果没有足够证据，不提交新包，保留：

`v1.9b_smoe_prewarm_guarded_attention_min4_cap128_candidate.zip`

## Immediate state

- Root `eval.zip` restored to v1.9b current best.
- v2.2 marked not promoted.
- Tomorrow should start with profiling and shape/capability probes, not blind submissions.
