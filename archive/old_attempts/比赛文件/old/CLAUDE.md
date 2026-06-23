# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository is for the **2026 Baidu Commercial AI Technology Innovation Contest (CTI)**: Generative Recommendation Advertising Sorting Inference Performance Optimization. The competition focuses on optimizing the inference performance of Baidu's **GRAB** (Generative Ranking for Ads at Baidu) model — an LLM-inspired, Transformer-based CTR prediction model for advertising ranking.

## Key Papers (included in repo)

- **GRAB**: `2602.01865v2.pdf` — Baidu's generative ranking framework using Causal Action-aware Multi-channel Attention (CamA). The target model architecture for this competition.
- **HSTU**: `2402.17152v3.pdf` — Meta's Hierarchical Sequential Transduction Units, a foundational GR architecture that GRAB builds upon.

## Competition Rules & Constraints

- **Model architecture must not be structurally modified.** Only inference performance optimizations are allowed.
- **No training on the test set.** Pre-trained model weights are provided by the competition.
- **Total inference time limit: 5 minutes** (pure inference). Build env (`build_env.sh`) must complete within 20 minutes.
- **Score = inference_efficiency_score × strategy_effectiveness_score** (both must be non-zero).
- Strategy effectiveness: AUC ≥ 0.65 AND PCOC ∈ [0.85, 1.15] required for ranking.

## Submission Structure

Participants submit a `.zip` containing:
- `infer.py` — Entry point for inference
- `build_env.sh` — Environment setup script
- `requirements.txt` — Python dependencies
- Optionally: quantized model files, custom Python environment, compiled kernels

The system calls `infer.py` automatically. Original model weights are available by default — do not upload them unless custom weights are needed.

## Model Architecture (GRAB)

1. **Sparse Feature Layer** — Converts raw categorical fields into sparse IDs via PSTable (PaddleBox)
2. **Dense Tokenizer** — Aggregates per-event field embeddings into sequential event tokens
3. **Sequence Modeling Layer** — Causal Transformer with CamA (Causal Action-aware Multi-channel Attention):
   - Sequence packing with user-isolated causal masking
   - Heterogeneous behavior tokens (Partial + Full token views)
   - Action-aware Relative Attention Bias (RAB) with sliding windows
   - Multi-channel attention with target-token gated mixing
   - STS (Sequence Then Sparse) training (not relevant for inference)

## Key Optimization Dimensions

From the deployment section of the GRAB paper, critical optimization techniques include:
- **Operator Fusion**: Fuse GEMM + Bias + LayerNorm to reduce kernel launch overhead (~43% latency improvement)
- **Mixed Precision**: TF32 for Transformer matmuls, FP16 for FC layers
- **KV-cache Reuse** (M-FALCON): Share user KV-cache across candidate micro-batches
- **Disaggregated Serving**: Separate User Interest Center for async KV-cache computation
- **Sequence Length Optimization**: Sliding window attention, stochastic length

## Development Resources

- **Baseline**: https://aistudio.baidu.com/projectdetail/10186630
- **Model weights**: https://aistudio.baidu.com/modelsdetail/45703/space
- **Dataset**: https://aistudio.baidu.com/dataset/detail/375013/file
- **Framework**: Baidu PaddlePaddle (PaddleBox for distributed training, Paddle Inference for deployment)
- **Hardware**: NVIDIA GPUs (DGX-Spark available as prize)
