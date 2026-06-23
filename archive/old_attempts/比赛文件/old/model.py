"""
model.py — GRAB inference model with KV-Cache / M-FALCON support.

Architecture (GRAB paper SS3):
  1. SparseEmbedding   — PSTable-style embedding lookup
  2. DenseTokenizer    — Per-event field aggregation to R^{dmodel}
  3. CamA Transformer  — Causal Action-aware Multi-channel Attention
  4. CTR Head          — MLP -> sigmoid

Inference optimisations:
  - KV-Cache: encode_user_history() caches K/V; score_candidates() reuses it.
  - Supports M-FALCON broadcast-attention: one user fwd pass scores N candidates.
"""

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Optional, Tuple, List


class GRABConfig:
    """Model hyper-parameters (GRAB-standard: nlayer=4, nhead=4, dmodel=64)."""
    def __init__(self, vocab_size=100000, d_model=64, n_layers=4, n_heads=4,
                 d_ff=128, n_channels=2, max_seq_len=1024,
                 num_time_buckets=64, num_act_buckets=8, num_pos_buckets=64,
                 sliding_window=256, dropout=0.0):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_channels = n_channels
        self.max_seq_len = max_seq_len
        self.num_time_buckets = num_time_buckets
        self.num_act_buckets = num_act_buckets
        self.num_pos_buckets = num_pos_buckets
        self.sliding_window = sliding_window
        self.dropout = dropout


class SparseEmbedding(nn.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, sparse=False)
    def forward(self, ids):
        return self.embedding(ids)


class DenseTokenizer(nn.Layer):
    def __init__(self, d_model, n_fields=8):
        super().__init__()
        self.field_proj = nn.Linear(n_fields * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, field_embeds):
        B, L, nF, D = field_embeds.shape
        flat = field_embeds.reshape((B * L, nF * D))
        tok = self.field_proj(flat).reshape((B, L, D))
        return self.norm(tok)


class ActionAwareRAB(nn.Layer):
    """Efficient action-aware relative attention bias (GRAB SS3.3.3, Eq.3-4)."""
    def __init__(self, cfg: GRABConfig):
        super().__init__()
        hd = cfg.d_model // cfg.n_heads
        self.head_dim = hd
        self.num_pos_buckets = cfg.num_pos_buckets
        self.num_time_buckets = cfg.num_time_buckets
        self.num_act_buckets = cfg.num_act_buckets
        init = nn.initializer.XavierNormal()
        self.codebook_pos = self.create_parameter([cfg.num_pos_buckets, hd], default_initializer=init)
        self.codebook_time = self.create_parameter([cfg.num_time_buckets, hd], default_initializer=init)
        self.codebook_act = self.create_parameter([cfg.num_act_buckets, hd], default_initializer=init)
        self.pos_proj = nn.Linear(hd, hd, bias_attr=False)
        self.time_proj = nn.Linear(hd, hd, bias_attr=False)
        self.act_proj = nn.Linear(hd, hd, bias_attr=False)

    def _bucketize(self, values, num_buckets, max_distance=512):
        half = num_buckets // 2
        is_neg = values < 0
        abs_v = paddle.abs(values)
        v = paddle.where(abs_v < half, abs_v, paddle.clip(abs_v, half, max_distance))
        log_scale = math.log(max_distance / max(1, half)) / max(1, half - 1) if half > 1 else 1.0
        log_part = half + paddle.floor(paddle.log(v / max(1, half) + 1e-6) / (log_scale + 1e-6))
        b = paddle.where(v < half, v.astype(paddle.int64),
                         paddle.clip(log_part.astype(paddle.int64), max=num_buckets - 1))
        b = paddle.where(is_neg, num_buckets - 1 - b, b)
        return paddle.clip(b, 0, num_buckets - 1)

    def forward(self, q, pos_deltas, time_deltas, act_deltas):
        B, nH, L, D = q.shape
        pb = self._bucketize(pos_deltas, self.num_pos_buckets)
        tb = self._bucketize(time_deltas, self.num_time_buckets)
        ab = paddle.clip(act_deltas.abs().astype(paddle.int64), 0, self.num_act_buckets - 1)
        qf = q.reshape((B * nH * L, D))
        sp = self.pos_proj(qf).reshape((B, nH, L, D))
        st = self.time_proj(qf).reshape((B, nH, L, D))
        sa = self.act_proj(qf).reshape((B, nH, L, D))
        def _gather(cb, s, idx):
            return (s.unsqueeze(3) * cb[paddle.clip(idx, 0, cb.shape[0] - 1)]).sum(axis=-1)
        return _gather(self.codebook_pos, sp, pb) + _gather(self.codebook_time, st, tb) + _gather(self.codebook_act, sa, ab)


class CausalChannelAttention(nn.Layer):
    def __init__(self, cfg: GRABConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.sliding_window = cfg.sliding_window
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias_attr=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias_attr=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias_attr=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias_attr=False)
        self.rab = ActionAwareRAB(cfg)

    def _split(self, x):
        B, L, D = x.shape
        return x.reshape((B, L, self.n_heads, self.head_dim)).transpose((0, 2, 1, 3))

    def _merge(self, x):
        return x.transpose((0, 2, 1, 3)).reshape((x.shape[0], x.shape[2], -1))

    def forward(self, x, pos_deltas=None, time_deltas=None, act_deltas=None,
                causal_mask=None, kv_cache=None):
        B, L, D = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))
        if kv_cache is not None:
            k = paddle.concat([kv_cache[0], k], axis=2)
            v = paddle.concat([kv_cache[1], v], axis=2)
        new_cache = (k, v)
        total_L = k.shape[2]
        scale = self.head_dim ** -0.5
        scores = paddle.matmul(q, k, transpose_y=True) * scale
        if pos_deltas is None:
            rp = paddle.arange(total_L, dtype=x.dtype).unsqueeze(1) - paddle.arange(total_L, dtype=x.dtype).unsqueeze(0)
            rp = rp.unsqueeze(0).expand((B, -1, -1))
            rt, ra = rp.clone(), paddle.zeros((B, total_L, total_L), dtype=x.dtype)
        else:
            rp, rt, ra = pos_deltas, time_deltas, act_deltas
        scores = scores + self.rab(q[:, :, -L:, :],
            rp[:, -L:, :total_L], rt[:, -L:, :total_L], ra[:, -L:, :total_L])
        mask = paddle.tril(paddle.ones((total_L, total_L), dtype=x.dtype))
        if self.sliding_window > 0 and total_L > self.sliding_window:
            w = (paddle.tril(paddle.ones((total_L, total_L), dtype=x.dtype), self.sliding_window - 1) -
                 paddle.tril(paddle.ones((total_L, total_L), dtype=x.dtype), -self.sliding_window))
            mask = mask * w
        scores = scores + (mask[-L:, :total_L].unsqueeze(0).unsqueeze(0) - 1.0) * 1e9
        attn = F.softmax(scores, axis=-1)
        return self.out_proj(self._merge(paddle.matmul(attn, v))), new_cache


class FFN(nn.Layer):
    def __init__(self, cfg: GRABConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CamALayer(nn.Layer):
    """C channels + target-token gated mixing (GRAB SS3.3.4)."""
    def __init__(self, cfg: GRABConfig, layer_id=0):
        super().__init__()
        C = cfg.n_channels
        self.n_channels = C
        self.channel_attn = nn.LayerList([CausalChannelAttention(cfg) for _ in range(C)])
        self.channel_ffn = nn.LayerList([FFN(cfg) for _ in range(C)])
        self.norm1 = nn.LayerList([nn.LayerNorm(cfg.d_model) for _ in range(C)])
        self.norm2 = nn.LayerList([nn.LayerNorm(cfg.d_model) for _ in range(C)])
        self.gate = nn.Linear(cfg.d_model * C, C * C)
        self.gate_act = nn.Sigmoid()

    def forward(self, channel_inputs, pos_deltas=None, time_deltas=None,
                act_deltas=None, causal_masks=None, kv_caches=None):
        C = self.n_channels
        new_caches, channel_out = [], []
        for c in range(C):
            x = channel_inputs[c]
            cache = kv_caches[c] if kv_caches else None
            a, nc = self.channel_attn[c](self.norm1[c](x),
                pos_deltas[c] if pos_deltas else None,
                time_deltas[c] if time_deltas else None,
                act_deltas[c] if act_deltas else None,
                causal_masks[c] if causal_masks else None, cache)
            x = channel_inputs[c] + a
            new_caches.append(nc)
            x = x + self.channel_ffn[c](self.norm2[c](x))
            channel_out.append(x)
        tgts = [channel_out[c][:, -1:, :] for c in range(C)]
        gw = self.gate_act(self.gate(paddle.concat(tgts, axis=-1))).reshape((-1, 1, C, C))
        mixed = []
        for c in range(C):
            m = paddle.zeros_like(tgts[c])
            for i in range(C):
                if i != c:
                    m = m + gw[:, :, i, c:c+1] * tgts[i]
            mixed.append(tgts[c] + m)
        return [paddle.concat([channel_out[c][:, :-1, :], mixed[c]], axis=1) for c in range(C)], new_caches


class GRABModel(nn.Layer):
    """Full GRAB model with two-phase KV-Cache inference (M-FALCON pattern)."""
    def __init__(self, cfg: GRABConfig):
        super().__init__()
        self.cfg = cfg
        self.sparse_embed = SparseEmbedding(cfg.vocab_size, cfg.d_model)
        self.tokenizer = DenseTokenizer(cfg.d_model, n_fields=8)
        self.layers = nn.LayerList([CamALayer(cfg) for _ in range(cfg.n_layers)])
        self.ctr_head = nn.Sequential(
            nn.Linear(cfg.d_model * cfg.n_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _embed(self, tok):
        e = self.sparse_embed(tok)
        return self.tokenizer(e.unsqueeze(2).expand((-1, -1, 1, -1)))

    @paddle.no_grad()
    def encode_user_history(self, user_histories):
        B, L_hist = user_histories.shape
        ht = self._embed(user_histories)
        cs = [ht] * self.cfg.n_channels
        rp = paddle.arange(L_hist, dtype=ht.dtype).unsqueeze(1) - paddle.arange(L_hist, dtype=ht.dtype).unsqueeze(0)
        rp = rp.unsqueeze(0)
        rt, ra = rp.clone(), paddle.zeros((1, L_hist, L_hist), dtype=ht.dtype)
        pd = [rp] * self.cfg.n_channels
        td = [rt] * self.cfg.n_channels
        ad_ = [ra] * self.cfg.n_channels
        cm_ = [paddle.tril(paddle.ones((L_hist, L_hist), dtype=ht.dtype)).unsqueeze(0)] * self.cfg.n_channels
        kc = [[None] * self.cfg.n_channels for _ in range(self.cfg.n_layers)]
        for li, layer in enumerate(self.layers):
            cs, nc = layer(cs, pos_deltas=pd, time_deltas=td, act_deltas=ad_, causal_masks=cm_, kv_caches=kc[li])
            kc[li] = nc
        return kc

    @paddle.no_grad()
    def score_candidates(self, ad_tokens, kv_caches):
        _, N = ad_tokens.shape
        at = self._embed(ad_tokens)
        ci = [at] * self.cfg.n_channels
        for li, layer in enumerate(self.layers):
            ci, _ = layer(ci, kv_caches=kv_caches[li])
        lt = paddle.concat([ci[c] for c in range(self.cfg.n_channels)], axis=-1)
        return F.sigmoid(self.ctr_head(lt).squeeze(-1))

    @paddle.no_grad()
    def forward(self, user_histories, ad_tokens):
        return self.score_candidates(ad_tokens, self.encode_user_history(user_histories))
