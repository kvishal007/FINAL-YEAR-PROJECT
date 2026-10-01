"""
Cross-attention fusion of Swin Transformer and EfficientNet tokens.

Each Swin token (global context) asks a question (Query) and looks across all
EfficientNet tokens (local detail), which supply the Keys and Values:

    Q = q_proj(swin_tokens)           [B, 49, 768]  -> [B, 49, D]
    K = k_proj(efficientnet_tokens)   [B, 49, 1280] -> [B, 49, D]
    V = v_proj(efficientnet_tokens)   [B, 49, 1280] -> [B, 49, D]

    attention = softmax(Q K^T / sqrt(D / num_heads))   per head, [B, H, 49, 49]
    attended  = attention V                             heads merged, [B, 49, D]
    tokens    = LayerNorm(Q + out_proj(attended))       [B, 49, D]

    fused     = mean of tokens over the 49 positions    [B, D]

D = attention_dim, H = num_heads.

Design choices made for this project (the DeFaX paper's exact values are
NOT reproduced here):
- attention_dim=256, num_heads=8 by default: small enough for a 4 GB GPU
  (49 x 49 attention per head) while giving 32 dimensions per head.
- Residual connection + LayerNorm around the attention, the standard
  Transformer arrangement, so the Swin information is kept and training
  stays stable.
- Mean pooling for the fused vector: every spatial position counts equally.
"""

import math

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        swin_dim=768,
        efficientnet_dim=1280,
        attention_dim=256,
        num_heads=8,
        dropout=0.1,
    ):
        """
        swin_dim:          feature size of Swin tokens (query source)
        efficientnet_dim:  feature size of EfficientNet tokens (key/value source)
        attention_dim:     common size both are projected to
        num_heads:         attention heads; must divide attention_dim exactly
        dropout:           dropout on attention weights (active only in training)
        """
        super().__init__()

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.swin_dim = swin_dim
        self.efficientnet_dim = efficientnet_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        # Projections into the common attention dimension
        self.q_proj = nn.Linear(swin_dim, attention_dim)
        self.k_proj = nn.Linear(efficientnet_dim, attention_dim)
        self.v_proj = nn.Linear(efficientnet_dim, attention_dim)
        self.out_proj = nn.Linear(attention_dim, attention_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(attention_dim)

    def _check_inputs(self, swin_tokens, efficientnet_tokens):
        for name, tokens, dim in (
            ("swin_tokens", swin_tokens, self.swin_dim),
            ("efficientnet_tokens", efficientnet_tokens, self.efficientnet_dim),
        ):
            if tokens.ndim != 3 or tokens.shape[-1] != dim:
                raise ValueError(
                    f"{name} must be [batch_size, num_tokens, {dim}], "
                    f"got {list(tokens.shape)}"
                )
        if swin_tokens.shape[0] != efficientnet_tokens.shape[0]:
            raise ValueError(
                f"Batch sizes differ: swin {swin_tokens.shape[0]}, "
                f"efficientnet {efficientnet_tokens.shape[0]}"
            )

    def project_qkv(self, swin_tokens, efficientnet_tokens):
        """Return Q (from Swin), K and V (from EfficientNet), each [B, N, D]."""
        self._check_inputs(swin_tokens, efficientnet_tokens)
        q = self.q_proj(swin_tokens)
        k = self.k_proj(efficientnet_tokens)
        v = self.v_proj(efficientnet_tokens)
        return q, k, v

    def _split_heads(self, x):
        """[B, N, D] -> [B, H, N, D/H]"""
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        """[B, H, N, D/H] -> [B, N, D]"""
        batch, _, tokens, _ = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, self.attention_dim)

    def forward(self, swin_tokens, efficientnet_tokens, return_attention=False):
        """
        swin_tokens:          [B, 49, swin_dim]
        efficientnet_tokens:  [B, 49, efficientnet_dim]

        Returns fused tokens [B, 49, attention_dim].
        With return_attention=True also returns the attention weights
        [B, num_heads, 49, 49] (row i = how Swin token i spreads its attention
        over the EfficientNet tokens; each row sums to 1).
        """
        q, k, v = self.project_qkv(swin_tokens, efficientnet_tokens)

        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)

        scores = q_heads @ k_heads.transpose(-2, -1) / math.sqrt(self.head_dim)
        attention = scores.softmax(dim=-1)
        attended = self.attention_dropout(attention) @ v_heads

        attended = self.out_proj(self._merge_heads(attended))
        tokens = self.norm(q + attended)

        if return_attention:
            return tokens, attention
        return tokens

    def forward_fused(self, swin_tokens, efficientnet_tokens):
        """Mean-pool the fused tokens over all positions -> [B, attention_dim]."""
        return self.forward(swin_tokens, efficientnet_tokens).mean(dim=1)
