# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class H2OPress(ScorerPress):
    """
    H2O (Heavy-Hitter Oracle) KV cache compression.

    Combines local window with observed attention-based importance scoring.
    Implements the approach from "H2O: Heavy-Hitter Oracle for Efficient
    Generative Inference of Large Language Models" (https://arxiv.org/abs/2306.14048).

    The cache is divided into:
    1. Recent tokens (local window): Always kept for maintaining generation quality
    2. Heavy hitters: Tokens with highest cumulative attention from the older portion

    Requires: attn_implementation="eager" to access attention weights.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    local_window_size : int, default=512
        Number of most recent tokens to always preserve (local window).
        These tokens are never pruned and help maintain local context.
    """

    compression_ratio: float = 0.0
    local_window_size: int = 512

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        assert attentions is not None, 'Set attn_implementation="eager" to use H2OPress'

        k_len = keys.shape[2]
        bsz, num_key_value_heads, n_tokens, _ = keys.shape

        # If sequence is shorter than local window, keep everything
        if k_len <= self.local_window_size:
            return torch.ones(bsz, num_key_value_heads, k_len,
                            device=keys.device, dtype=keys.dtype)

        # Compute observed attention scores (same as ObservedAttentionPress)
        # Sum attention across query positions and normalize
        attention_scores = attentions.sum(2)
        n_tokens_in_sum = torch.arange(n_tokens, 0, -1).to(attentions.device, attentions.dtype)
        attention_scores = attention_scores / n_tokens_in_sum
        attention_scores = attention_scores.view(bsz, num_key_value_heads, -1, n_tokens).mean(2)

        # Create scores tensor
        scores = torch.zeros(bsz, num_key_value_heads, k_len,
                           device=keys.device, dtype=keys.dtype)

        # Local window: assign very high scores to recent tokens (guaranteed to be kept)
        local_start = k_len - self.local_window_size
        scores[:, :, local_start:] = float('inf')

        # Heavy hitters: use observed attention scores for older tokens
        scores[:, :, :local_start] = attention_scores[:, :, :local_start]

        return scores
