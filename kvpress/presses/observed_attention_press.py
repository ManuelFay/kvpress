# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class ObservedAttentionPress(ScorerPress):
    """
    Observed attention-based KV cache compression.

    Computes importance scores based on actual attention weights observed during
    forward pass. Score for each key-value pair is the average attention weight
    it receives from all query tokens.

    Requires: attn_implementation="eager".

    Related to H2O (https://arxiv.org/abs/2306.14048).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    n_sink : int, default=0
        Number of initial "sink" tokens to always preserve (assigned infinite score).
        Set to 4 for H2O-style behavior where first tokens act as attention sinks.
    """

    compression_ratio: float = 0.0
    n_sink: int = 0

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        assert attentions is not None, 'Set attn_implementation="eager" to use this hook'
        scores = attentions.sum(2)
        bsz, num_key_value_heads, n_tokens, _ = keys.shape
        n_tokens_in_sum = torch.arange(n_tokens, 0, -1).to(attentions.device, attentions.dtype)
        scores = scores / n_tokens_in_sum
        scores = scores.view(bsz, num_key_value_heads, -1, n_tokens).mean(2)

        # Protect sink tokens by assigning infinite score (never evicted)
        if self.n_sink > 0 and "cache_position" in kwargs:
            first_pos = kwargs["cache_position"][0].item()
            q_len = hidden_states.shape[1]

            # Sink tokens (absolute positions 0 to n_sink-1) get score inf (never evicted)
            n_sinks_in_chunk = max(0, min(self.n_sink - first_pos, q_len))
            if n_sinks_in_chunk > 0:
                scores[:, :, :n_sinks_in_chunk] = float("inf")

        return scores
