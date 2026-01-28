# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class StreamingLLMPress(ScorerPress):
    """
    StreamingLLM: Window-based KV cache compression with sink tokens.

    Implements sliding window approach preserving first few tokens (sink tokens)
    and most recent tokens, while pruning middle tokens.

    Based on StreamingLLM (https://arxiv.org/abs/2309.17453).
    To fully match the implementation described in the paper, use the KeyRerotationPress wrapper (see issue #158).

    This scorer is designed to work with DMSPress for threshold-based eviction:
    - Sink tokens (first n_sink positions) receive score=1.0 and are never evicted
    - All other tokens receive score=0.0 and will be evicted when they leave DMSPress's sliding window
    - The recent window protection is handled by DMSPress.sliding_window_size, not by this class

    When used standalone (not wrapped in DMSPress), the sliding_window_size parameter
    controls which tokens to keep based on compression_ratio.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
        NOTE: This parameter is ignored when wrapped in DMSPress.
    n_sink : int, default=4
        Number of initial tokens to always preserve (sink tokens).
        These tokens are never pruned and serve as "attention sinks" that help
        maintain model stability. Language models often assign high attention
        weights to early tokens regardless of semantic content.
    sliding_window_size : int, optional
        Number of most recent tokens to preserve when used standalone.
        NOTE: When wrapped in DMSPress, use DMSPress.sliding_window_size instead.
    """

    compression_ratio: float = 0.0
    n_sink: int = 4
    sliding_window_size: int = 0

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        q_len = keys.shape[2]

        # Check if we have cache_position info (when called from DMSPress)
        if "cache_position" in kwargs and kwargs["cache_position"] is not None:
            # DMSPress mode: use absolute positions to identify sink tokens
            # DMSPress handles the sliding window, we only need to protect sinks
            first_pos = kwargs["cache_position"][0].item()

            # All tokens start with score 0 (will be evicted when leaving DMSPress window)
            scores = torch.zeros_like(keys[..., 0])

            # Sink tokens (absolute positions 0 to n_sink-1) get score 1 (never evicted)
            n_sinks_in_chunk = max(0, min(self.n_sink - first_pos, q_len))
            if n_sinks_in_chunk > 0:
                scores[:, :, :n_sinks_in_chunk] = 1.0

            return scores

        # Standalone mode: original behavior for backward compatibility
        k_len = q_len
        assert k_len > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"

        scores = torch.ones_like(keys[..., 0])

        if self.sliding_window_size > 0:
            # Fixed structure: n_sink + sliding_window_size
            # Keep first n_sink tokens and last sliding_window_size tokens, prune middle
            n_to_keep = self.n_sink + self.sliding_window_size
            if k_len > n_to_keep:
                # Prune tokens between sink and sliding window
                start_prune = self.n_sink
                end_prune = k_len - self.sliding_window_size
                scores[:, :, start_prune:end_prune] = 0
        else:
            # Original behavior: use compression_ratio
            n_pruned = k_len - int(k_len * (1 - self.compression_ratio))
            scores[:, :, self.n_sink : self.n_sink + n_pruned] = 0

        return scores
