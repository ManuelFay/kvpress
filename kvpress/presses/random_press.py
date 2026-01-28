# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class RandomPress(ScorerPress):
    """
    Random KV cache compression for baseline comparison.

    Randomly selects which key-value pairs to prune. Useful for establishing baseline
    performance metrics and validating other compression methods.

    When used with DMSPress (detected via cache_position in kwargs), sink tokens
    (first n_sink positions) are protected with score=inf to ensure they are never evicted.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    seed : int, optional
        Random seed for reproducible compression results.
    n_sink : int, default=4
        Number of initial tokens to always preserve (sink tokens).
        Only used when wrapped in DMSPress (when cache_position is available).
    """

    compression_ratio: float = 0.0
    seed: Optional[int] = None
    n_sink: int = 4

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=keys.device)
            generator.manual_seed(self.seed)

        # Generate random scores in [0, 1)
        scores = torch.rand(*keys.shape[:-1], generator=generator, device=keys.device, dtype=keys.dtype)

        # If called from DMSPress, protect sink tokens with score=inf
        if "cache_position" in kwargs and kwargs["cache_position"] is not None:
            q_len = keys.shape[2]
            first_pos = kwargs["cache_position"][0].item()

            # Sink tokens (absolute positions 0 to n_sink-1) get score inf (never evicted)
            n_sinks_in_chunk = max(0, min(self.n_sink - first_pos, q_len))
            if n_sinks_in_chunk > 0:
                scores[:, :, :n_sinks_in_chunk] = float("inf")

        return scores
