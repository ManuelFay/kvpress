# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values


@dataclass
class DMSPress(BasePress):
    """
    Based on Dynamic Memory Sparsification (DMS, https://arxiv.org/abs/2506.05345) inference.
    Wraps a ScorerPress and evicts keys/values with scores below a given threshold.

    Unlike most presses that use a fixed compression_ratio, DMSPress uses a score threshold
    to determine which KV pairs to evict. This allows for adaptive compression where the actual
    compression ratio depends on the input content.

    Importantly, this press can be used both during prefilling and during decoding (if decoding=True).

    A sliding window protects the most recent tokens from eviction, ensuring that recently
    generated tokens are always available for attention.

    Parameters
    ----------
    press : ScorerPress
        The underlying scorer press used to compute importance scores for each token.
    threshold : float, optional
        Tokens with scores below this threshold are evicted. The optimal threshold
        depends on the scorer press being used. Mutually exclusive with target_density.
    target_density : float, optional
        Target fraction of tokens to keep (0.0 to 1.0). When set, the threshold is
        computed dynamically at each eviction step to achieve this density.
        For example, target_density=0.3 keeps approximately 30% of compressible tokens.
        Mutually exclusive with threshold.
    sliding_window_size : int, default=128
        Number of recent tokens protected from eviction.
    decoding : bool, default=False
        If True, compression is also applied during the decoding phase (token generation).
        If False, compression only occurs during prefill.
    accumulate_attention : bool, default=False
        If True, accumulate attention scores across chunks for methods that use observed
        attention (like H2O/ObservedAttentionPress). This ensures that tokens receive
        credit for attention from ALL queries, not just queries in the same chunk.
        Required for proper H2O behavior where sink tokens should accumulate attention
        from the entire sequence.
    """

    press: ScorerPress
    threshold: Optional[float] = None
    target_density: Optional[float] = None
    sliding_window_size: int = 128
    decoding: bool = False
    accumulate_attention: bool = False

    # Internal buffers
    scores_buffer: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    compression_ratios: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    # For cumulative attention tracking
    cumulative_attention: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    buffer_start_pos: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    total_queries_seen: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.threshold is not None and self.target_density is not None:
            raise ValueError("Cannot specify both threshold and target_density. Choose one.")
        if self.threshold is None and self.target_density is None:
            raise ValueError("Must specify either threshold or target_density.")
        if self.target_density is not None and not (0.0 < self.target_density < 1.0):
            raise ValueError(f"target_density must be between 0 and 1, got {self.target_density}")

    def _compute_threshold_for_density(self, scores: torch.Tensor) -> float:
        """
        Compute the threshold that keeps target_density fraction of tokens.

        Parameters
        ----------
        scores : torch.Tensor
            Scores for tokens to potentially evict (batch, num_heads, n_tokens)

        Returns
        -------
        float
            Threshold value such that target_density fraction of scores are >= threshold
        """
        # Flatten scores and filter out inf (protected tokens like sinks)
        flat_scores = scores.flatten()
        finite_mask = torch.isfinite(flat_scores)
        finite_scores = flat_scores[finite_mask]

        if finite_scores.numel() == 0:
            # All tokens are protected, return -inf to keep everything
            return float("-inf")

        # Compute the percentile that corresponds to keeping target_density fraction
        # If target_density=0.3, we want the 70th percentile (evict bottom 70%)
        eviction_percentile = (1.0 - self.target_density) * 100.0
        threshold = torch.quantile(finite_scores.float(), eviction_percentile / 100.0).item()

        return threshold

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        """Average compression ratio across all layers (computed after forward pass)."""
        assert len(self.compression_ratios) > 0, "Forward pass must be run to compute the compression ratio"
        return sum(self.compression_ratios.values()) / len(self.compression_ratios)

    @compression_ratio.setter
    def compression_ratio(self, value):
        """Compression ratio is read-only since it depends on threshold and input content."""
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

    def _accumulate_attention_scores(
        self,
        layer_idx: int,
        full_attentions: torch.Tensor,
        keys: torch.Tensor,
        q_len: int,
        cache_len: int,
        prefilling: bool,
        kwargs: dict,
    ) -> torch.Tensor:
        """
        Accumulate attention scores across chunks for all positions in the buffer.

        This allows positions to receive credit for attention from ALL queries that have
        attended to them, not just queries from their original chunk.

        Parameters
        ----------
        layer_idx : int
            Current layer index
        full_attentions : torch.Tensor
            Full attention matrix (batch, num_heads, q_len, cache_len)
        keys : torch.Tensor
            Keys tensor (batch, num_kv_heads, cache_len, head_dim)
        q_len : int
            Number of queries in current chunk
        cache_len : int
            Total cache length
        prefilling : bool
            Whether we're in prefilling phase
        kwargs : dict
            Forward pass kwargs

        Returns
        -------
        torch.Tensor
            Normalized scores for all positions in buffer (batch, num_kv_heads, buffer_len)
        """
        bsz, num_heads, _, k_len = full_attentions.shape
        num_kv_heads = keys.shape[1]
        num_groups = num_heads // num_kv_heads

        # Sum attention from this chunk's queries to each key position
        # Shape: (batch, num_heads, cache_len)
        attn_to_keys = full_attentions.sum(dim=2)

        # Handle GQA: average over query head groups to get per-kv-head attention
        # Shape: (batch, num_kv_heads, cache_len)
        attn_to_keys = attn_to_keys.view(bsz, num_kv_heads, num_groups, k_len).mean(dim=2)

        if prefilling:
            # Initialize cumulative attention buffer
            self.cumulative_attention[layer_idx] = attn_to_keys
            self.buffer_start_pos[layer_idx] = 0
            self.total_queries_seen[layer_idx] = q_len
        else:
            buffer_start = self.buffer_start_pos[layer_idx]
            buffer_len = self.cumulative_attention[layer_idx].shape[-1]
            buffer_end = buffer_start + buffer_len

            # Update cumulative attention for existing buffer positions
            # attn_to_keys[:, :, buffer_start:buffer_end] is attention to positions currently in buffer
            self.cumulative_attention[layer_idx] += attn_to_keys[:, :, buffer_start:buffer_end]

            # Append new positions (if any) to the buffer
            if cache_len > buffer_end:
                new_attn = attn_to_keys[:, :, buffer_end:cache_len]
                self.cumulative_attention[layer_idx] = torch.cat([
                    self.cumulative_attention[layer_idx],
                    new_attn
                ], dim=-1)

            self.total_queries_seen[layer_idx] += q_len

        # Compute normalized scores: average attention per query
        # This normalizes by how many queries have had a chance to attend
        total_queries = self.total_queries_seen[layer_idx]
        scores = self.cumulative_attention[layer_idx] / total_queries

        # Apply sink token protection if the underlying press has n_sink
        if hasattr(self.press, 'n_sink') and self.press.n_sink > 0:
            buffer_start = self.buffer_start_pos[layer_idx]
            n_sink = self.press.n_sink
            # How many sink positions are still in the buffer?
            n_sinks_in_buffer = max(0, n_sink - buffer_start)
            if n_sinks_in_buffer > 0:
                scores[:, :, :n_sinks_in_buffer] = float("inf")

        return scores

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        cache_len = kwargs["cache_position"][-1] + 1
        prefilling = cache_len == q_len

        # Extract layer index as int for type safety
        layer_idx: int = module.layer_idx  # type: ignore[assignment]

        # Reset buffers if we are in prefilling
        if prefilling and (layer_idx == 0):
            self.scores_buffer.clear()
            self.compression_ratios.clear()
            self.cumulative_attention.clear()
            self.buffer_start_pos.clear()
            self.total_queries_seen.clear()

        # Skip compression during decoding if not enabled
        if not prefilling and not self.decoding:
            return output

        keys, values = extract_keys_and_values(cache, layer_idx)

        # Check if we should use cumulative attention scoring
        use_cumulative = self.accumulate_attention and output[1] is not None

        if use_cumulative:
            # Use full attention matrix to accumulate scores for all buffer positions
            full_attentions = output[1]  # (batch, num_heads, q_len, cache_len)
            scores = self._accumulate_attention_scores(
                layer_idx, full_attentions, keys, q_len, cache_len, prefilling, kwargs
            )
            # scores is already the full buffer, store it
            self.scores_buffer[layer_idx] = scores
        else:
            # Original behavior: score only new tokens
            attentions = output[1][:, :, :, -q_len:] if output[1] is not None else None
            scores = self.press.score(module, hidden_states, keys[:, :, -q_len:], values[:, :, -q_len:], attentions, kwargs)

            # Accumulate scores in the buffer
            if prefilling:
                self.scores_buffer[layer_idx] = scores
            else:
                self.scores_buffer[layer_idx] = torch.cat([self.scores_buffer[layer_idx], scores], dim=-1)

        # Once the buffer exceeds the sliding window, evict tokens with low scores
        if self.scores_buffer[layer_idx].shape[-1] > self.sliding_window_size:
            # Determine how many tokens have left the sliding window and can be evicted
            n_to_evict = self.scores_buffer[layer_idx].shape[-1] - self.sliding_window_size
            scores_to_evict = self.scores_buffer[layer_idx][..., :n_to_evict]
            self.scores_buffer[layer_idx] = self.scores_buffer[layer_idx][..., n_to_evict:]

            # Also trim the cumulative attention buffer if using cumulative mode
            if use_cumulative:
                self.cumulative_attention[layer_idx] = self.cumulative_attention[layer_idx][..., n_to_evict:]
                self.buffer_start_pos[layer_idx] += n_to_evict

            # Determine threshold: either fixed or computed dynamically based on target density
            if self.target_density is not None:
                threshold = self._compute_threshold_for_density(scores_to_evict)
            else:
                threshold = self.threshold

            # Find tokens below threshold: returns (batch_idx, head_idx, token_idx) tuples
            new_masked_key_indices = list(torch.where(scores_to_evict < threshold))

            if len(new_masked_key_indices[0]) > 0:
                # Convert buffer-relative indices to cache-absolute indices
                # During prefill shift=0; during decoding we offset by the number of previously processed tokens
                shift = cache_len - scores_to_evict.shape[2] - self.sliding_window_size
                new_masked_key_indices[-1] += shift

                # Merge new masked indices with existing ones
                # Use getattr to handle case where attribute doesn't exist yet
                existing_indices = getattr(module, "masked_key_indices", None)
                if existing_indices is None:
                    module.masked_key_indices = new_masked_key_indices  # type: ignore[assignment]
                else:
                    module.masked_key_indices = list(  # type: ignore[assignment]
                        torch.cat([i, new_i]) for i, new_i in zip(existing_indices, new_masked_key_indices)
                    )

        # Track compression ratio as the fraction of masked tokens
        # Use getattr to handle case where attribute doesn't exist yet
        masked_indices = getattr(module, "masked_key_indices", None)
        if masked_indices is not None:
            bsz, num_key_value_heads, cache_len, _ = keys.shape
            n_masked = len(masked_indices[0])
            self.compression_ratios[layer_idx] = n_masked / (bsz * num_key_value_heads * cache_len)
        else:
            self.compression_ratios[layer_idx] = 0

        return output
