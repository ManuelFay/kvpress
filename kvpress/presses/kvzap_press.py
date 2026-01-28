# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from kvpress.presses.scorer_press import ScorerPress


class KVzapConfig(PretrainedConfig):
    model_type: str = "kvzap"
    input_dim: int
    output_dim: int
    hidden_dim: Optional[int] = None
    n_modules: int


class KVzapModel(PreTrainedModel):
    config_class = KVzapConfig  # type: ignore[assignment]
    _tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        if config.hidden_dim is None:
            # Linear model
            self.layers = nn.ModuleList(
                [nn.Linear(config.input_dim, config.output_dim) for _ in range(config.n_modules)]
            )
        else:
            # 2-layer MLP model
            self.layers = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, config.output_dim),
                )
                for _ in range(config.n_modules)
            )

    @property
    def all_tied_weights_keys(self):
        return self._tied_weights_keys

    def forward(self, x):
        return torch.stack([module(x[:, i, :]) for i, module in enumerate(self.layers)], dim=1)


@dataclass
class KVzapPress(ScorerPress):
    """
    KVzap (https://arxiv.org/abs/2601.07891) is a fast approximation of KVzip that works
    in both prefilling and decoding. It applies a lightweight surrogate model to the hidden
    states to predict importance scores for every KV pair.
    KVzapPress is designed to be used in conjunction with the DMSPress
    model_type can be "linear" or "mlp".

    Parameters
    ----------
    model_type : Literal["linear", "mlp"], default="mlp"
        Type of KVzap model to use
    n_sink : int, default=4
        Number of initial tokens to always preserve (sink tokens).
        Only used when wrapped in DMSPress (when cache_position is available).
    """

    model_type: Literal["linear", "mlp"] = "mlp"
    n_sink: int = 4
    kvzap_model_name: Optional[str] = field(default=None, init=False)

    def post_init_from_model(self, model):
        kvzap_model_name = f"nvidia/KVzap-{self.model_type}-{model.config.name_or_path.split('/')[-1]}"
        if kvzap_model_name != self.kvzap_model_name:
            self.kvzap_model_name = kvzap_model_name
            self.kvzap_model = KVzapModel.from_pretrained(self.kvzap_model_name)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        kvzap_module = self.kvzap_model.layers[module.layer_idx]
        kvzap_module = kvzap_module.to(hidden_states.device, dtype=hidden_states.dtype).eval()
        with torch.no_grad():
            scores = kvzap_module(hidden_states).transpose(1, 2)

        # If called from DMSPress, protect sink tokens with score=inf
        if "cache_position" in kwargs and kwargs["cache_position"] is not None:
            q_len = keys.shape[2]
            first_pos = kwargs["cache_position"][0].item()

            # Sink tokens (absolute positions 0 to n_sink-1) get score inf (never evicted)
            n_sinks_in_chunk = max(0, min(self.n_sink - first_pos, q_len))
            if n_sinks_in_chunk > 0:
                scores[:, :, :n_sinks_in_chunk] = float("inf")

        return scores
