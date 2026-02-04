#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KV Cache Compression Methods Comparison

This script evaluates and compares different KV cache compression methods on the
same validation dataset to understand their performance-compression tradeoffs.

Compared Methods:
1. Baseline - No compression
2. Random - Random token eviction
3. StreamingLLM - Sliding window + sink tokens
4. Random Validation - Random with threshold=1.0 (should match StreamingLLM exactly)
5. ExpectedAttention - Statistical prediction of future attention
6. ObservedAttention - Historical attention weights (requires eager mode)
7. H2O - Heavy-hitter oracle with local window + observed attention (requires eager mode)
8. KVzap - Fast approximation with learned surrogate models

All methods are evaluated using the same chunked evaluation approach to ensure
fair comparison.

Usage Examples
--------------

Compare all methods (except ObservedAttention/H2O) with default settings:
    python compare_compression_methods.py \\
        --data_path /path/to/govreport.val.jsonl \\
        --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \\
        --max_samples 100 \\
        --chunk_size 1

Validation test (Random with threshold=1.0 should match StreamingLLM):
    python compare_compression_methods.py \\
        --data_path /path/to/data.jsonl \\
        --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \\
        --methods streaming_llm random_validation \\
        --max_samples 10

Fast approximate comparison with chunk_size=128:
    python compare_compression_methods.py \\
        --data_path /path/to/data.jsonl \\
        --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \\
        --chunk_size 128

Compare specific methods only:
    python compare_compression_methods.py \\
        --data_path /path/to/data.jsonl \\
        --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \\
        --methods random streaming_llm kvzap \\
        --max_samples 50

Include ObservedAttention and H2O (requires eager attention, slower):
    python compare_compression_methods.py \\
        --data_path /path/to/data.jsonl \\
        --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \\
        --include_observed_attention \\
        --max_samples 50

Multi-GPU evaluation:
    python compare_compression_methods.py \\
        --data_path /path/to/data.jsonl \\
        --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \\
        --num_gpus -1 \\
        --max_samples 100

Adjust compression aggressiveness with custom thresholds:
    python compare_compression_methods.py \\
        --data_path /path/to/data.jsonl \\
        --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \\
        --thresholds '{"kvzap": -7.0, "expected_attention": 0.05, "random": 0.3}'

Note: Edit DEFAULT_THRESHOLDS dict at the top of the file to change default thresholds

Output
------
The script produces:
1. Console output with comparison table
2. detailed_results.json - Full results for all methods
3. comparison_summary.json - Summary statistics and deltas vs baseline
"""

import json
import torch
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

# Import from kvpress package
from kvpress import (
    RandomPress,
    StreamingLLMPress,
    ExpectedAttentionPress,
    ObservedAttentionPress,
    KVzapPress,
    DMSPress,
)
from kvpress.presses.h2o_press import H2OPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress

# Import from existing evaluation script
from evaluate_ppl_chunked import (
    load_jsonl_dataset,
    calculate_perplexity_chunked,
    CustomKVzapPress,
    DMS_SLIDING_WINDOW_SIZE,
    get_available_gpus,
    aggregate_results,
)
from kvzap.loaders import Sample, create_loader


# ==============================================================================
# Threshold Configuration for Each Method
# ==============================================================================

# Default threshold values for each compression method
# These values can be tuned to adjust the sparsity/performance tradeoff
DEFAULT_THRESHOLDS = {
    # KVzap: Uses log-probabilities (negative values)
    # More negative = less important tokens
    # Range: typically -10 to 0
    "kvzap": -6.5,

    # KVzap without sink tokens (n_sink=0)
    # Same threshold as regular kvzap for fair comparison
    "kvzap_no_sink": -6.5,

    # ExpectedAttention (with vnorm=True): Softmax probabilities × value norms
    # Scores typically in range [0, 10]
    # Higher threshold = more aggressive compression
    "expected_attention": 0.3,

    # Random: Uniform random scores in [0, 1]
    # threshold=0.5 evicts ~50% of tokens
    "random": 0.8,

    # Random validation: threshold=1.0 should match StreamingLLM
    # (keeps only sink + window, evicts everything else)
    "random_validation": 1.0,

    # ObservedAttention: Uses target_density by default (see DEFAULT_DENSITIES)
    # If threshold is needed, typical values are 0.001-0.05
    "observed_attention": None,

    # H2O: Uses target_density by default (see DEFAULT_DENSITIES)
    # If threshold is needed, typical values are 0.001-0.05
    "h2o": None,
}

# Default target densities for methods that benefit from adaptive thresholds
# target_density = fraction of compressible tokens to keep (0.0 to 1.0)
# This is preferred over fixed thresholds for observed attention methods
# because attention score distributions vary with sequence length.
DEFAULT_DENSITIES = {
    # ObservedAttention: Keep 30% of compressible tokens
    "observed_attention": 0.2,

    # H2O: Keep 30% of compressible tokens (same as observed_attention)
    "h2o": 0.2,
}



DEFAULT_THRESHOLDS = {
    # KVzap: Uses log-probabilities (negative values)
    # More negative = less important tokens
    # Range: typically -10 to 0
    "kvzap": -5,

    # KVzap without sink tokens (n_sink=0)
    # Same threshold as regular kvzap for fair comparison
    "kvzap_no_sink": -5,

    # ExpectedAttention (with vnorm=True): Softmax probabilities × value norms
    # Scores typically in range [0, 10]
    # Higher threshold = more aggressive compression
    "expected_attention": 0.35,

    # Random: Uniform random scores in [0, 1]
    # threshold=0.5 evicts ~50% of tokens
    "random": 0.85,
    # Random validation: threshold=1.0 should match StreamingLLM
    # (keeps only sink + window, evicts everything else)
    "random_validation": 1.0,

    # ObservedAttention: Uses target_density by default (see DEFAULT_DENSITIES)
    # If threshold is needed, typical values are 0.001-0.05
    "observed_attention": None,

    # H2O: Uses target_density by default (see DEFAULT_DENSITIES)
    # If threshold is needed, typical values are 0.001-0.05
    "h2o": None,
}

# Default target densities for methods that benefit from adaptive thresholds
# target_density = fraction of compressible tokens to keep (0.0 to 1.0)
# This is preferred over fixed thresholds for observed attention methods
# because attention score distributions vary with sequence length.
DEFAULT_DENSITIES = {
    # ObservedAttention: Keep 30% of compressible tokens
    "observed_attention": 0.15,

    # H2O: Keep 30% of compressible tokens (same as observed_attention)
    "h2o": 0.15,
}



# ==============================================================================
# Visualization Functions
# ==============================================================================

def visualize_compression_heatmap(
    scores_dict: Dict[str, torch.Tensor],
    threshold: Optional[float],
    method_name: str,
    output_dir: Path,
    max_tokens: int = 1024,
    sliding_window_size: int = 128,
    n_sink: int = 0,
    target_density: Optional[float] = None,
):
    """
    Create a heatmap visualization showing which tokens are kept/evicted per layer.

    Parameters
    ----------
    scores_dict : dict[int, torch.Tensor]
        Dictionary mapping layer_idx -> binary decisions tensor (batch, num_heads, seq_len)
        where 1 = kept, 0 = evicted (already computed, not raw scores)
    threshold : float, optional
        Threshold value used (for display purposes only)
    method_name : str
        Name of the compression method
    output_dir : Path
        Directory to save the visualization
    max_tokens : int
        Maximum number of tokens to visualize (should match actual data)
    sliding_window_size : int
        Size of the sliding window (these tokens are protected)
    n_sink : int
        Number of sink tokens (these tokens are protected)
    target_density : float, optional
        Target density used (for display purposes only)
    """
    if not scores_dict:
        print(f"  No scores to visualize for {method_name}")
        return

    # Collect binary decisions from all layers
    all_layer_decisions = []
    for layer_idx in sorted(scores_dict.keys()):
        layer_decisions = scores_dict[layer_idx]  # (batch, num_heads, seq_len) - already binary
        if layer_decisions is not None:
            all_layer_decisions.append(layer_decisions.cpu())

    if not all_layer_decisions:
        print(f"  No valid decisions for {method_name}")
        return

    # Stack all layers: (num_layers, batch, num_heads, seq_len)
    decisions_tensor = torch.stack(all_layer_decisions, dim=0)

    # Take first batch element: (num_layers, num_heads, seq_len)
    decisions_tensor = decisions_tensor[:, 0, :, :]

    seq_len = decisions_tensor.shape[2]
    num_layers = decisions_tensor.shape[0]
    num_heads = decisions_tensor.shape[1]

    # Convert to float32 and numpy
    decisions_tensor = decisions_tensor.float().cpu()

    # decisions_tensor shape: (num_layers, num_heads, seq_len)
    # Each element is 1 (kept) or 0 (evicted) at the KEY level (layer, head, position)

    # Count keys at the granular level (layer × head × position)
    # Total keys = num_layers × num_heads × seq_len
    total_keys = num_layers * num_heads * seq_len

    # Count how many keys are kept (sum all 1s in the tensor)
    keys_kept_total = decisions_tensor.sum().item()

    # Define protected regions at the TOKEN level
    is_protected_tokens = np.zeros(seq_len, dtype=bool)
    if n_sink > 0:
        is_protected_tokens[:min(n_sink, seq_len)] = True  # Sink tokens
    if sliding_window_size > 0 and seq_len > sliding_window_size:
        is_protected_tokens[-sliding_window_size:] = True  # Sliding window

    # Calculate compressible region at KEY level
    num_protected_tokens = is_protected_tokens.sum()
    num_compressible_tokens = seq_len - num_protected_tokens

    # Total keys in each region
    total_protected_keys = num_protected_tokens * num_layers * num_heads
    total_compressible_keys = num_compressible_tokens * num_layers * num_heads

    # Count keys kept in compressible region
    # For each position that's NOT protected, sum kept keys across all layers and heads
    compressible_mask = ~is_protected_tokens  # Shape: (seq_len,)
    keys_kept_in_compressible = decisions_tensor[:, :, compressible_mask].sum().item()

    # Calculate densities at KEY level
    density_overall = (keys_kept_total / total_keys * 100) if total_keys > 0 else 0
    density_compressible = (keys_kept_in_compressible / total_compressible_keys * 100) if total_compressible_keys > 0 else 0

    # For visualization, average across heads for each layer: (num_layers, seq_len)
    # This gives us the fraction of heads that kept each token at each layer
    layer_avg_decisions = decisions_tensor.mean(dim=1).numpy()

    # Check if we have non-binary values (indicating head disagreement)
    non_binary_mask = (layer_avg_decisions > 0.01) & (layer_avg_decisions < 0.99)
    num_non_binary = non_binary_mask.sum()
    if num_non_binary > 0:
        print(f"    Found {num_non_binary} layer-token pairs where heads disagree on eviction")
    else:
        print(f"    All decisions are binary (all heads agree at every layer)")

    # NOW force protected regions to be shown as kept in the visualization
    # (DMSPress protects them regardless of scores)
    decisions_for_viz = layer_avg_decisions.copy()
    if n_sink > 0:
        decisions_for_viz[:, :min(n_sink, seq_len)] = 1.0  # Force sinks green
    if sliding_window_size > 0 and seq_len > sliding_window_size:
        decisions_for_viz[:, -sliding_window_size:] = 1.0  # Force window green

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10),
                                         gridspec_kw={'height_ratios': [5, 2, 1]})

    # Plot 1: Per-layer heatmap (layers x tokens)
    # Use decisions_for_viz which has protected regions forced to 1.0
    # Use gradient colormap to show head disagreement (0=all evicted, 0.5=half kept, 1=all kept)
    sns.heatmap(decisions_for_viz,
                cmap='RdYlGn',  # Red -> Yellow -> Green gradient (reversed for better aesthetics)
                cbar_kws={'label': 'Fraction of heads keeping token'},
                ax=ax1,
                yticklabels=[f'L{i}' if i % 4 == 0 else '' for i in range(num_layers)],
                xticklabels=False,
                vmin=0, vmax=1)
    ax1.set_ylabel('Layer', fontsize=10)
    ax1.set_xlabel('Token Position', fontsize=10)

    # Enhanced title with density metrics
    if target_density is not None:
        param_str = f'target_density={target_density:.2f}'
    elif threshold is not None:
        param_str = f'threshold={threshold:.6f}'
    else:
        param_str = 'default params'
    title_line1 = f'{method_name} - Compression State at {seq_len} Tokens ({param_str})'
    title_line2 = f'Overall Density: {density_overall:.2f}% ({int(keys_kept_total)}/{total_keys} keys) | Compressible Region Density: {density_compressible:.2f}% ({int(keys_kept_in_compressible)}/{total_compressible_keys} keys)'
    ax1.set_title(f'{title_line1}\n{title_line2}', fontsize=11, pad=10)

    # Add vertical lines to mark protected regions with prettier colors
    if n_sink > 0:
        ax1.axvline(x=n_sink, color='#3498db', linestyle='--', linewidth=2.5, alpha=0.7, label=f'Sink boundary (n={n_sink})')
    if sliding_window_size > 0 and seq_len > sliding_window_size:
        window_start = seq_len - sliding_window_size
        ax1.axvline(x=window_start, color='#e67e22', linestyle='--', linewidth=2.5, alpha=0.7,
                   label=f'Window start (size={sliding_window_size})')

    if n_sink > 0 or (sliding_window_size > 0 and seq_len > sliding_window_size):
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Plot 2: Average across layers (single row heatmap)
    avg_for_viz = decisions_for_viz.mean(axis=0).reshape(1, -1)
    sns.heatmap(avg_for_viz,
                cmap='RdYlGn',  # Matching gradient
                cbar_kws={'label': 'Avg fraction kept'},
                ax=ax2,
                yticklabels=['Avg'],
                xticklabels=False,
                vmin=0, vmax=1)
    ax2.set_xlabel('Token Position', fontsize=10)
    ax2.set_title('Average Across All Layers', fontsize=10)

    # Add vertical lines with matching colors
    if n_sink > 0:
        ax2.axvline(x=n_sink, color='#3498db', linestyle='--', linewidth=2.5, alpha=0.7)
    if sliding_window_size > 0 and seq_len > sliding_window_size:
        ax2.axvline(x=seq_len - sliding_window_size, color='#e67e22', linestyle='--', linewidth=2.5, alpha=0.7)

    # Plot 3: Protected regions visualization with prettier colors
    protected_data = is_protected_tokens.astype(np.float32).reshape(1, -1)
    sns.heatmap(protected_data,
                cmap=['#ecf0f1', '#3498db'],  # Light gray for compressible, Nice blue for protected
                cbar_kws={'label': 'Region type', 'ticks': [0, 1]},
                ax=ax3,
                yticklabels=['Protected'],
                xticklabels=False,
                vmin=0, vmax=1)
    ax3.set_xlabel('Token Position', fontsize=10)
    ax3.set_title('Protected Regions (Blue = Sinks + Window, Gray = Compressible)', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"{method_name.replace('(', '_').replace(')', '_').replace('=', '_')}_compression_heatmap.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f"  Saved visualization to: {output_file}")
    print(f"    Tokens visualized: {seq_len}")
    print(f"    Total keys: {total_keys} ({num_layers} layers × {num_heads} heads × {seq_len} tokens)")
    print(f"    Overall density: {density_overall:.2f}% ({int(keys_kept_total)}/{total_keys} keys kept)")
    print(f"    Compressible region density: {density_compressible:.2f}% ({int(keys_kept_in_compressible)}/{total_compressible_keys} keys kept)")
    print(f"    Protected: {total_protected_keys} keys (sinks={min(n_sink, seq_len)} tokens × {num_layers} × {num_heads}, window={min(sliding_window_size, seq_len) if sliding_window_size > 0 and seq_len > sliding_window_size else 0} tokens × {num_layers} × {num_heads})")


def capture_scores_first_sample(
    model,
    tokenizer,
    text,  # Can be str, Sample, or serialized dict
    press,
    max_length: int = 8192,
    chunk_size: int = 128,
    device: str = "cuda",
    target_tokens: int = 1024,
) -> Dict[int, torch.Tensor]:
    """
    Process first sample and capture which tokens were kept/evicted.

    Returns a binary tensor for each layer showing which of the first target_tokens
    were kept (1) or evicted (0).

    Parameters
    ----------
    text : str, Sample, or dict
        Either a text string (will be tokenized), a Sample object (uses pre-tokenized),
        or a serialized sample dict (from Sample.to_serializable())
    target_tokens : int
        Number of tokens to process before capturing (default: 1024)

    Returns
    -------
    dict[int, torch.Tensor]
        Dictionary mapping layer_idx -> binary tensor (batch, num_heads, target_tokens)
        where 1 = kept, 0 = evicted
    """
    model.eval()

    # Handle different input types
    if isinstance(text, str):
        # Regular text string - tokenize it
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)
    elif isinstance(text, dict) and "tokens" in text:
        # Serialized sample dict
        tokens = text["tokens"]
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        input_ids = torch.tensor([tokens], device=device)
    elif hasattr(text, "tokens"):
        # Sample object
        tokens = text.tokens
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        input_ids = torch.tensor([tokens], device=device)
    else:
        raise ValueError(f"Unsupported text type: {type(text)}. Expected str, Sample, or dict with 'tokens' key.")

    seq_len = input_ids.size(1)

    if seq_len < 2:
        return {}

    from transformers import DynamicCache
    cache = DynamicCache()

    # Calculate how many chunks we need to process to reach target_tokens
    num_chunks_to_process = (target_tokens + chunk_size - 1) // chunk_size
    # But don't exceed the actual sequence length
    max_possible_chunks = (seq_len + chunk_size - 1) // chunk_size
    num_chunks_to_process = min(num_chunks_to_process, max_possible_chunks)

    with torch.inference_mode():
        # CRITICAL: Keep press context open across ALL chunks
        # This ensures masked_key_indices persist between chunks for correct eviction
        if press is not None:
            with press(model):
                for chunk_idx in range(num_chunks_to_process):
                    start_pos = chunk_idx * chunk_size
                    end_pos = min((chunk_idx + 1) * chunk_size, seq_len)
                    chunk_ids = input_ids[:, start_pos:end_pos]

                    position_ids = torch.arange(start_pos, end_pos, device=device).unsqueeze(0)
                    cache_position = torch.arange(start_pos, end_pos, device=device)

                    outputs = model(
                        input_ids=chunk_ids,
                        position_ids=position_ids,
                        past_key_values=cache,
                        cache_position=cache_position,
                        use_cache=True,
                        output_attentions=True,  # Required for H2O and ObservedAttention
                        return_dict=True,
                    )

                # CRITICAL: Capture binary decisions INSIDE the press context
                # masked_key_indices is cleared when context exits, so we must capture here
                actual_tokens_processed = min(num_chunks_to_process * chunk_size, seq_len)
                print(f"    Processed {num_chunks_to_process} chunks = {actual_tokens_processed} tokens for visualization")

                # Reconstruct which positions were kept/evicted by looking at masked_key_indices
                binary_decisions = {}
                num_kv_heads = model.config.num_key_value_heads
                batch_size = 1

                for layer_idx, layer in enumerate(model.model.layers):
                    attn_module = layer.self_attn

                    # Check if this layer has masked indices (evicted positions)
                    if hasattr(attn_module, 'masked_key_indices') and attn_module.masked_key_indices is not None:
                        # Start with all positions kept (1)
                        kept_mask = torch.ones(batch_size, num_kv_heads, actual_tokens_processed,
                                              dtype=torch.float32, device=device)

                        # Mark evicted positions as 0
                        if len(attn_module.masked_key_indices) >= 3:
                            batch_idx = attn_module.masked_key_indices[0]
                            head_idx = attn_module.masked_key_indices[1]
                            pos_idx = attn_module.masked_key_indices[2]

                            # Only include positions < actual_tokens_processed
                            valid_mask = pos_idx < actual_tokens_processed
                            if valid_mask.any():
                                kept_mask[batch_idx[valid_mask], head_idx[valid_mask], pos_idx[valid_mask]] = 0.0

                        binary_decisions[layer_idx] = kept_mask.cpu()
                    else:
                        # No eviction happened - all tokens kept
                        binary_decisions[layer_idx] = torch.ones(batch_size, num_kv_heads, actual_tokens_processed,
                                                                 dtype=torch.float32)
        else:
            # Baseline - no compression
            for chunk_idx in range(num_chunks_to_process):
                start_pos = chunk_idx * chunk_size
                end_pos = min((chunk_idx + 1) * chunk_size, seq_len)
                chunk_ids = input_ids[:, start_pos:end_pos]

                position_ids = torch.arange(start_pos, end_pos, device=device).unsqueeze(0)
                cache_position = torch.arange(start_pos, end_pos, device=device)

                outputs = model(
                    input_ids=chunk_ids,
                    position_ids=position_ids,
                    past_key_values=cache,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True,
                )

            # Baseline: all tokens kept
            actual_tokens_processed = min(num_chunks_to_process * chunk_size, seq_len)
            print(f"    Processed {num_chunks_to_process} chunks = {actual_tokens_processed} tokens for visualization")

            binary_decisions = {}
            num_kv_heads = model.config.num_key_value_heads
            batch_size = 1
            for layer_idx in range(model.config.num_hidden_layers):
                binary_decisions[layer_idx] = torch.ones(batch_size, num_kv_heads, actual_tokens_processed,
                                                         dtype=torch.float32)

    if binary_decisions:
        first_layer = next(iter(binary_decisions.values()))
        print(f"    Captured binary decisions shape: {first_layer.shape} (1=kept, 0=evicted)")
        print(f"    Tokens in visualization: {first_layer.shape[2]}")

        # Check if decisions vary across heads
        if first_layer.shape[1] > 1:
            # Compare head 0 to head 1
            head_agreement = (first_layer[0, 0, :] == first_layer[0, 1, :]).float().mean().item()
            print(f"    Head agreement (head 0 vs head 1): {head_agreement*100:.1f}% identical decisions")

    return binary_decisions


# ==============================================================================
# Phase 1: Press Factory Functions
# ==============================================================================

def create_random_press(threshold: float, sliding_window_size: int = DMS_SLIDING_WINDOW_SIZE, seed: int = 42, n_sink: int = 4):
    """
    Create RandomPress wrapped in DMSPress for threshold-based eviction.

    Parameters
    ----------
    threshold : float
        Score threshold for eviction (tokens with score < threshold are evicted)
    sliding_window_size : int
        Number of recent tokens protected from eviction
    seed : int
        Random seed for reproducibility
    n_sink : int
        Number of initial tokens to always preserve (sink tokens)

    Returns
    -------
    DMSPress wrapping RandomPress
    """
    random_scorer = RandomPress(compression_ratio=0.0, seed=seed, n_sink=n_sink)
    return DMSPress(
        random_scorer,
        threshold=threshold,
        sliding_window_size=sliding_window_size,
        decoding=True,
    )


def create_streaming_llm_press(n_sink: int = 4, sliding_window_size: int = 128):
    """
    Create StreamingLLMPress wrapped in DMSPress for threshold-based eviction.

    StreamingLLM assigns scores of 1.0 to sink tokens and recent window tokens,
    and 0.0 to all other tokens. Wrapping in DMSPress with threshold=0.5 ensures
    that only tokens with scores of 1.0 are kept (sink + recent window).

    Parameters
    ----------
    n_sink : int
        Number of initial tokens to always preserve (attention sinks)
    sliding_window_size : int
        Number of recent tokens to preserve (default: 128)

    Returns
    -------
    DMSPress wrapping StreamingLLMPress
    """
    streaming_scorer = StreamingLLMPress(compression_ratio=0.0, n_sink=n_sink, sliding_window_size=sliding_window_size)
    return DMSPress(
        streaming_scorer,
        threshold=0.5,
        sliding_window_size=sliding_window_size,
        decoding=True,
    )


def create_expected_attention_press(
    threshold: float,
    sliding_window_size: int = DMS_SLIDING_WINDOW_SIZE,
    n_future_positions: int = 512,
    n_sink: int = 4,
    use_covariance: bool = True,
    use_vnorm: bool = True,
):
    """
    Create ExpectedAttentionPress wrapped in DMSPress for threshold-based eviction.

    Parameters
    ----------
    threshold : float
        Score threshold for eviction. Tokens with scores below this value are evicted.
        With use_vnorm=True (default), scores are softmax probabilities × value norms,
        typically in range [0, 10]. Good starting values: 0.05-0.2
    sliding_window_size : int
        Number of recent tokens protected from eviction
    n_future_positions : int
        Number of future positions to consider
    n_sink : int
        Number of sink tokens to preserve
    use_covariance : bool
        Include covariance in expected attention computation
    use_vnorm : bool
        Rescale scores by value norms

    Returns
    -------
    DMSPress wrapping ExpectedAttentionPress
    """
    expected_scorer = ExpectedAttentionPress(
        compression_ratio=0.0,
        n_future_positions=n_future_positions,
        n_sink=n_sink,
        use_covariance=use_covariance,
        use_vnorm=use_vnorm,
    )
    return DMSPress(
        expected_scorer,
        threshold=threshold,
        sliding_window_size=sliding_window_size,
        decoding=True,
    )


def create_observed_attention_press(
    threshold: Optional[float] = None,
    target_density: Optional[float] = None,
    sliding_window_size: int = DMS_SLIDING_WINDOW_SIZE,
):
    """
    Create ObservedAttentionPress wrapped in DMSPress for threshold-based eviction.

    NOTE: Requires model loaded with attn_implementation="eager"

    Parameters
    ----------
    threshold : float, optional
        Score threshold for eviction. Mutually exclusive with target_density.
    target_density : float, optional
        Target fraction of tokens to keep (0.0 to 1.0). Mutually exclusive with threshold.
        Recommended for consistent compression across different sequence lengths.
    sliding_window_size : int
        Number of recent tokens protected from eviction

    Returns
    -------
    DMSPress wrapping ObservedAttentionPress
    """
    observed_scorer = ObservedAttentionPress(compression_ratio=0.0)
    return DMSPress(
        observed_scorer,
        threshold=threshold,
        target_density=target_density,
        sliding_window_size=sliding_window_size,
        decoding=True,
        accumulate_attention=True,  # Accumulate attention across chunks for proper scoring
    )


def create_h2o_press(
    threshold: Optional[float] = None,
    target_density: Optional[float] = None,
    sliding_window_size: int = DMS_SLIDING_WINDOW_SIZE,
    local_window_size: int = 512,
    n_sink: int = 4,
):
    """
    Create H2O-style press using ObservedAttentionPress wrapped in DMSPress.

    Uses observed attention scores for importance, with DMSPress handling
    the sliding window protection and ObservedAttentionPress protecting sink tokens.

    H2O (Heavy-Hitter Oracle) keeps:
    - Sink tokens: First n_sink tokens (attention sinks, always kept)
    - Heavy hitters: Tokens with high cumulative attention (score >= threshold)
    - Recent window: Last sliding_window_size tokens

    NOTE: Requires model loaded with attn_implementation="eager"

    Parameters
    ----------
    threshold : float, optional
        Score threshold for eviction. Tokens with score < threshold are evicted.
        Mutually exclusive with target_density.
    target_density : float, optional
        Target fraction of tokens to keep (0.0 to 1.0). Mutually exclusive with threshold.
        Recommended for consistent compression across different sequence lengths.
    sliding_window_size : int
        Number of recent tokens protected from eviction by DMSPress
    local_window_size : int
        (Unused - kept for API compatibility) DMSPress handles window protection.
    n_sink : int
        Number of initial "sink" tokens to always preserve (default 4 for H2O).
        These act as attention sinks where models tend to dump extra attention.

    Returns
    -------
    DMSPress wrapping ObservedAttentionPress with sink token protection
    """
    # Use ObservedAttentionPress with sink tokens - DMSPress's sliding_window handles recent token protection
    observed_attention_scorer = ObservedAttentionPress(compression_ratio=0.0, n_sink=n_sink)
    return DMSPress(
        observed_attention_scorer,
        threshold=threshold,
        target_density=target_density,
        sliding_window_size=sliding_window_size,
        decoding=True,
        accumulate_attention=True,  # Accumulate attention across chunks for proper H2O scoring
    )



def create_kvzap_press(
    threshold: float,
    kvzap_model_type: str = "mlp",
    kvzap_scorer_model: Optional[str] = None,
    sliding_window_size: int = DMS_SLIDING_WINDOW_SIZE,
    n_sink: int = 4,
):
    """
    Create KVzap press wrapped in DMSPress for threshold-based eviction.

    Parameters
    ----------
    threshold : float
        Score threshold for eviction (more negative = more aggressive)
    kvzap_model_type : str
        Type of KVzap model ("mlp" or "linear")
    kvzap_scorer_model : str, optional
        Explicit HuggingFace model name for KVzap scorer
    sliding_window_size : int
        Number of recent tokens protected from eviction
    n_sink : int
        Number of initial tokens to always preserve (sink tokens)

    Returns
    -------
    DMSPress wrapping KVzapPress
    """
    if kvzap_scorer_model is not None:
        kvzap_press = CustomKVzapPress(
            model_type=kvzap_model_type,
            explicit_model_name=kvzap_scorer_model,
            n_sink=n_sink
        )
    else:
        kvzap_press = KVzapPress(model_type=kvzap_model_type, n_sink=n_sink)

    return DMSPress(
        kvzap_press,
        threshold=threshold,
        sliding_window_size=sliding_window_size,
        decoding=True,
    )


def create_kvzap_no_sink_press(
    threshold: float,
    kvzap_model_type: str = "mlp",
    kvzap_scorer_model: Optional[str] = None,
    sliding_window_size: int = DMS_SLIDING_WINDOW_SIZE,
):
    """
    Create KVzap press WITHOUT sink tokens (n_sink=0).

    This variant does not preserve initial tokens, only the sliding window.
    Useful for comparing the impact of sink tokens on performance.

    Parameters
    ----------
    threshold : float
        Score threshold for eviction (more negative = more aggressive)
    kvzap_model_type : str
        Type of KVzap model ("mlp" or "linear")
    kvzap_scorer_model : str, optional
        Explicit HuggingFace model name for KVzap scorer
    sliding_window_size : int
        Number of recent tokens protected from eviction

    Returns
    -------
    DMSPress wrapping KVzapPress with n_sink=0
    """
    if kvzap_scorer_model is not None:
        kvzap_press = CustomKVzapPress(
            model_type=kvzap_model_type,
            explicit_model_name=kvzap_scorer_model,
            n_sink=0  # No sink tokens
        )
    else:
        kvzap_press = KVzapPress(model_type=kvzap_model_type, n_sink=0)

    return DMSPress(
        kvzap_press,
        threshold=threshold,
        sliding_window_size=sliding_window_size,
        decoding=True,
    )


# ==============================================================================
# Multi-GPU Support
# ==============================================================================

def worker_evaluate_all_methods(
    gpu_id: int,
    texts: List[str],
    model_name: str,
    max_length: int,
    chunk_size: int,
    methods_config: List[Dict[str, Any]],
    queue: mp.Queue,
):
    """
    Worker function to process a subset of texts on a specific GPU for ALL methods.

    This runs in a separate process and evaluates its assigned texts on the
    specified GPU with ALL compression methods sequentially, keeping the model
    loaded. Results are returned via a multiprocessing queue.

    Parameters
    ----------
    gpu_id : int
        GPU index to use (e.g., 0, 1, 2, 3)
    texts : list[str]
        Subset of texts to process on this GPU
    model_name : str
        HuggingFace model identifier
    max_length : int
        Maximum sequence length
    chunk_size : int
        Chunk size for evaluation
    methods_config : list[dict]
        List of method configurations, each with:
        - "method_name": str - Name of the method
        - "press_type": str or None - Type of press
        - "press_params": dict or None - Parameters for press creation
        - "requires_eager": bool - Whether method requires eager attention
    queue : mp.Queue
        Queue to return results
    """
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        # Determine input representation (raw text vs serialized samples)
        worker_samples = None
        use_serialized_samples = False
        if texts:
            first_item = texts[0]
            if isinstance(first_item, Sample):
                worker_samples = list(texts)
                use_serialized_samples = True
            elif isinstance(first_item, dict) and "tokens" in first_item:
                worker_samples = [
                    sample if isinstance(sample, Sample) else Sample.from_serializable(sample)
                    for sample in texts
                ]
                use_serialized_samples = True

        # Determine if any method requires eager attention
        any_requires_eager = any(m.get("requires_eager", False) for m in methods_config)

        # Load tokenizer once
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model once (use eager if any method needs it)
        if any_requires_eager:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": gpu_id},
                attn_implementation="eager",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": gpu_id},
            )
        model.eval()

        # Evaluate all methods with the same loaded model
        all_method_results = {}
        for idx, method_cfg in enumerate(methods_config, 1):
            method_name = method_cfg["method_name"]
            press_type = method_cfg.get("press_type")
            press_params = method_cfg.get("press_params")

            # Clear progress indicator
            print(f"[GPU {gpu_id}] Processing method {idx}/{len(methods_config)}: {method_name}")
            print(f"[GPU {gpu_id}] {'='*60}")

            # Create the press based on type
            press = None
            if press_type is not None:
                if press_type == "random":
                    press = create_random_press(**press_params)
                elif press_type == "streaming_llm":
                    press = create_streaming_llm_press(**press_params)
                elif press_type == "expected_attention":
                    press = create_expected_attention_press(**press_params)
                elif press_type == "kvzap":
                    press = create_kvzap_press(**press_params)
                elif press_type == "kvzap_no_sink":
                    press = create_kvzap_no_sink_press(**press_params)
                elif press_type == "observed_attention":
                    press = create_observed_attention_press(**press_params)
                elif press_type == "h2o":
                    press = create_h2o_press(**press_params)
                else:
                    raise ValueError(f"Unknown press type: {press_type}")

            results = calculate_perplexity_chunked(
                model=model,
                tokenizer=tokenizer,
                texts=None if use_serialized_samples else texts,
                samples=worker_samples if use_serialized_samples else None,
                max_length=max_length,
                chunk_size=chunk_size,
                press=press,
                device=device,
                disable_tqdm=True,  # Disable tqdm in multi-GPU mode to reduce clutter
            )
            all_method_results[method_name] = results

            # Print completion message
            ppl = results.get("perplexity", 0)
            nll = results.get("avg_nll", 0)
            print(f"[GPU {gpu_id}] ✓ {method_name} complete: PPL={ppl:.4f}, NLL={nll:.6f}")
            print()

        queue.put({
            "success": True,
            "gpu_id": gpu_id,
            "results": all_method_results,
            "num_texts": len(texts)
        })

    except Exception as e:
        import traceback
        queue.put({
            "success": False,
            "gpu_id": gpu_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def worker_evaluate_texts(
    gpu_id: int,
    texts: List[str],
    model_name: str,
    max_length: int,
    chunk_size: int,
    press_type: Optional[str],
    press_params: Optional[Dict[str, Any]],
    queue: mp.Queue,
    model_cache: Optional[Dict] = None,
):
    """
    Worker function to process a subset of texts on a specific GPU.

    This runs in a separate process and evaluates its assigned texts on the
    specified GPU with the given compression method. Results are returned via
    a multiprocessing queue.

    Parameters
    ----------
    gpu_id : int
        GPU index to use (e.g., 0, 1, 2, 3)
    texts : list[str]
        Subset of texts to process on this GPU
    model_name : str
        HuggingFace model identifier
    max_length : int
        Maximum sequence length
    chunk_size : int
        Chunk size for evaluation
    press_type : str, optional
        Type of press to create: "random", "streaming_llm", "expected_attention",
        "observed_attention", "h2o", "kvzap", or None for baseline
    press_params : dict, optional
        Parameters to pass to the press factory function
    queue : mp.Queue
        Queue to return results
    model_cache : dict, optional
        Not used in subprocess (kept for API compatibility)
    """
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        # Determine input representation (raw text vs serialized samples)
        worker_samples = None
        use_serialized_samples = False
        if texts:
            first_item = texts[0]
            if isinstance(first_item, Sample):
                worker_samples = list(texts)
                use_serialized_samples = True
            elif isinstance(first_item, dict) and "tokens" in first_item:
                worker_samples = [
                    sample if isinstance(sample, Sample) else Sample.from_serializable(sample)
                    for sample in texts
                ]
                use_serialized_samples = True

        # Determine if we need eager attention mode
        requires_eager = press_type in ["observed_attention", "h2o"]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if requires_eager:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": gpu_id},
                attn_implementation="eager",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": gpu_id},
            )
        model.eval()

        # Create the press based on type
        press = None
        if press_type is not None:
            if press_type == "random":
                press = create_random_press(**press_params)
            elif press_type == "streaming_llm":
                press = create_streaming_llm_press(**press_params)
            elif press_type == "expected_attention":
                press = create_expected_attention_press(**press_params)
            elif press_type == "kvzap":
                press = create_kvzap_press(**press_params)
            elif press_type == "kvzap_no_sink":
                press = create_kvzap_no_sink_press(**press_params)
            elif press_type == "observed_attention":
                press = create_observed_attention_press(**press_params)
            elif press_type == "h2o":
                press = create_h2o_press(**press_params)
            else:
                raise ValueError(f"Unknown press type: {press_type}")

        results = calculate_perplexity_chunked(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=max_length,
            chunk_size=chunk_size,
            press=press,
            device=device,
        )

        queue.put({
            "success": True,
            "gpu_id": gpu_id,
            "results": results,
            "num_texts": len(texts)
        })

    except Exception as e:
        import traceback
        queue.put({
            "success": False,
            "gpu_id": gpu_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def evaluate_multigpu(
    texts: List[str],
    model_name: str,
    max_length: int,
    chunk_size: int,
    press_type: Optional[str],
    press_params: Optional[Dict[str, Any]],
    num_gpus: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate perplexity using multiple GPUs in parallel.

    Parameters
    ----------
    texts : list[str]
        All texts to evaluate
    model_name : str
        HuggingFace model identifier
    max_length : int
        Maximum sequence length
    chunk_size : int
        Chunk size for evaluation
    press_type : str, optional
        Type of press ("random", "streaming_llm", "expected_attention", "kvzap", or None for baseline)
    press_params : dict, optional
        Parameters for press creation
    num_gpus : int, optional
        Number of GPUs to use. If None, uses all available GPUs.

    Returns
    -------
    dict
        Aggregated results from all GPUs
    """
    available_gpus = get_available_gpus()
    if not available_gpus:
        raise RuntimeError("No GPUs available for parallel evaluation")

    gpus_to_use = available_gpus if num_gpus is None else available_gpus[:num_gpus]
    num_workers = len(gpus_to_use)
    print(f"\nUsing {num_workers} GPU(s) for parallel evaluation: {gpus_to_use}")

    # Split texts across GPUs
    texts_per_gpu = len(texts) // num_workers
    text_splits = []
    for i in range(num_workers):
        start_idx = i * texts_per_gpu
        end_idx = start_idx + texts_per_gpu if i < num_workers - 1 else len(texts)
        text_splits.append(texts[start_idx:end_idx])

    for i, split in enumerate(text_splits):
        print(f"  GPU {gpus_to_use[i]}: {len(split)} texts")

    ctx = mp.get_context('spawn')
    queue = ctx.Queue()

    processes = []
    for i, gpu_id in enumerate(gpus_to_use):
        p = ctx.Process(
            target=worker_evaluate_texts,
            args=(gpu_id, text_splits[i], model_name, max_length, chunk_size,
                  press_type, press_params, queue)
        )
        p.start()
        processes.append(p)

    worker_results = []
    for _ in range(num_workers):
        result = queue.get()
        if not result["success"]:
            for p in processes:
                p.terminate()
            error_msg = f"Worker failed on GPU {result['gpu_id']}: {result['error']}"
            if "traceback" in result:
                error_msg += f"\n{result['traceback']}"
            raise RuntimeError(error_msg)
        worker_results.append(result)
        print(f"  GPU {result['gpu_id']} completed: {result['num_texts']} texts")

    for p in processes:
        p.join()

    # Aggregate results from all workers
    return aggregate_results(worker_results)


def evaluate_all_methods_multigpu(
    texts: List[str],
    model_name: str,
    max_length: int,
    chunk_size: int,
    methods_config: List[Dict[str, Any]],
    num_gpus: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate ALL methods using multiple GPUs in parallel, loading model only once per GPU.

    This is more efficient than calling evaluate_multigpu multiple times because
    each GPU worker loads the model once and evaluates all methods sequentially.

    Parameters
    ----------
    texts : list[str]
        All texts to evaluate
    model_name : str
        HuggingFace model identifier
    max_length : int
        Maximum sequence length
    chunk_size : int
        Chunk size for evaluation
    methods_config : list[dict]
        List of method configurations, each with:
        - "method_name": str - Name of the method (used as key in results)
        - "press_type": str or None - Type of press
        - "press_params": dict or None - Parameters for press creation
        - "requires_eager": bool - Whether method requires eager attention
    num_gpus : int, optional
        Number of GPUs to use. If None, uses all available GPUs.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping method_name -> aggregated results
    """
    available_gpus = get_available_gpus()
    if not available_gpus:
        raise RuntimeError("No GPUs available for parallel evaluation")

    gpus_to_use = available_gpus if num_gpus is None else available_gpus[:num_gpus]
    num_workers = len(gpus_to_use)

    print()
    print("="*80)
    print(f"MULTI-GPU EVALUATION")
    print("="*80)
    print(f"GPUs: {num_workers} workers on {gpus_to_use}")
    print(f"Methods: {len(methods_config)} total")
    print(f"Samples: {len(texts)} total")
    print()

    # Split texts across GPUs
    texts_per_gpu = len(texts) // num_workers
    text_splits = []
    for i in range(num_workers):
        start_idx = i * texts_per_gpu
        end_idx = start_idx + texts_per_gpu if i < num_workers - 1 else len(texts)
        text_splits.append(texts[start_idx:end_idx])

    print("Sample distribution:")
    for i, split in enumerate(text_splits):
        print(f"  GPU {gpus_to_use[i]}: {len(split)} samples")
    print()
    print("Each GPU will process all methods sequentially.")
    print("Progress will be shown per GPU worker.")
    print("="*80)
    print()

    ctx = mp.get_context('spawn')
    queue = ctx.Queue()

    processes = []
    for i, gpu_id in enumerate(gpus_to_use):
        p = ctx.Process(
            target=worker_evaluate_all_methods,
            args=(gpu_id, text_splits[i], model_name, max_length, chunk_size,
                  methods_config, queue)
        )
        p.start()
        processes.append(p)

    worker_results = []
    completed_workers = 0
    for _ in range(num_workers):
        result = queue.get()
        if not result["success"]:
            for p in processes:
                p.terminate()
            error_msg = f"Worker failed on GPU {result['gpu_id']}: {result['error']}"
            if "traceback" in result:
                error_msg += f"\n{result['traceback']}"
            raise RuntimeError(error_msg)
        worker_results.append(result)
        completed_workers += 1
        print()
        print("="*80)
        print(f"[GPU {result['gpu_id']}] ✓ ALL METHODS COMPLETE ({result['num_texts']} samples)")
        print(f"Progress: {completed_workers}/{num_workers} GPUs finished")
        print("="*80)
        print()

    for p in processes:
        p.join()

    # Aggregate results from all workers for each method
    all_method_results = {}
    method_names = [m["method_name"] for m in methods_config]

    for method_name in method_names:
        # Collect results for this method from all workers
        method_worker_results = []
        for wr in worker_results:
            method_worker_results.append({
                "success": True,
                "gpu_id": wr["gpu_id"],
                "results": wr["results"][method_name],
                "num_texts": wr["num_texts"],
            })
        all_method_results[method_name] = aggregate_results(method_worker_results)

    return all_method_results


# ==============================================================================
# Phase 1: Core Evaluation Wrapper
# ==============================================================================

def evaluate_with_press(
    model: Optional[AutoModelForCausalLM],
    tokenizer: Optional[AutoTokenizer],
    texts: List[str],
    press: Optional[Any],
    method_name: str,
    press_type: Optional[str] = None,
    press_params: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    max_length: int = 8192,
    chunk_size: int = 1,
    device: str = "cuda:0",
    num_gpus: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate perplexity with a given compression method.

    This wraps calculate_perplexity_chunked and standardizes the output format
    for easy comparison across methods. Supports both single-GPU and multi-GPU evaluation.

    Parameters
    ----------
    model : AutoModelForCausalLM, optional
        The language model to evaluate (for single-GPU mode)
    tokenizer : AutoTokenizer, optional
        Tokenizer for the model (for single-GPU mode)
    texts : list[str]
        List of text strings to evaluate
    press : Press or None, optional
        Compression method to use (None for baseline) - for single-GPU mode
    method_name : str
        Name of the method for reporting
    press_type : str, optional
        Type of press for multi-GPU mode ("random", "streaming_llm", etc.)
    press_params : dict, optional
        Parameters for press creation in multi-GPU mode
    model_name : str, optional
        Model name for multi-GPU mode
    max_length : int
        Maximum sequence length
    chunk_size : int
        Chunk size for evaluation
    device : str
        Device to use (for single-GPU mode)
    num_gpus : int, optional
        Number of GPUs to use. If > 1 or -1, uses multi-GPU mode.

    Returns
    -------
    dict
        Standardized results dictionary with method name and metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*80}")

    # Determine whether to use multi-GPU
    use_multigpu = num_gpus is not None and num_gpus != 1
    if num_gpus == -1:
        num_gpus_actual = None  # Will use all available
    else:
        num_gpus_actual = num_gpus

    if use_multigpu:
        # Multi-GPU evaluation
        if model_name is None:
            raise ValueError("model_name must be provided for multi-GPU evaluation")

        results = evaluate_multigpu(
            texts=texts,
            model_name=model_name,
            max_length=max_length,
            chunk_size=chunk_size,
            press_type=press_type,
            press_params=press_params,
            num_gpus=num_gpus_actual,
        )
    else:
        # Single-GPU evaluation
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer must be provided for single-GPU evaluation")

        results = calculate_perplexity_chunked(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=max_length,
            chunk_size=chunk_size,
            press=press,
            device=device,
        )

    # Add method name to results
    results["method_name"] = method_name

    # Print summary
    print(f"\n{method_name} Results:")
    print(f"  Perplexity: {results['perplexity']:.4f}")
    print(f"  Avg NLL: {results['avg_nll']:.4f}")
    print(f"  Total tokens: {results['total_tokens']:,}")

    if "compression_stats" in results:
        stats = results["compression_stats"]
        density_with_window = (1 - stats['sparsity_with_window']) * 100
        density_without_window = (1 - stats['sparsity_without_window']) * 100
        print(f"\nCompression Statistics:")
        print(f"  Overall Density (w/ window): {density_with_window:.2f}%")
        print(f"  Compressible Region Density (w/o window): {density_without_window:.2f}%")
        print(f"  Keys Kept: {stats['total_keys_kept']:,} / {stats['total_keys_original']:,}")
        print(f"  Sliding Window Size: {stats['sliding_window_size']}")

    return results


# ==============================================================================
# Main Comparison Function (Skeleton - to be implemented in phases)
# ==============================================================================

def compare_compression_methods(
    data_path: str = None,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    compression_ratio: float = 0.5,
    thresholds: Optional[Dict[str, float]] = None,
    densities: Optional[Dict[str, float]] = None,
    sliding_window_size: int = DMS_SLIDING_WINDOW_SIZE,
    kvzap_scorer_model: Optional[str] = None,
    kvzap_model_type: str = "mlp",
    max_samples: Optional[int] = None,
    max_length: int = 8192,
    chunk_size: int = 1,
    device: str = "cuda:0",
    output_dir: str = "./comparison_results",
    methods: Optional[List[str]] = None,
    include_observed_attention: bool = False,
    h2o_local_window_size: int = 512,
    num_gpus: Optional[int] = None,
    visualize_first_sample: bool = True,
    # New loader selection args
    data_loader: str = "jsonl_text",
    text_field: str = None,
    # AMAIA loader args
    amaia_sources_config: str = None,
    amaia_seq_len: int = None,
    amaia_seed: int = 42,
    amaia_shuffle_buffer_size: int = 1,
    amaia_tokenizer_name: str = "tiktoken",
    amaia_tokenizer_path: str = None,
    amaia_path: str = "/storage/home/manufay/amaia",
    # Token file loader args
    token_data_path: str = None,
    token_format: str = "auto",
    token_field_names: str = None,
):
    """
    Compare different KV cache compression methods.

    Methods use different compression strategies:
    - StreamingLLM: Fixed structure (n_sink=4 + sliding_window=128)
    - Random: Threshold-based with random scores
    - ExpectedAttention: Threshold-based with statistical prediction
    - H2O, ObservedAttention, KVzap: Threshold-based compression

    Parameters
    ----------
    data_path : str
        Path to JSONL dataset
    model_name : str
        HuggingFace model identifier
    compression_ratio : float
        Deprecated - not used anymore (kept for backwards compatibility)
    thresholds : dict[str, float], optional
        Method-specific threshold values. Keys are method names:
        "kvzap", "expected_attention", "random", "observed_attention", "h2o"
        If not provided, uses DEFAULT_THRESHOLDS. You can override specific methods:
        thresholds={"kvzap": -7.0, "expected_attention": 0.05}
        See DEFAULT_THRESHOLDS dict for default values and score ranges.
    sliding_window_size : int
        Number of recent tokens protected from eviction (for threshold-based methods)
    kvzap_scorer_model : str, optional
        Explicit KVzap scorer model name
    kvzap_model_type : str
        Type of KVzap model ("mlp" or "linear")
    max_samples : int, optional
        Limit number of samples for testing
    max_length : int
        Maximum sequence length
    chunk_size : int
        Chunk size for evaluation.
        CRITICAL: Must be > 1 for compression to actually work!
        - chunk_size=1: NO compression applied (kvpress skips when q_len=1)
        - chunk_size>1: Compression is applied (use 128 or higher for faster eval)
    device : str
        Device to use (for single-GPU mode)
    output_dir : str
        Directory to save results
    methods : list[str], optional
        List of methods to evaluate. If None, runs all except ObservedAttention/H2O.
        Options: ["random", "streaming_llm", "random_validation", "expected_attention",
                  "observed_attention", "h2o", "kvzap"]
        Note: "random_validation" uses threshold=1.0 and should match StreamingLLM exactly
    include_observed_attention : bool
        Whether to include ObservedAttention and H2O (requires eager mode, slower)
    h2o_local_window_size : int
        Local window size for H2O press (number of recent tokens to preserve)
    num_gpus : int, optional
        Number of GPUs to use for parallel evaluation.
        - None or 1: Single-GPU mode (default)
        - -1: Use all available GPUs
        - N > 1: Use exactly N GPUs
    visualize_first_sample : bool, default=True
        Whether to generate heatmap visualizations of compression decisions
        for the first sample (first 1024 tokens). Saves PNG files to output_dir.

    Returns
    -------
    dict
        Comparison results
    """
    # Parse thresholds if passed as JSON string (from command line)
    if thresholds is not None and thresholds != "":
        if isinstance(thresholds, str):
            try:
                thresholds = json.loads(thresholds)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse thresholds JSON string: {e}\nProvided: {repr(thresholds)}")

        # Validate thresholds is a dict
        if not isinstance(thresholds, dict):
            raise ValueError(f"thresholds must be a dict, got {type(thresholds)}: {thresholds}")
    elif thresholds == "":
        # Fire passes empty string when not provided
        thresholds = None

    # Determine which methods to run
    if methods is None:
        methods_to_run = ["h2o", "observed_attention", "kvzap", "random", "streaming_llm", "expected_attention"]
        if include_observed_attention:
            methods_to_run.extend(["observed_attention"])
    else:
        # Handle Fire CLI passing a string instead of a list
        if isinstance(methods, str):
            # Could be comma-separated or a single method
            if "," in methods:
                methods_to_run = [m.strip() for m in methods.split(",") if m.strip()]
            else:
                methods_to_run = [methods.strip()]
        elif isinstance(methods, (list, tuple)):
            methods_to_run = list(methods)
        else:
            raise ValueError(f"methods must be a list or string, got {type(methods)}: {methods}")

    # Merge user-provided thresholds with defaults
    active_thresholds = DEFAULT_THRESHOLDS.copy()
    if thresholds is not None:
        active_thresholds.update(thresholds)

    # Merge user-provided densities with defaults
    active_densities = DEFAULT_DENSITIES.copy()
    if densities is not None:
        active_densities.update(densities)

    # Determine if using multi-GPU (needed for configuration display)
    use_multigpu = num_gpus is not None and num_gpus != 1

    # CRITICAL WARNING: chunk_size=1 does NOT apply compression!
    # The kvpress forward_hook skips compression when q_len=1 (autoregressive mode).
    if chunk_size == 1 and methods_to_run:
        import warnings
        warnings.warn(
            "\\n" + "="*80 + "\\n"
            "CRITICAL: chunk_size=1 will NOT apply any KV cache compression!\\n"
            "The kvpress forward_hook skips compression when processing single tokens.\\n"
            "All compression methods will produce IDENTICAL results to baseline.\\n"
            "Use chunk_size > 1 (e.g., 128) for compression to take effect.\\n"
            + "="*80,
            UserWarning
        )

    print("="*80)
    print("KV CACHE COMPRESSION METHODS COMPARISON")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {data_path}")
    print(f"  Chunk size: {chunk_size}")
    if not use_multigpu:
        print(f"  No-chunking baseline: Will be evaluated (single-GPU mode only)")
    print(f"  Thresholds/Densities:")
    for method in methods_to_run:
        thresh = active_thresholds.get(method)
        density = active_densities.get(method)
        if density is not None:
            print(f"    - {method}: target_density={density}")
        elif thresh is not None:
            print(f"    - {method}: threshold={thresh}")
        else:
            print(f"    - {method}: (using defaults)")
    print(f"  Sliding window size: {sliding_window_size}")
    print(f"  Methods: {', '.join(methods_to_run)}")
    if num_gpus is not None:
        if num_gpus == -1:
            print(f"  GPUs: All available")
        else:
            print(f"  GPUs: {num_gpus}")
    else:
        print(f"  GPUs: 1 (single-GPU mode)")
    print()

    # Check if any method requires eager attention mode
    needs_eager = "observed_attention" in methods_to_run or "h2o" in methods_to_run

    # Phase 1: Load model and dataset
    print("Loading dataset...")

    # Only load model in single-GPU mode (multi-GPU loads on each worker)
    if use_multigpu:
        print("Multi-GPU mode: Model will be loaded once per GPU worker")
        tokenizer = None
        model = None
    else:
        print("Loading model on single GPU...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use eager attention if any method requires it
        if needs_eager:
            print("  (using eager attention for observed_attention/h2o methods)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        model.eval()

    # Validate loader arguments
    if data_loader == "jsonl_text" and data_path is None:
        raise ValueError("--data_path is required for jsonl_text loader")
    if data_loader == "amaia" and amaia_sources_config is None:
        raise ValueError("--amaia_sources_config is required for amaia loader")
    if data_loader == "tokens_file" and token_data_path is None and data_path is None:
        raise ValueError("--token_data_path (or --data_path) is required for tokens_file loader")

    # Parse token field names if provided
    token_field = "tokens"
    mask_field = "mask"
    if token_field_names:
        parts = token_field_names.split(",")
        token_field = parts[0].strip()
        if len(parts) > 1:
            mask_field = parts[1].strip()

    # Load dataset using the appropriate loader
    print(f"\nLoading dataset using {data_loader} loader...")

    if data_loader == "jsonl_text":
        # For jsonl_text, we need the tokenizer to create the loader
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        loader = create_loader(
            loader_type="jsonl_text",
            file_path=data_path,
            tokenizer=tokenizer,
            max_samples=max_samples,
            max_length=max_length,
            text_field=text_field,
        )
    elif data_loader == "amaia":
        seq_len = amaia_seq_len if amaia_seq_len is not None else max_length
        loader = create_loader(
            loader_type="amaia",
            amaia_sources_config=amaia_sources_config,
            amaia_seq_len=seq_len,
            max_samples=max_samples,
            amaia_seed=amaia_seed,
            amaia_shuffle_buffer_size=amaia_shuffle_buffer_size,
            amaia_tokenizer_name=amaia_tokenizer_name,
            amaia_tokenizer_path=amaia_tokenizer_path,
            amaia_path=amaia_path,
        )
    elif data_loader == "tokens_file":
        loader = create_loader(
            loader_type="tokens_file",
            token_data_path=token_data_path or data_path,
            max_samples=max_samples,
            token_format=token_format,
            token_field=token_field,
            mask_field=mask_field,
        )
    else:
        raise ValueError(f"Unknown data_loader: {data_loader}. Choose from: jsonl_text, amaia, tokens_file")

    samples = loader.load()
    print(f"Loaded {len(samples)} samples via {loader.get_description()}")

    # For backward compatibility, also extract texts for multi-GPU and visualization
    texts = None
    if use_multigpu or visualize_first_sample:
        if data_loader == "jsonl_text" and data_path:
            data = load_jsonl_dataset(data_path, max_samples=max_samples)
            if isinstance(data[0], dict):
                if "text" in data[0]:
                    texts = [item["text"] for item in data]
                elif "content" in data[0]:
                    texts = [item["content"] for item in data]
                else:
                    first_key = list(data[0].keys())[0]
                    texts = [item[first_key] for item in data]
            else:
                texts = data
        else:
            # For other loaders, serialize samples
            texts = [s.to_serializable() for s in samples]

    # Create output directory for results and visualizations
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Storage for all results
    all_results = {}

    # Build methods configuration for all methods to evaluate
    methods_config = []

    # Baseline (no compression)
    methods_config.append({
        "method_name": "Baseline",
        "press_type": None,
        "press_params": None,
        "requires_eager": False,
    })

    # Random Press
    if "random" in methods_to_run:
        methods_config.append({
            "method_name": "Random",
            "press_type": "random",
            "press_params": {
                "threshold": active_thresholds["random"],
                "sliding_window_size": sliding_window_size,
                "seed": None,  # None for true randomness (not strided pattern)
                "n_sink": 4,
            },
            "requires_eager": False,
        })

    # StreamingLLM Press
    if "streaming_llm" in methods_to_run:
        methods_config.append({
            "method_name": "StreamingLLM",
            "press_type": "streaming_llm",
            "press_params": {"n_sink": 4, "sliding_window_size": 128},
            "requires_eager": False,
        })

    # Random Validation (should match StreamingLLM exactly)
    if "random_validation" in methods_to_run:
        methods_config.append({
            "method_name": "Random(threshold=1.0)",
            "press_type": "random",
            "press_params": {
                "threshold": active_thresholds["random_validation"],
                "sliding_window_size": sliding_window_size,
                "seed": 42,
                "n_sink": 4,
            },
            "requires_eager": False,
        })

    # ExpectedAttention Press
    if "expected_attention" in methods_to_run:
        methods_config.append({
            "method_name": "ExpectedAttention",
            "press_type": "expected_attention",
            "press_params": {
                "threshold": active_thresholds["expected_attention"],
                "sliding_window_size": sliding_window_size,
                "n_future_positions": 512,
                "n_sink": 4,
                "use_covariance": True,
                "use_vnorm": True,
            },
            "requires_eager": False,
        })

    # KVzap Press
    if "kvzap" in methods_to_run:
        methods_config.append({
            "method_name": "KVzap",
            "press_type": "kvzap",
            "press_params": {
                "threshold": active_thresholds["kvzap"],
                "kvzap_model_type": kvzap_model_type,
                "kvzap_scorer_model": kvzap_scorer_model,
                "sliding_window_size": DMS_SLIDING_WINDOW_SIZE,
                "n_sink": 4,
            },
            "requires_eager": False,
        })

    # KVzap Press (No Sink Tokens)
    if "kvzap_no_sink" in methods_to_run:
        methods_config.append({
            "method_name": "KVzap(no_sink)",
            "press_type": "kvzap_no_sink",
            "press_params": {
                "threshold": active_thresholds["kvzap_no_sink"],
                "kvzap_model_type": kvzap_model_type,
                "kvzap_scorer_model": kvzap_scorer_model,
                "sliding_window_size": DMS_SLIDING_WINDOW_SIZE,
            },
            "requires_eager": False,
        })

    # ObservedAttention Press (requires eager)
    if "observed_attention" in methods_to_run:
        # Use target_density from DEFAULT_DENSITIES, or fall back to threshold
        obs_density = active_densities.get("observed_attention")
        obs_threshold = active_thresholds.get("observed_attention")
        methods_config.append({
            "method_name": "ObservedAttention",
            "press_type": "observed_attention",
            "press_params": {
                "threshold": obs_threshold,
                "target_density": obs_density,
                "sliding_window_size": sliding_window_size
            },
            "requires_eager": True,
        })

    # H2O Press (requires eager)
    if "h2o" in methods_to_run:
        # Use target_density from DEFAULT_DENSITIES, or fall back to threshold
        h2o_density = active_densities.get("h2o")
        h2o_threshold = active_thresholds.get("h2o")
        methods_config.append({
            "method_name": "H2O",
            "press_type": "h2o",
            "press_params": {
                "threshold": h2o_threshold,
                "target_density": h2o_density,
                "sliding_window_size": sliding_window_size,
                "local_window_size": h2o_local_window_size,
                "n_sink": 4,  # H2O uses 4 sink tokens by default
            },
            "requires_eager": True,
        })

    # Evaluate all methods
    print("\n" + "="*80)
    print("EVALUATING ALL METHODS")
    print("="*80)

    if use_multigpu:
        # Multi-GPU mode: evaluate all methods in parallel, model loaded once per GPU
        multigpu_results = evaluate_all_methods_multigpu(
            texts=texts,
            model_name=model_name,
            max_length=max_length,
            chunk_size=chunk_size,
            methods_config=methods_config,
            num_gpus=num_gpus if num_gpus != -1 else None,
        )

        # Map results to all_results
        for method_cfg in methods_config:
            method_name = method_cfg["method_name"]
            result = multigpu_results[method_name]
            result["method_name"] = method_name

            if method_name == "Baseline":
                all_results["baseline"] = result
            else:
                # Convert method name to key
                method_key = method_name.lower().replace(" ", "_")
                if method_key == "expectedattention":
                    method_key = "expected_attention"
                elif method_key == "streamingllm":
                    method_key = "streaming_llm"
                elif method_key == "observedattention":
                    method_key = "observed_attention"
                all_results[method_key] = result

            # Print summary for this method
            print(f"\n{method_name} Results:")
            print(f"  Perplexity: {result['perplexity']:.4f}")
            print(f"  Avg NLL: {result['avg_nll']:.4f}")
            print(f"  Total tokens: {result['total_tokens']:,}")
            if "compression_stats" in result:
                stats = result["compression_stats"]
                density_with_window = (1 - stats['sparsity_with_window']) * 100
                density_without_window = (1 - stats['sparsity_without_window']) * 100
                print(f"  Compression Statistics:")
                print(f"    Overall Density (w/ window): {density_with_window:.2f}%")
                print(f"    Compressible Region Density (w/o window): {density_without_window:.2f}%")

    else:
        # Single-GPU mode: evaluate all methods sequentially with the same model
        print(f"\nSingle-GPU mode: Evaluating {len(methods_config)} methods with model loaded once")

        # First, evaluate baseline with NO chunking (full sequence in one pass)
        print(f"\n{'='*80}")
        print(f"Evaluating: Baseline (No Chunking)")
        print(f"{'='*80}")
        print("Note: Processing entire sequence in one forward pass (no compression artifacts)")

        result_no_chunk = calculate_perplexity_chunked(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            max_length=max_length,
            chunk_size=max_length,  # Process entire sequence at once
            press=None,
            device=device,
        )
        result_no_chunk["method_name"] = "Baseline (No Chunking)"
        all_results["baseline_no_chunking"] = result_no_chunk

        print(f"\nBaseline (No Chunking) Results:")
        print(f"  Perplexity: {result_no_chunk['perplexity']:.4f}")
        print(f"  Avg NLL: {result_no_chunk['avg_nll']:.4f}")
        print(f"  Total tokens: {result_no_chunk['total_tokens']:,}")

        # Now evaluate all methods (including chunked baseline)
        for method_cfg in methods_config:
            method_name = method_cfg["method_name"]
            press_type = method_cfg.get("press_type")
            press_params = method_cfg.get("press_params")

            print(f"\n{'='*80}")
            print(f"Evaluating: {method_name}")
            print(f"{'='*80}")

            # Create the press based on type
            press = None
            if press_type is not None:
                if press_type == "random":
                    press = create_random_press(**press_params)
                elif press_type == "streaming_llm":
                    press = create_streaming_llm_press(**press_params)
                elif press_type == "expected_attention":
                    press = create_expected_attention_press(**press_params)
                elif press_type == "kvzap":
                    press = create_kvzap_press(**press_params)
                elif press_type == "kvzap_no_sink":
                    press = create_kvzap_no_sink_press(**press_params)
                elif press_type == "observed_attention":
                    press = create_observed_attention_press(**press_params)
                elif press_type == "h2o":
                    press = create_h2o_press(**press_params)

            result = calculate_perplexity_chunked(
                model=model,
                tokenizer=tokenizer,
                samples=samples,
                max_length=max_length,
                chunk_size=chunk_size,
                press=press,
                device=device,
            )
            result["method_name"] = method_name

            # Map to all_results with correct key
            if method_name == "Baseline":
                all_results["baseline"] = result
            else:
                method_key = method_name.lower().replace(" ", "_")
                if method_key == "expectedattention":
                    method_key = "expected_attention"
                elif method_key == "streamingllm":
                    method_key = "streaming_llm"
                elif method_key == "observedattention":
                    method_key = "observed_attention"
                all_results[method_key] = result

            # Print summary
            print(f"\n{method_name} Results:")
            print(f"  Perplexity: {result['perplexity']:.4f}")
            print(f"  Avg NLL: {result['avg_nll']:.4f}")
            print(f"  Total tokens: {result['total_tokens']:,}")
            if "compression_stats" in result:
                stats = result["compression_stats"]
                density_with_window = (1 - stats['sparsity_with_window']) * 100
                density_without_window = (1 - stats['sparsity_without_window']) * 100
                print(f"  Compression Statistics:")
                print(f"    Overall Density (w/ window): {density_with_window:.2f}%")
                print(f"    Compressible Region Density (w/o window): {density_without_window:.2f}%")

            # Generate visualization for first sample
            if visualize_first_sample and not use_multigpu and press_type is not None:
                print(f"\n  Generating compression visualization for {method_name}...")

                # Create a FRESH press instance for visualization (don't reuse the one from evaluation)
                viz_press = None
                if press_type == "random":
                    viz_press = create_random_press(**press_params)
                elif press_type == "streaming_llm":
                    viz_press = create_streaming_llm_press(**press_params)
                elif press_type == "expected_attention":
                    viz_press = create_expected_attention_press(**press_params)
                elif press_type == "kvzap":
                    viz_press = create_kvzap_press(**press_params)
                elif press_type == "kvzap_no_sink":
                    viz_press = create_kvzap_no_sink_press(**press_params)
                elif press_type == "observed_attention":
                    viz_press = create_observed_attention_press(**press_params)
                elif press_type == "h2o":
                    viz_press = create_h2o_press(**press_params)

                # Capture scores from first sample
                first_text = texts[0] if texts else None
                if first_text and viz_press is not None:
                    scores_dict = capture_scores_first_sample(
                        model=model,
                        tokenizer=tokenizer,
                        text=first_text,
                        press=viz_press,
                        max_length=max_length,
                        chunk_size=chunk_size,
                        device=device,
                    )

                    # Determine threshold, target_density, and n_sink for this method
                    threshold = press_params.get("threshold") if press_params else None
                    target_density = press_params.get("target_density") if press_params else None
                    n_sink = press_params.get("n_sink", 0) if press_params else 0

                    # Generate visualization
                    visualize_compression_heatmap(
                        scores_dict=scores_dict,
                        threshold=threshold,
                        method_name=method_name,
                        output_dir=output_path,
                        max_tokens=1024,
                        sliding_window_size=sliding_window_size,
                        n_sink=n_sink,
                        target_density=target_density,
                    )

    baseline_ppl = all_results["baseline"]["perplexity"]
    baseline_results = all_results["baseline"]

    # Phase 5: Aggregate and save results
    print("\n" + "="*80)
    print("PHASE 5: AGGREGATING RESULTS")
    print("="*80)

    # Compute comparison statistics
    comparison_results = []
    for method_key, method_results in all_results.items():
        if method_key in ["baseline", "baseline_no_chunking"]:
            continue

        method_name = method_results["method_name"]
        ppl = method_results["perplexity"]
        nll = method_results["avg_nll"]

        ppl_delta = ppl - baseline_ppl
        ppl_delta_pct = (ppl_delta / baseline_ppl) * 100
        nll_delta = nll - baseline_results["avg_nll"]

        # Get density metrics from compression stats
        if "compression_stats" in method_results:
            stats = method_results["compression_stats"]

            # Overall density (includes all tokens: sinks + compressible + window)
            total_keys = stats["total_keys_original"]
            keys_kept = stats["total_keys_kept"]
            density_overall = (keys_kept / total_keys * 100) if total_keys > 0 else 100.0

            # Compressible region density (excludes sinks + window)
            # This is correctly calculated in evaluate_ppl_chunked.py
            density_compressible = (1 - stats["sparsity_without_window"]) * 100

        else:
            density_overall = None
            density_compressible = None

        comparison_results.append({
            "method": method_name,
            "method_key": method_key,
            "perplexity": ppl,
            "avg_nll": nll,
            "ppl_delta": ppl_delta,
            "ppl_delta_pct": ppl_delta_pct,
            "nll_delta": nll_delta,
            "nll_delta_pct": (nll_delta / baseline_results["avg_nll"]) * 100,
            "density_overall": density_overall,
            "density_compressible": density_compressible,
        })

    # Sort by NLL (best to worst)
    comparison_results.sort(key=lambda x: x["avg_nll"])

    # Print comparison table
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)

    # Show both baseline NLLs
    if "baseline_no_chunking" in all_results:
        baseline_no_chunk = all_results["baseline_no_chunking"]
        print(f"\nBaseline (No Chunking) NLL: {baseline_no_chunk['avg_nll']:.6f} | PPL: {baseline_no_chunk['perplexity']:.4f}")
        print(f"Baseline (Chunked)     NLL: {baseline_results['avg_nll']:.6f} | PPL: {baseline_ppl:.4f}")
        chunking_artifact = baseline_results['avg_nll'] - baseline_no_chunk['avg_nll']
        chunking_artifact_pct = (chunking_artifact / baseline_no_chunk['avg_nll']) * 100
        print(f"Chunking Artifact:     ΔNLL: {chunking_artifact:+.6f} ({chunking_artifact_pct:+.2f}%)")
    else:
        print(f"\nBaseline (Chunked) NLL: {baseline_results['avg_nll']:.6f}")
        print(f"Baseline (Chunked) PPL: {baseline_ppl:.4f}")
    print()

    # Table header
    print(f"{'Method':<25} {'NLL':>12} {'ΔNLL':>12} {'ΔNLL %':>10} {'Dens(all)':>12} {'Dens(comp)':>12}")
    print("-" * 100)
    print("Note: ΔNLL is relative to Baseline (Chunked)")
    print("      Dens(all) = Overall density (all tokens)")
    print("      Dens(comp) = Density in compressible region (excluding window, may include sinks)")
    print("-" * 100)

    # Baseline rows
    if "baseline_no_chunking" in all_results:
        baseline_no_chunk = all_results["baseline_no_chunking"]
        delta_no_chunk = baseline_no_chunk['avg_nll'] - baseline_results['avg_nll']
        delta_pct_no_chunk = (delta_no_chunk / baseline_results['avg_nll']) * 100
        print(f"{'Baseline (No Chunking)':<25} {baseline_no_chunk['avg_nll']:>12.6f} {delta_no_chunk:>+12.6f} {delta_pct_no_chunk:>+9.2f}% {'100.00%':>12} {'100.00%':>12}")

    print(f"{'Baseline (Chunked)':<25} {baseline_results['avg_nll']:>12.6f} {0.0:>12.6f} {0.0:>9.2f}% {'100.00%':>12} {'100.00%':>12}")

    # Other methods
    for result in comparison_results:
        method = result["method"]
        nll = result["avg_nll"]
        delta = result["nll_delta"]
        delta_pct = result["nll_delta_pct"]
        dens_all = result["density_overall"]
        dens_comp = result["density_compressible"]

        dens_all_str = f"{dens_all:.2f}%" if dens_all is not None else "N/A"
        dens_comp_str = f"{dens_comp:.2f}%" if dens_comp is not None else "N/A"

        print(f"{method:<25} {nll:>12.6f} {delta:>+12.6f} {delta_pct:>+9.2f}% {dens_all_str:>12} {dens_comp_str:>12}")

    print()

    # Save results to JSON (output_path already created earlier)
    # Prepare output data
    output_data = {
        "configuration": {
            "model_name": model_name,
            "data_path": data_path,
            "max_samples": max_samples,
            "max_length": max_length,
            "chunk_size": chunk_size,
            "thresholds": active_thresholds,
            "sliding_window_size": sliding_window_size,
            "kvzap_model_type": kvzap_model_type,
            "kvzap_scorer_model": kvzap_scorer_model,
            "num_texts": len(samples),
            "data_loader": data_loader,
            "num_gpus": num_gpus,
            "use_multigpu": use_multigpu,
        },
        "baseline": baseline_results,
        "methods": all_results,
        "comparison": comparison_results,
    }

    # Save detailed results
    detailed_file = output_path / "detailed_results.json"
    with open(detailed_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=float)
    print(f"Detailed results saved to: {detailed_file}")

    # Save comparison summary
    summary_file = output_path / "comparison_summary.json"
    summary_data = {
        "baseline_perplexity": baseline_ppl,
        "baseline_nll": baseline_results["avg_nll"],
        "comparisons": comparison_results,
        "configuration": output_data["configuration"],
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=float)
    print(f"Comparison summary saved to: {summary_file}")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

    return output_data


if __name__ == "__main__":
    import fire
    fire.Fire(compare_compression_methods)
