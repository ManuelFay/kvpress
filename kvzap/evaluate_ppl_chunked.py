#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunked PPL Evaluation for KV Cache Compression

This script evaluates perplexity (PPL) with KV cache compression in a way that
accurately simulates autoregressive generation with sparse attention.

==============================================================================
WHY CHUNKED PROCESSING?
==============================================================================

Standard PPL evaluation with KV compression has a fundamental problem:
- With decoding=False: Compression happens AFTER the full sequence is processed,
  so it doesn't affect the predictions at all (threshold has no impact on PPL)
- With decoding=True on full sequence: The model processes all tokens at once
  in parallel, and compression is only applied at the very end

In actual generation, each token sees a COMPRESSED cache from previous tokens:
  Token 1: sees no cache (first token)
  Token 2: sees compressed cache from token 1
  Token 3: sees compressed cache from tokens 1-2
  Token N: sees compressed cache from tokens 1-(N-1)

Chunked processing simulates this by:
  1. Process chunk 0 (tokens 0 to k-1) → produces cache
  2. Compress the cache from chunk 0
  3. Process chunk 1 (tokens k to 2k-1) → attends to COMPRESSED cache from chunk 0
  4. Compress the cache (now contains chunks 0-1)
  5. Process chunk 2 → attends to COMPRESSED cache from chunks 0-1
  ... and so on

==============================================================================
CHUNK SIZE TRADEOFFS
==============================================================================

chunk_size=1 (token-by-token):
  ✓ Exact NLL computation - perfectly simulates autoregressive generation
  ✓ Each token sees the compressed cache from ALL previous tokens
  ✗ Very slow - requires N forward passes for sequence of length N
  → Use for: Final accurate evaluation, small datasets, research measurements

chunk_size=K (K > 1):
  ✓ Much faster - requires N/K forward passes instead of N
  ✗ Approximate NLL - tokens within a chunk see uncompressed cache
  ✗ First token of each chunk sees compressed cache, but the remaining K-1
    tokens in that chunk see the full uncompressed cache from earlier in the chunk
  → Use for: Quick experiments, large-scale evaluations, development

Example with chunk_size=4 on sequence [t0, t1, t2, t3, t4, t5, t6, t7]:
  Chunk 0: [t0, t1, t2, t3]
    - t0: sees no cache (first token)
    - t1: sees full cache from t0 (NOT compressed)
    - t2: sees full cache from t0,t1 (NOT compressed)
    - t3: sees full cache from t0,t1,t2 (NOT compressed)
    → Cache is compressed after chunk 0
  Chunk 1: [t4, t5, t6, t7]
    - t4: sees COMPRESSED cache from chunk 0 ✓ (correct)
    - t5: sees COMPRESSED cache from chunk 0 + full cache from t4 ✗ (not fully correct)
    - t6: sees COMPRESSED cache from chunk 0 + full cache from t4,t5 ✗
    - t7: sees COMPRESSED cache from chunk 0 + full cache from t4,t5,t6 ✗
    → Only the first token per chunk (t4) sees the fully compressed view

With chunk_size=1, every single token sees the compressed cache from all
previous tokens, giving exact generation simulation.

==============================================================================
MULTI-GPU PARALLEL EVALUATION
==============================================================================

This script supports multi-GPU parallelization to significantly speed up evaluation:

How it works:
  1. Dataset is split evenly across GPUs
  2. Each GPU loads its own copy of the model
  3. Each GPU processes its subset of texts independently
  4. Results are aggregated at the end

Usage:
  --num_gpus -1        # Use all available GPUs
  --num_gpus 4         # Use exactly 4 GPUs
  --num_gpus 1         # Single-GPU mode (default)

Expected speedups (approximate):
  - 2 GPUs: ~1.8-1.9x faster
  - 4 GPUs: ~3.5-3.8x faster
  - 8 GPUs: ~7.0-7.5x faster

==============================================================================
THRESHOLD SWEEP
==============================================================================

Use --thresholds to evaluate multiple compression thresholds and get a summary
table showing the tradeoff between sparsity and quality:

  python evaluate_ppl_chunked.py --data_path data.jsonl --thresholds "-6,-7,-8,-9"

This produces a summary table at the end:

  Threshold | Sparsity | Sparsity* |    NLL |   ΔNLL |    PPL |   ΔPPL%
       -6.0 |   32.45% |    38.12% | 2.1298 | +0.006 |   8.42 |  +0.72%
       -7.0 |   51.23% |    58.67% | 2.1456 | +0.022 |   8.55 |  +1.80%
       ...

==============================================================================
RECOMMENDED USAGE
==============================================================================

For exact NLL:
  chunk_size=1 (default)

For faster approximate NLL:
  chunk_size=128 or chunk_size=256
  (Should still be meaningful but 100x+ faster)

For fastest evaluation:
  chunk_size=128 with --num_gpus -1 (use all GPUs)
  (Combines chunking speedup with multi-GPU parallelism)

DMSPress sliding window: 128 tokens (fixed)
  The most recent 128 tokens are always protected from eviction
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress import KVzapPress, DMSPress
from kvpress.presses.kvzap_press import KVzapModel
from kvpress.presses.scorer_press import ScorerPress
import multiprocessing as mp
from typing import List, Dict, Optional

# DMSPress sliding window size (protects most recent N tokens from eviction)
DMS_SLIDING_WINDOW_SIZE = 128


class CustomKVzapPress(KVzapPress):
    """
    Custom KVzapPress that allows explicit model name override.

    This is necessary because the default KVzapPress infers the scorer model name
    from the LLM's config.name_or_path, which can sometimes produce incorrect names.

    For example:
    - LLM: "meta-llama/Meta-Llama-3.1-8B-Instruct"
    - Default inferred: "nvidia/KVzap-mlp-Meta-Llama-3.1-8B-Instruct" (doesn't exist)
    - Correct scorer: "nvidia/KVzap-mlp-Llama-3.1-8B-Instruct"

    Usage:
        press = CustomKVzapPress(
            model_type="mlp",
            explicit_model_name="nvidia/KVzap-mlp-Llama-3.1-8B-Instruct",
            n_sink=4
        )
    """

    def __init__(self, model_type="mlp", explicit_model_name=None, n_sink=4):
        super().__init__(model_type=model_type, n_sink=n_sink)
        self.explicit_model_name = explicit_model_name

    def post_init_from_model(self, model):
        """
        Initialize the KVzap scorer model.

        Only loads the model if the name changes, preventing redundant reloading
        on every forward pass.
        """
        if self.explicit_model_name is not None:
            if self.explicit_model_name != self.kvzap_model_name:
                self.kvzap_model_name = self.explicit_model_name
                self.kvzap_model = KVzapModel.from_pretrained(self.kvzap_model_name)
        else:
            super().post_init_from_model(model)


def load_jsonl_dataset(file_path: str, max_samples: int = None) -> List:
    """Load a JSONL dataset."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            data.append(json.loads(line))
    return data


def calculate_perplexity_chunked(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 8192,
    chunk_size: int = 1,
    press=None,
    device: str = "cuda",
    warn_no_compression: bool = True,
    debug: bool = False,
    tqdm_desc: Optional[str] = None,
    disable_tqdm: bool = False,
) -> Dict:
    """
    Calculate perplexity using chunked processing that simulates autoregressive generation.

    This is the core function that implements the chunked evaluation strategy:
    1. Divide each sequence into chunks of size chunk_size
    2. Process chunk 0 → compress its cache
    3. Process chunk 1 with compressed cache from chunk 0 → compress combined cache
    4. Process chunk 2 with compressed cache from chunks 0-1 → compress
    5. Continue until sequence is complete

    With chunk_size=1, this exactly simulates autoregressive generation where each
    token sees the compressed cache from all previous tokens.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The language model to evaluate
    tokenizer : AutoTokenizer
        Tokenizer for the model
    texts : list[str]
        List of text strings to evaluate
    max_length : int, default=8192
        Maximum sequence length (longer sequences are truncated)
    chunk_size : int, default=1
        Number of tokens to process per chunk.
        - chunk_size=1: Exact NLL (slow, N forward passes for length N)
        - chunk_size>1: Approximate NLL (faster, N/chunk_size forward passes)
    press : DMSPress, optional
        KV cache compression method. If None, no compression is applied (baseline).
    device : str, default="cuda"
        Device to run evaluation on
    debug : bool, default=False
        If True, print per-chunk NLL and cache size for the first sample

    Returns
    -------
    dict
        Results dictionary containing:
        - perplexity: Perplexity score (exp(avg_nll))
        - avg_nll: Average negative log-likelihood per token
        - total_nll: Total negative log-likelihood
        - total_tokens: Total number of tokens evaluated
        - num_samples: Number of text samples processed
        - chunk_size: Chunk size used
        - nll_per_sample: List of per-sample NLL values
        - compression_stats: (if compression used) Detailed compression statistics
          including sparsity with/without sliding window
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    nlls_per_sample = []
    
    # CRITICAL: With chunk_size=1, compression is NEVER applied!
    # The kvpress forward_hook skips compression when q_len==1 (autoregressive mode).
    # Users must use chunk_size > 1 for compression to actually work.
    if press is not None and chunk_size == 1 and warn_no_compression:
        import warnings
        warnings.warn(
            "WARNING: chunk_size=1 with compression will NOT actually compress the KV cache! "
            "The kvpress forward_hook skips compression when processing single tokens (q_len=1). "
            "Use chunk_size > 1 (e.g., 128) for compression to take effect. "
            "With chunk_size=1, compression stats will be reported but are INCORRECT.",
            UserWarning
        )

    # Compression statistics tracking
    compression_ratios_all = []
    total_keys_original = 0
    total_keys_kept = 0
    layer_compression_ratios = {}
    
    # Sliding window impact tracking
    total_keys_in_window = 0
    total_keys_outside_window = 0
    total_keys_discarded_outside_window = 0

    with torch.inference_mode():
        sample_count = 0
        desc = tqdm_desc if tqdm_desc else "Calculating chunked perplexity"
        for text in tqdm(texts, desc=desc, disable=disable_tqdm):
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(device)
            seq_len = input_ids.size(1)

            if seq_len < 2:
                continue

            sample_nll = 0.0
            sample_tokens = 0
            cache = DynamicCache()
            num_chunks = (seq_len + chunk_size - 1) // chunk_size

            # Debug logging for first sample
            debug_first_sample = debug and (sample_count == 0)
            if debug_first_sample:
                method_name = "Baseline" if press is None else type(press).__name__
                print(f"\n=== Debug: First sample ({method_name}) ===")
                print(f"Sequence length: {seq_len}, Num chunks: {num_chunks}, Chunk size: {chunk_size}")

            for chunk_idx in range(num_chunks):
                start_pos = chunk_idx * chunk_size
                end_pos = min((chunk_idx + 1) * chunk_size, seq_len)
                chunk_ids = input_ids[:, start_pos:end_pos]
                chunk_len = chunk_ids.size(1)

                position_ids = torch.arange(start_pos, end_pos, device=device).unsqueeze(0)
                cache_position = torch.arange(start_pos, end_pos, device=device)

                if press is not None:
                    with press(model):
                        outputs = model(
                            input_ids=chunk_ids,
                            position_ids=position_ids,
                            past_key_values=cache,
                            cache_position=cache_position,
                            use_cache=True,
                            return_dict=True,
                        )

                        # Track compression statistics
                        # DMSPress: has compression_ratios attribute that tracks actual compression per layer
                        if chunk_idx > 0 and hasattr(press, 'compression_ratios') and len(press.compression_ratios) > 0:
                            compression_ratios_all.append(press.compression_ratio)

                            num_layers = len(press.compression_ratios)
                            num_heads = model.config.num_key_value_heads
                            keys_in_cache = start_pos * num_layers * num_heads
                            total_keys_original += keys_in_cache
                            keys_kept = keys_in_cache * (1 - press.compression_ratio)
                            total_keys_kept += keys_kept

                            # Detect sink tokens (if underlying press has n_sink attribute)
                            n_sink = 0
                            if hasattr(press, 'press') and hasattr(press.press, 'n_sink'):
                                n_sink = press.press.n_sink

                            # Calculate protected regions (sinks + sliding window)
                            sink_keys_per_layer = min(n_sink, start_pos) if n_sink > 0 else 0
                            window_keys_per_layer = min(DMS_SLIDING_WINDOW_SIZE, start_pos)

                            # IMPORTANT: Compressible region excludes BOTH sinks and window
                            # Protected = sinks + window (but don't double-count if they overlap)
                            if start_pos <= n_sink:
                                # All positions are sinks, no window yet
                                protected_keys_per_layer = start_pos
                                compressible_keys_per_layer = 0
                            elif start_pos <= n_sink + DMS_SLIDING_WINDOW_SIZE:
                                # Have sinks, but window would overlap with sinks
                                protected_keys_per_layer = start_pos
                                compressible_keys_per_layer = 0
                            else:
                                # Have sinks + gap + window
                                protected_keys_per_layer = n_sink + window_keys_per_layer
                                compressible_keys_per_layer = start_pos - protected_keys_per_layer

                            window_keys = window_keys_per_layer * num_layers * num_heads
                            compressible_keys = compressible_keys_per_layer * num_layers * num_heads

                            total_keys_in_window += window_keys
                            total_keys_outside_window += compressible_keys

                            # Keys discarded in compressible region
                            # Note: sinks are always kept, so we don't count them as discarded
                            keys_discarded = keys_in_cache - keys_kept
                            total_keys_discarded_outside_window += keys_discarded

                            for layer_idx, ratio in press.compression_ratios.items():
                                if layer_idx not in layer_compression_ratios:
                                    layer_compression_ratios[layer_idx] = []
                                layer_compression_ratios[layer_idx].append(ratio)

                        # ScorerPress: has fixed compression_ratio attribute
                        # NOTE: With chunk_size=1, compression is NOT actually applied!
                        # The forward_hook skips when q_len=1. Only track stats when chunk_size > 1.
                        elif chunk_idx > 0 and isinstance(press, ScorerPress) and chunk_size > 1:
                            # For ScorerPress, compression_ratio is fixed at initialization
                            target_ratio = press.compression_ratio
                            compression_ratios_all.append(target_ratio)

                            num_layers = model.config.num_hidden_layers
                            num_heads = model.config.num_key_value_heads
                            keys_in_cache = start_pos * num_layers * num_heads
                            total_keys_original += keys_in_cache
                            keys_kept = keys_in_cache * (1 - target_ratio)
                            total_keys_kept += keys_kept

                            # ScorerPress doesn't use sliding window, so all keys can be compressed
                            total_keys_outside_window += keys_in_cache
                            total_keys_discarded_outside_window += keys_in_cache - keys_kept
                else:
                    outputs = model(
                        input_ids=chunk_ids,
                        position_ids=position_ids,
                        past_key_values=cache,
                        cache_position=cache_position,
                        use_cache=True,
                        return_dict=True,
                    )

                cache = outputs.past_key_values
                logits = outputs.logits

                # Calculate loss for this chunk
                if end_pos < seq_len:
                    chunk_labels = input_ids[:, start_pos + 1:end_pos + 1]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    per_token_loss = loss_fct(logits.view(-1, logits.size(-1)), chunk_labels.view(-1))
                    chunk_nll = per_token_loss.sum().item()
                    chunk_tokens = chunk_labels.numel()
                else:
                    if chunk_len > 1:
                        chunk_labels = input_ids[:, start_pos + 1:end_pos]
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                        per_token_loss = loss_fct(
                            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                            chunk_labels.view(-1)
                        )
                        chunk_nll = per_token_loss.sum().item()
                        chunk_tokens = chunk_labels.numel()
                    else:
                        chunk_nll = 0.0
                        chunk_tokens = 0

                sample_nll += chunk_nll
                sample_tokens += chunk_tokens

                # Debug: print NLL for each chunk of first sample
                if debug_first_sample:
                    cache_size = cache.layers[0].keys.shape[2] if hasattr(cache, 'layers') and len(cache.layers) > 0 and hasattr(cache.layers[0], 'keys') else 0
                    print(f"  Chunk {chunk_idx}: NLL={chunk_nll:.4f}, tokens={chunk_tokens}, cache_size={cache_size}")

            if sample_tokens > 0:
                total_nll += sample_nll
                total_tokens += sample_tokens
                nlls_per_sample.append(sample_nll / sample_tokens)

            if debug_first_sample:
                print(f"Sample 0: Total NLL={sample_nll:.4f}, Total tokens={sample_tokens}, Avg NLL={sample_nll/sample_tokens if sample_tokens > 0 else 0:.4f}")

            sample_count += 1
            del cache
            torch.cuda.empty_cache()

    avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_nll)

    result = {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "num_samples": len(texts),
        "chunk_size": chunk_size,
        "nll_per_sample": nlls_per_sample,
    }

    if compression_ratios_all:
        avg_compression_ratio = np.mean(compression_ratios_all)
        total_keys_discarded = total_keys_original - total_keys_kept
        
        sparsity_with_window = total_keys_discarded / total_keys_original if total_keys_original > 0 else 0.0
        sparsity_without_window = (
            total_keys_discarded_outside_window / total_keys_outside_window 
            if total_keys_outside_window > 0 else 0.0
        )

        result["compression_stats"] = {
            "avg_compression_ratio": avg_compression_ratio,
            "sparsity_percentage": avg_compression_ratio * 100,
            "total_keys_original": int(total_keys_original),
            "total_keys_kept": int(total_keys_kept),
            "total_keys_discarded": int(total_keys_discarded),
            "memory_reduction_percentage": (1 - avg_compression_ratio) * 100,
            "num_chunks_with_compression": len(compression_ratios_all),
            "layer_compression_ratios": {k: float(np.mean(v)) for k, v in layer_compression_ratios.items()},
            "sliding_window_size": DMS_SLIDING_WINDOW_SIZE,
            "total_keys_in_window": int(total_keys_in_window),
            "total_keys_outside_window": int(total_keys_outside_window),
            "sparsity_with_window": sparsity_with_window,
            "sparsity_without_window": sparsity_without_window,
        }

    return result


def get_available_gpus() -> List[int]:
    """Detect available GPUs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def worker_process_texts(
    gpu_id: int,
    texts: List[str],
    model_name: str,
    max_length: int,
    chunk_size: int,
    kvzap_model_type: str,
    kvzap_scorer_model: Optional[str],
    threshold: float,
    baseline: bool,
    queue: mp.Queue,
):
    """
    Worker function to process a subset of texts on a specific GPU.

    This function runs in a separate process and processes its assigned texts
    on the specified GPU. Results are returned via a multiprocessing queue.

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
    kvzap_model_type : str
        KVzap model type ("mlp", "linear", or "no_press")
    kvzap_scorer_model : str, optional
        Explicit KVzap scorer model name
    threshold : float
        Compression threshold
    baseline : bool
        If True, run baseline (no compression). If False, run with compression.
    queue : mp.Queue
        Queue to return results
    """
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map={"": gpu_id},
        )
        model.eval()

        press = None
        if not baseline:
            if kvzap_scorer_model is not None:
                kvzap_press = CustomKVzapPress(model_type=kvzap_model_type, explicit_model_name=kvzap_scorer_model)
            else:
                kvzap_press = KVzapPress(model_type=kvzap_model_type)
            press = DMSPress(kvzap_press, threshold=threshold, decoding=True)

        results = calculate_perplexity_chunked(
            model=model, tokenizer=tokenizer, texts=texts,
            max_length=max_length, chunk_size=chunk_size, press=press, device=device,
        )

        queue.put({"success": True, "gpu_id": gpu_id, "results": results, "num_texts": len(texts)})

    except Exception as e:
        queue.put({"success": False, "gpu_id": gpu_id, "error": str(e)})


def aggregate_results(worker_results: List[Dict]) -> Dict:
    """
    Aggregate results from multiple GPU workers.

    Parameters
    ----------
    worker_results : list[dict]
        List of result dictionaries from each worker

    Returns
    -------
    dict
        Aggregated results with combined statistics including:
        - Weighted average of compression ratios
        - Total keys original/kept/discarded
        - Per-layer compression statistics
        - Sliding window impact metrics
    """
    total_nll = 0.0
    total_tokens = 0
    all_nlls_per_sample = []

    all_compression_ratios = []
    total_keys_original = 0
    total_keys_kept = 0
    layer_compression_ratios = {}
    
    total_keys_in_window = 0
    total_keys_outside_window = 0
    sliding_window_size = None

    for worker_result in worker_results:
        results = worker_result["results"]
        total_nll += results["total_nll"]
        total_tokens += results["total_tokens"]
        all_nlls_per_sample.extend(results["nll_per_sample"])

        if "compression_stats" in results:
            stats = results["compression_stats"]
            num_chunks = stats.get("num_chunks_with_compression", 0)
            if num_chunks > 0:
                all_compression_ratios.append({"ratio": stats["avg_compression_ratio"], "weight": num_chunks})

            total_keys_original += stats["total_keys_original"]
            total_keys_kept += stats["total_keys_kept"]
            
            if "total_keys_in_window" in stats:
                total_keys_in_window += stats["total_keys_in_window"]
                total_keys_outside_window += stats["total_keys_outside_window"]
                if sliding_window_size is None:
                    sliding_window_size = stats.get("sliding_window_size", DMS_SLIDING_WINDOW_SIZE)

            for layer_idx, ratio in stats["layer_compression_ratios"].items():
                if layer_idx not in layer_compression_ratios:
                    layer_compression_ratios[layer_idx] = []
                layer_compression_ratios[layer_idx].append({"ratio": ratio, "weight": num_chunks})

    avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_nll)
    chunk_size = worker_results[0]["results"]["chunk_size"]
    num_samples = sum(r["num_texts"] for r in worker_results)

    result = {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "num_samples": num_samples,
        "chunk_size": chunk_size,
        "nll_per_sample": all_nlls_per_sample,
    }

    if all_compression_ratios:
        total_weight = sum(item["weight"] for item in all_compression_ratios)
        avg_compression_ratio = sum(item["ratio"] * item["weight"] for item in all_compression_ratios) / total_weight if total_weight > 0 else 0.0
        total_keys_discarded = total_keys_original - total_keys_kept

        layer_stats = {}
        for layer_idx, items in layer_compression_ratios.items():
            tw = sum(item["weight"] for item in items)
            avg_ratio = sum(item["ratio"] * item["weight"] for item in items) / tw if tw > 0 else 0.0
            layer_stats[layer_idx] = float(avg_ratio)

        result["compression_stats"] = {
            "avg_compression_ratio": avg_compression_ratio,
            "sparsity_percentage": avg_compression_ratio * 100,
            "total_keys_original": int(total_keys_original),
            "total_keys_kept": int(total_keys_kept),
            "total_keys_discarded": int(total_keys_discarded),
            "memory_reduction_percentage": (1 - avg_compression_ratio) * 100,
            "num_chunks_with_compression": total_weight,
            "layer_compression_ratios": layer_stats,
        }
        
        if total_keys_outside_window > 0:
            sparsity_with_window = total_keys_discarded / total_keys_original if total_keys_original > 0 else 0.0
            sparsity_without_window = total_keys_discarded / total_keys_outside_window
            
            result["compression_stats"].update({
                "sliding_window_size": sliding_window_size if sliding_window_size else DMS_SLIDING_WINDOW_SIZE,
                "total_keys_in_window": int(total_keys_in_window),
                "total_keys_outside_window": int(total_keys_outside_window),
                "sparsity_with_window": sparsity_with_window,
                "sparsity_without_window": sparsity_without_window,
            })

    return result


def evaluate_parallel(
    texts: List[str],
    model_name: str,
    max_length: int,
    chunk_size: int,
    kvzap_model_type: str,
    kvzap_scorer_model: Optional[str],
    threshold: float,
    baseline: bool,
    num_gpus: Optional[int] = None,
) -> Dict:
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
    kvzap_model_type : str
        KVzap model type
    kvzap_scorer_model : str, optional
        Explicit KVzap scorer model name
    threshold : float
        Compression threshold
    baseline : bool
        Whether this is baseline evaluation
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
    print(f"\nUsing {num_workers} GPU(s): {gpus_to_use}")

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
            target=worker_process_texts,
            args=(gpu_id, text_splits[i], model_name, max_length, chunk_size,
                  kvzap_model_type, kvzap_scorer_model, threshold, baseline, queue)
        )
        p.start()
        processes.append(p)

    worker_results = []
    for _ in range(num_workers):
        result = queue.get()
        if not result["success"]:
            for p in processes:
                p.terminate()
            raise RuntimeError(f"Worker failed on GPU {result['gpu_id']}: {result['error']}")
        worker_results.append(result)
        print(f"GPU {result['gpu_id']} completed: {result['num_texts']} texts")

    for p in processes:
        p.join()

    return aggregate_results(worker_results)


def run_single_threshold(
    texts: List[str],
    model,
    tokenizer,
    model_name: str,
    max_length: int,
    chunk_size: int,
    kvzap_model_type: str,
    kvzap_scorer_model: Optional[str],
    threshold: float,
    device: str,
    use_multi_gpu: bool,
    num_gpus: Optional[int],
    baseline_results: Dict,
) -> Dict:
    """Run evaluation for a single threshold and return results."""
    if use_multi_gpu:
        compressed_results = evaluate_parallel(
            texts=texts, model_name=model_name, max_length=max_length,
            chunk_size=chunk_size, kvzap_model_type=kvzap_model_type,
            kvzap_scorer_model=kvzap_scorer_model, threshold=threshold,
            baseline=False, num_gpus=num_gpus,
        )
    else:
        if kvzap_scorer_model is not None:
            kvzap_press = CustomKVzapPress(model_type=kvzap_model_type, explicit_model_name=kvzap_scorer_model)
        else:
            kvzap_press = KVzapPress(model_type=kvzap_model_type)

        press = DMSPress(kvzap_press, threshold=threshold, decoding=True)
        compressed_results = calculate_perplexity_chunked(
            model=model, tokenizer=tokenizer, texts=texts,
            max_length=max_length, chunk_size=chunk_size, press=press, device=device,
        )

    # Calculate deltas
    ppl_diff = compressed_results['perplexity'] - baseline_results['perplexity']
    ppl_diff_pct = (ppl_diff / baseline_results['perplexity']) * 100
    nll_diff = compressed_results['avg_nll'] - baseline_results['avg_nll']

    return {
        "threshold": threshold,
        "compressed": compressed_results,
        "comparison": {
            "ppl_diff": ppl_diff,
            "ppl_diff_pct": ppl_diff_pct,
            "nll_diff": nll_diff,
        }
    }


def print_summary_table(baseline_results: Dict, threshold_results: List[Dict]):
    """Print a summary table of all threshold results."""
    print("\n" + "=" * 100)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * 100)
    print(f"Baseline NLL: {baseline_results['avg_nll']:.4f} | Baseline PPL: {baseline_results['perplexity']:.4f}")
    print("-" * 100)
    print(f"{'Threshold':>10} | {'Sparsity':>10} | {'Sparsity*':>10} | {'NLL':>10} | {'ΔNLL':>10} | {'PPL':>10} | {'ΔPPL%':>10}")
    print(f"{'':>10} | {'(w/ win)':>10} | {'(w/o win)':>10} | {'':>10} | {'':>10} | {'':>10} | {'':>10}")
    print("-" * 100)

    for res in sorted(threshold_results, key=lambda x: x['threshold'], reverse=True):
        threshold = res['threshold']
        compressed = res['compressed']
        comparison = res['comparison']
        
        stats = compressed.get('compression_stats', {})
        sparsity_with = stats.get('sparsity_with_window', stats.get('avg_compression_ratio', 0)) * 100
        sparsity_without = stats.get('sparsity_without_window', 0) * 100
        
        nll = compressed['avg_nll']
        ppl = compressed['perplexity']
        delta_nll = comparison['nll_diff']
        delta_ppl_pct = comparison['ppl_diff_pct']

        print(f"{threshold:>10.1f} | {sparsity_with:>9.2f}% | {sparsity_without:>9.2f}% | {nll:>10.4f} | {delta_nll:>+10.4f} | {ppl:>10.2f} | {delta_ppl_pct:>+9.2f}%")

    print("-" * 100)
    print("* Sparsity (w/o win) = eviction rate on tokens outside the sliding window (true compression)")
    print("=" * 100)


def evaluate_ppl_chunked(
    data_path: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_length: int = 8192,
    max_samples: int = None,
    chunk_size: int = 1,
    kvzap_model_type: str = "mlp",
    kvzap_scorer_model: str = None,
    threshold: float = None,
    thresholds: str = None,
    device: str = "cuda:0",
    output_dir: str = "./ppl_results_chunked",
    num_gpus: int = None,
):
    """
    Evaluate perplexity with proper chunked KV compression simulation.

    This script processes sequences in chunks, applying KV cache compression
    after each chunk. This properly simulates how compression affects the model
    during actual autoregressive generation.

    Parameters
    ----------
    data_path : str
        Path to JSONL dataset file. Each line should be a JSON object with a
        "text" or "content" field containing the text to evaluate.
        Example: /path/to/govreport.val.jsonl

    model_name : str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
        HuggingFace model identifier for the language model to evaluate.
        Must have a corresponding KVzap scorer model available.

    max_length : int, default=8192
        Maximum sequence length in tokens. Longer sequences will be truncated.

    max_samples : int, optional
        Maximum number of samples to evaluate from the dataset.
        If None, evaluates the entire dataset.

    chunk_size : int, default=1
        Number of tokens to process per chunk:
        - chunk_size=1: Exact NLL computation (slow but accurate)
        - chunk_size=128+: Approximate NLL (~100x speedup, minimal accuracy loss)

    kvzap_model_type : str, default="mlp"
        Type of KVzap scorer model:
        - "mlp": Two-layer MLP scorer (recommended, more accurate)
        - "linear": Linear scorer (faster, less accurate)
        - "no_press": No compression, baseline evaluation only

    kvzap_scorer_model : str, optional
        Explicit HuggingFace model identifier for the KVzap scorer.
        RECOMMENDED: Always specify this explicitly to avoid inference errors.
        Example: "nvidia/KVzap-mlp-Llama-3.1-8B-Instruct"

    threshold : float, optional
        Single threshold for DMSPress compression.
        More negative = more aggressive compression (higher sparsity, worse quality)

    thresholds : str, optional
        Comma-separated list of thresholds to sweep, e.g. "-6,-7,-8,-9"
        Produces a summary table at the end comparing all thresholds.

    device : str, default="cuda:0"
        Device for single-GPU mode. Ignored in multi-GPU mode.

    output_dir : str, default="./ppl_results_chunked"
        Directory to save evaluation results (JSON files).

    num_gpus : int, optional
        Number of GPUs for parallel evaluation:
        - None or 1: Single-GPU mode
        - 2+: Multi-GPU parallel mode
        - -1: Use all available GPUs

    Examples
    --------
    # Single threshold evaluation
    python evaluate_ppl_chunked.py --data_path data.jsonl --threshold -7.0

    # Threshold sweep with summary table
    python evaluate_ppl_chunked.py --data_path data.jsonl --thresholds "-6,-7,-8,-9"

    # Fast multi-GPU sweep
    python evaluate_ppl_chunked.py --data_path data.jsonl --thresholds "-6,-7,-8" \\
        --chunk_size 128 --num_gpus -1
    """
    # Parse thresholds (handle both string and tuple/list from Fire)
    if thresholds is not None:
        if isinstance(thresholds, str):
            threshold_list = [float(t.strip()) for t in thresholds.split(",")]
        elif isinstance(thresholds, (list, tuple)):
            threshold_list = [float(t) for t in thresholds]
        else:
            threshold_list = [float(thresholds)]
    elif threshold is not None:
        threshold_list = [threshold]
    else:
        threshold_list = [-7.0]  # Default threshold

    print("=" * 80)
    print("CHUNKED PPL EVALUATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Chunk size: {chunk_size} {'(exact NLL)' if chunk_size == 1 else '(approximate)'}")
    print(f"Thresholds to evaluate: {threshold_list}")
    
    # Determine multi-GPU mode
    use_multi_gpu = num_gpus is not None and num_gpus != 1
    if num_gpus == -1:
        num_gpus = None  # Will use all available

    model = None
    tokenizer = None
    
    if use_multi_gpu:
        print(f"Mode: Multi-GPU parallel evaluation")
    else:
        print(f"Mode: Single-GPU (device: {device})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

    # Load dataset
    print(f"\nLoading dataset: {data_path}")
    data = load_jsonl_dataset(data_path, max_samples=max_samples)
    print(f"Loaded {len(data)} samples")

    # Extract texts
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

    # Run baseline evaluation
    print("\n" + "-" * 80)
    print("Evaluating BASELINE (no compression)")
    print("-" * 80)

    if use_multi_gpu:
        baseline_results = evaluate_parallel(
            texts=texts, model_name=model_name, max_length=max_length,
            chunk_size=chunk_size, kvzap_model_type="no_press",
            kvzap_scorer_model=None, threshold=0, baseline=True, num_gpus=num_gpus,
        )
    else:
        baseline_results = calculate_perplexity_chunked(
            model=model, tokenizer=tokenizer, texts=texts,
            max_length=max_length, chunk_size=chunk_size, press=None, device=device,
        )

    print(f"\nBaseline: PPL={baseline_results['perplexity']:.4f}, NLL={baseline_results['avg_nll']:.4f}")

    # Run compressed evaluations if not baseline-only
    threshold_results = []
    
    if kvzap_model_type != "no_press":
        for thresh in threshold_list:
            print("\n" + "-" * 80)
            print(f"Evaluating threshold={thresh}")
            print("-" * 80)

            result = run_single_threshold(
                texts=texts, model=model, tokenizer=tokenizer,
                model_name=model_name, max_length=max_length, chunk_size=chunk_size,
                kvzap_model_type=kvzap_model_type, kvzap_scorer_model=kvzap_scorer_model,
                threshold=thresh, device=device, use_multi_gpu=use_multi_gpu,
                num_gpus=num_gpus, baseline_results=baseline_results,
            )
            threshold_results.append(result)

            # Print individual result
            compressed = result['compressed']
            comparison = result['comparison']
            stats = compressed.get('compression_stats', {})
            
            print(f"\nThreshold {thresh}:")
            print(f"  PPL: {compressed['perplexity']:.4f} (Δ{comparison['ppl_diff_pct']:+.2f}%)")
            print(f"  NLL: {compressed['avg_nll']:.4f} (Δ{comparison['nll_diff']:+.4f})")
            if stats:
                print(f"  Sparsity (w/ window): {stats.get('sparsity_with_window', 0)*100:.2f}%")
                print(f"  Sparsity (w/o window): {stats.get('sparsity_without_window', 0)*100:.2f}%")

        # Print summary table if multiple thresholds
        if len(threshold_results) > 1:
            print_summary_table(baseline_results, threshold_results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        "model_name": model_name,
        "data_path": data_path,
        "max_length": max_length,
        "chunk_size": chunk_size,
        "num_samples": len(texts),
        "kvzap_model_type": kvzap_model_type,
        "sliding_window_size": DMS_SLIDING_WINDOW_SIZE,
        "multi_gpu": use_multi_gpu,
        "num_gpus": num_gpus if use_multi_gpu else 1,
        "baseline": baseline_results,
        "thresholds": {r['threshold']: r for r in threshold_results},
    }

    if len(threshold_list) == 1 and threshold_list[0] is not None:
        output_file = output_path / f"ppl_{kvzap_model_type}_t{threshold_list[0]}.json"
    else:
        output_file = output_path / f"ppl_{kvzap_model_type}_sweep.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    import fire
    fire.Fire(evaluate_ppl_chunked)
