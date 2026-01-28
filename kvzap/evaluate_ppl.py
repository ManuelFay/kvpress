#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PPL evaluation script for KVzap with custom dataset.

This script evaluates perplexity with and without KV compression on a JSONL dataset.
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import KVzapPress, DMSPress
from kvpress.presses.kvzap_press import KVzapModel


class CustomKVzapPress(KVzapPress):
    """Custom KVzapPress that allows explicit model name override."""

    def __init__(self, model_type="mlp", explicit_model_name=None):
        super().__init__(model_type=model_type)
        self.explicit_model_name = explicit_model_name

    def post_init_from_model(self, model):
        if self.explicit_model_name is not None:
            # Use explicitly provided model name, but only load if not already loaded
            if self.explicit_model_name != self.kvzap_model_name:
                self.kvzap_model_name = self.explicit_model_name
                self.kvzap_model = KVzapModel.from_pretrained(self.kvzap_model_name)
        else:
            # Use default inference logic
            super().post_init_from_model(model)


def load_jsonl_dataset(file_path, max_samples=None):
    """
    Load a JSONL dataset.

    Parameters
    ----------
    file_path : str
        Path to the JSONL file
    max_samples : int, optional
        Maximum number of samples to load

    Returns
    -------
    list
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            data.append(json.loads(line))
    return data


def calculate_perplexity(
    model,
    tokenizer,
    texts,
    max_length=8192,
    press=None,
    batch_size=1,
    device="cuda"
):
    """
    Calculate perplexity on a list of texts.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The language model
    tokenizer : AutoTokenizer
        The tokenizer
    texts : list
        List of text strings to evaluate
    max_length : int
        Maximum sequence length
    press : KVPress, optional
        KV cache compression method
    batch_size : int
        Batch size for evaluation
    device : str
        Device to use

    Returns
    -------
    dict
        Dictionary with perplexity, NLL statistics, and compression stats
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    nlls = []

    # Compression statistics
    compression_ratios = []
    total_keys_original = 0
    total_keys_kept = 0
    layer_compression_ratios = {}

    with torch.inference_mode():
        for text in tqdm(texts, desc="Calculating perplexity"):
            # Tokenize
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encodings.input_ids.to(device)

            # Skip if sequence is too short
            seq_len = input_ids.size(1)
            if seq_len < 2:
                continue

            # Apply press if provided
            if press is not None:
                with press(model):
                    outputs = model(input_ids, labels=input_ids)

                    # Collect compression statistics from DMSPress
                    if hasattr(press, 'compression_ratios') and len(press.compression_ratios) > 0:
                        compression_ratios.append(press.compression_ratio)

                        # Calculate total keys
                        # num_predictions * num_layers * num_heads
                        num_predictions = seq_len - 1
                        num_layers = len(press.compression_ratios)
                        # Assuming the model config is available
                        num_heads = model.config.num_key_value_heads

                        keys_per_sample = num_predictions * num_layers * num_heads
                        total_keys_original += keys_per_sample

                        # Calculate keys kept (1 - compression_ratio)
                        keys_kept = keys_per_sample * (1 - press.compression_ratio)
                        total_keys_kept += keys_kept

                        # Track per-layer compression
                        for layer_idx, ratio in press.compression_ratios.items():
                            if layer_idx not in layer_compression_ratios:
                                layer_compression_ratios[layer_idx] = []
                            layer_compression_ratios[layer_idx].append(ratio)
            else:
                outputs = model(input_ids, labels=input_ids)

            # Calculate NLL
            # For causal LM, HF shifts labels internally, so we get N-1 predictions for N tokens
            num_predictions = seq_len - 1
            # The loss is the mean NLL per prediction
            nll = outputs.loss.item() * num_predictions

            total_nll += nll
            total_tokens += num_predictions
            nlls.append(outputs.loss.item())

            # Clear cache
            torch.cuda.empty_cache()

    # Calculate perplexity
    avg_nll = total_nll / total_tokens
    perplexity = np.exp(avg_nll)

    result = {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "num_samples": len(texts),
        "nll_per_sample": nlls,
    }

    # Add compression statistics if available
    if compression_ratios:
        avg_compression_ratio = np.mean(compression_ratios)
        total_keys_discarded = total_keys_original - total_keys_kept

        result["compression_stats"] = {
            "avg_compression_ratio": avg_compression_ratio,
            "total_keys_original": int(total_keys_original),
            "total_keys_kept": int(total_keys_kept),
            "total_keys_discarded": int(total_keys_discarded),
            "sparsity_percentage": avg_compression_ratio * 100,
            "layer_compression_ratios": {k: float(np.mean(v)) for k, v in layer_compression_ratios.items()},
        }

    return result


def evaluate_ppl(
    data_path: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_length: int = 8192,
    max_samples: int = None,
    kvzap_model_type: str = "mlp",
    kvzap_scorer_model: str = None,
    threshold: float = -4.0,
    device: str = "cuda:0",
    output_dir: str = "./ppl_results",
):
    """
    Evaluate perplexity with and without KV compression.

    Parameters
    ----------
    data_path : str
        Path to JSONL dataset file
    model_name : str
        HuggingFace model name
    max_length : int
        Maximum sequence length
    max_samples : int, optional
        Maximum number of samples to evaluate
    kvzap_model_type : str
        KVzap model type ("mlp", "linear", or "no_press")
    threshold : float
        Threshold for DMSPress
    device : str
        Device to use
    output_dir : str
        Directory to save results
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Loading dataset: {data_path}")
    data = load_jsonl_dataset(data_path, max_samples=max_samples)
    print(f"Loaded {len(data)} samples")

    # Extract text from data
    # Assuming the JSONL has a "text" field - adjust if needed
    if isinstance(data[0], dict):
        if "text" in data[0]:
            texts = [item["text"] for item in data]
        elif "content" in data[0]:
            texts = [item["content"] for item in data]
        else:
            # Use the first string field found
            first_key = list(data[0].keys())[0]
            texts = [item[first_key] for item in data]
    else:
        texts = data

    # Evaluate without compression (baseline)
    print("\n" + "="*80)
    print("Evaluating WITHOUT KV compression (baseline)")
    print("="*80)
    baseline_results = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        press=None,
        device=device,
    )
    print(f"\nBaseline Results:")
    print(f"  Perplexity: {baseline_results['perplexity']:.4f}")
    print(f"  Avg NLL: {baseline_results['avg_nll']:.4f}")
    print(f"  Total tokens: {baseline_results['total_tokens']}")

    # Evaluate with KV compression
    if kvzap_model_type != "no_press":
        print("\n" + "="*80)
        print(f"Evaluating WITH KV compression (KVzap {kvzap_model_type}, threshold={threshold})")
        print("="*80)

        # Use CustomKVzapPress if explicit model name is provided
        if kvzap_scorer_model is not None:
            print(f"Using explicit KVzap scorer model: {kvzap_scorer_model}")
            kvzap_press = CustomKVzapPress(
                model_type=kvzap_model_type,
                explicit_model_name=kvzap_scorer_model
            )
        else:
            kvzap_press = KVzapPress(model_type=kvzap_model_type)

        press = DMSPress(
            kvzap_press,
            threshold=threshold,
            decoding=False,  # Prefilling only
        )

        compressed_results = calculate_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=max_length,
            press=press,
            device=device,
        )
        print(f"\nCompressed Results:")
        print(f"  Perplexity: {compressed_results['perplexity']:.4f}")
        print(f"  Avg NLL: {compressed_results['avg_nll']:.4f}")
        print(f"  Total tokens: {compressed_results['total_tokens']}")

        # Print compression statistics
        if "compression_stats" in compressed_results:
            stats = compressed_results["compression_stats"]
            print(f"\nCompression Statistics:")
            print(f"  Average Compression Ratio: {stats['avg_compression_ratio']:.4f}")
            print(f"  Sparsity Percentage: {stats['sparsity_percentage']:.2f}%")
            print(f"  Total Keys (Original): {stats['total_keys_original']:,}")
            print(f"  Total Keys Kept: {stats['total_keys_kept']:,}")
            print(f"  Total Keys Discarded: {stats['total_keys_discarded']:,}")
            print(f"  Memory Reduction: {(1 - stats['avg_compression_ratio']) * 100:.2f}%")
            print(f"\n  Per-Layer Compression Ratios:")
            for layer_idx in sorted(stats['layer_compression_ratios'].keys()):
                ratio = stats['layer_compression_ratios'][layer_idx]
                print(f"    Layer {layer_idx}: {ratio:.4f} ({ratio * 100:.2f}% discarded)")

        # Calculate differences
        ppl_diff = compressed_results['perplexity'] - baseline_results['perplexity']
        ppl_diff_pct = (ppl_diff / baseline_results['perplexity']) * 100
        nll_diff = compressed_results['avg_nll'] - baseline_results['avg_nll']

        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"Baseline Perplexity:    {baseline_results['perplexity']:.4f}")
        print(f"Compressed Perplexity:  {compressed_results['perplexity']:.4f}")
        print(f"Difference:             {ppl_diff:+.4f} ({ppl_diff_pct:+.2f}%)")
        print(f"\nBaseline NLL:           {baseline_results['avg_nll']:.4f}")
        print(f"Compressed NLL:         {compressed_results['avg_nll']:.4f}")
        print(f"Difference:             {nll_diff:+.4f}")

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "model_name": model_name,
            "data_path": data_path,
            "max_length": max_length,
            "num_samples": len(texts),
            "kvzap_model_type": kvzap_model_type,
            "threshold": threshold,
            "baseline": baseline_results,
            "compressed": compressed_results,
            "comparison": {
                "ppl_diff": ppl_diff,
                "ppl_diff_pct": ppl_diff_pct,
                "nll_diff": nll_diff,
            }
        }

        # Remove the nll_per_sample lists for cleaner JSON
        results["baseline"]["nll_per_sample"] = None
        results["compressed"]["nll_per_sample"] = None

        output_file = output_path / f"ppl_results_{kvzap_model_type}_threshold_{threshold}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    else:
        # Save baseline only
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "model_name": model_name,
            "data_path": data_path,
            "max_length": max_length,
            "num_samples": len(texts),
            "baseline": baseline_results,
        }

        results["baseline"]["nll_per_sample"] = None

        output_file = output_path / "ppl_results_baseline.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    import fire
    fire.Fire(evaluate_ppl)
