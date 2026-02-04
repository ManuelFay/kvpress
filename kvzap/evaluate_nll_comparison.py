#!/usr/bin/env python3
"""
Combined NLL Evaluation: Qwen3-8B vs Qwen3-8B-DMS-8x

Computes NLL and Perplexity for both models on the same dataset,
measures DMS compression density, and computes the NLL delta.

Usage:
    python evaluate_nll_comparison.py --max_samples 200 --max_length 8192
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import torch
import numpy as np
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from loaders import create_loader, Sample


def compute_nll_qwen3(
    model,
    samples: list,
    max_length: int = 4096,
    device: str = "cuda",
) -> dict:
    """Compute NLL for base Qwen3-8B model."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    nlls_per_sample = []

    with torch.inference_mode():
        for sample in tqdm(samples, desc="Computing NLL (Qwen3-8B)"):
            tokens = sample.tokens
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            input_ids = torch.tensor([tokens], device=device)
            seq_len = input_ids.size(1)

            if seq_len < 2:
                continue

            attention_mask = torch.ones_like(input_ids)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            sample_nll = loss.item()
            sample_tokens = shift_labels.numel()

            nlls_per_sample.append(sample_nll / sample_tokens)
            total_nll += sample_nll
            total_tokens += sample_tokens

    avg_nll = total_nll / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_nll) if avg_nll < 100 else float('inf')

    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "num_samples": len(nlls_per_sample),
    }


def compute_nll_dms(
    model,
    samples: list,
    max_length: int = 4096,
    device: str = "cuda",
) -> dict:
    """
    Compute NLL for DMS model and measure compression.

    DMS compression is measured by comparing actual cache size to sequence length.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    nlls_per_sample = []

    # Compression tracking
    compression_stats = []
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    dms_window_size = model.config.dms_window_size
    dms_cr = model.config.dms_cr  # Target compression ratio (8x)

    with torch.inference_mode():
        for sample in tqdm(samples, desc="Computing NLL (DMS)"):
            tokens = sample.tokens
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            input_ids = torch.tensor([tokens], device=device)
            seq_len = input_ids.size(1)

            if seq_len < 2:
                continue

            attention_mask = torch.ones_like(input_ids)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,  # Enable cache to measure compression
                return_dict=True,
            )

            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            sample_nll = loss.item()
            sample_tokens = shift_labels.numel()

            nlls_per_sample.append(sample_nll / sample_tokens)
            total_nll += sample_nll
            total_tokens += sample_tokens

            # Measure compression from the DMS cache
            past_kv = outputs.past_key_values
            if past_kv is not None and hasattr(past_kv, 'layers') and len(past_kv.layers) > 0:
                # Get cache stats from first layer (representative)
                layer_cache = past_kv.layers[0]
                if hasattr(layer_cache, 'cache_seq_lengths') and layer_cache.cache_seq_lengths is not None:
                    cache_lengths = layer_cache.cache_seq_lengths.float()
                    avg_cache_len = cache_lengths.mean().item()

                    # Density WITH window (overall)
                    density_with_window = avg_cache_len / seq_len if seq_len > 0 else 1.0

                    # Density WITHOUT window (compressible region only)
                    # Compressible region = seq_len - window_size
                    compressible_region = max(seq_len - dms_window_size, 1)
                    kept_outside_window = max(avg_cache_len - dms_window_size, 0)
                    density_without_window = kept_outside_window / compressible_region

                    compression_stats.append({
                        'seq_len': seq_len,
                        'avg_cache_len': avg_cache_len,
                        'density_with_window': density_with_window,
                        'density_without_window': density_without_window,
                        'cr_with_window': 1.0 / density_with_window if density_with_window > 0 else float('inf'),
                        'cr_without_window': 1.0 / density_without_window if density_without_window > 0 else float('inf'),
                    })

    avg_nll = total_nll / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_nll) if avg_nll < 100 else float('inf')

    # Aggregate compression stats
    if compression_stats:
        avg_density_with_window = np.mean([s['density_with_window'] for s in compression_stats])
        avg_density_without_window = np.mean([s['density_without_window'] for s in compression_stats])
        avg_cr_with_window = np.mean([s['cr_with_window'] for s in compression_stats])
        avg_cr_without_window = np.mean([s['cr_without_window'] for s in compression_stats])
    else:
        avg_density_with_window = 1.0
        avg_density_without_window = 1.0
        avg_cr_with_window = 1.0
        avg_cr_without_window = 1.0

    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "num_samples": len(nlls_per_sample),
        "density_with_window": avg_density_with_window,
        "density_without_window": avg_density_without_window,
        "cr_with_window": avg_cr_with_window,
        "cr_without_window": avg_cr_without_window,
        "target_compression_ratio": dms_cr,
        "dms_window_size": dms_window_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare NLL: Qwen3-8B vs DMS")
    parser.add_argument("--amaia_sources_config", type=str,
                        default="/home/manufay/kvpress/kvzap/amaia_sources.yaml",
                        help="Path to AMAIA sources config YAML")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_length", type=int, default=8192,
                        help="Maximum sequence length (used as amaia_seq_len)")
    parser.add_argument("--output_file", type=str, default="nll_comparison_results.json",
                        help="Output file for results")
    parser.add_argument("--amaia_path", type=str, default="/storage/home/manufay/amaia",
                        help="Path to AMAIA repository")
    args = parser.parse_args()

    print("=" * 70)
    print("NLL Comparison: Qwen3-8B vs Qwen3-8B-DMS-8x")
    print("=" * 70)
    print(f"AMAIA sources config: {args.amaia_sources_config}")
    print(f"Max samples: {args.max_samples}")
    print(f"Sequence length: {args.max_length}")

    # Load tokenizer (shared) - needed for reference but AMAIA uses its own tokenizer
    print("\n[1/5] Loading HuggingFace tokenizer (for reference)...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)

    # Load data using AMAIA loader
    print(f"\n[2/5] Loading data via AMAIA loader...")
    print(f"  Sources config: {args.amaia_sources_config}")
    print(f"  Sequence length: {args.max_length}")

    # Path to Qwen3-8B tokenizer in HF cache
    qwen_tokenizer_path = "/home/manufay/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

    loader = create_loader(
        loader_type="amaia",
        amaia_sources_config=args.amaia_sources_config,
        amaia_seq_len=args.max_length,
        max_samples=args.max_samples,
        amaia_seed=42,
        amaia_shuffle_buffer_size=1,
        amaia_tokenizer_name="huggingface",  # Use HuggingFace tokenizer
        amaia_tokenizer_path=qwen_tokenizer_path,
        amaia_path=args.amaia_path,
    )
    samples = loader.load()
    print(f"Loaded {len(samples)} samples")
    lengths = [len(s.tokens) for s in samples]
    print(f"Token lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")

    # =========================================================================
    # Evaluate Base Qwen3-8B
    # =========================================================================
    print("\n[3/5] Evaluating base Qwen3-8B...")
    print("-" * 50)

    model_qwen3 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    results_qwen3 = compute_nll_qwen3(
        model=model_qwen3,
        samples=samples,
        max_length=args.max_length,
        device="cuda",
    )

    print(f"\nQwen3-8B Results:")
    print(f"  Perplexity: {results_qwen3['perplexity']:.4f}")
    print(f"  Average NLL: {results_qwen3['avg_nll']:.4f}")
    print(f"  Total tokens: {results_qwen3['total_tokens']:,}")

    # Free memory
    del model_qwen3
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Evaluate DMS Model
    # =========================================================================
    print("\n[4/5] Evaluating Qwen3-8B-DMS-8x...")
    print("-" * 50)

    config_dms = AutoConfig.from_pretrained(
        "nvidia/Qwen3-8B-DMS-8x",
        trust_remote_code=True,
        local_files_only=True,
    )
    config_dms.pad_token_id = tokenizer.pad_token_id

    model_dms = AutoModelForCausalLM.from_pretrained(
        "nvidia/Qwen3-8B-DMS-8x",
        config=config_dms,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    results_dms = compute_nll_dms(
        model=model_dms,
        samples=samples,
        max_length=args.max_length,
        device="cuda",
    )

    print(f"\nQwen3-8B-DMS-8x Results:")
    print(f"  Perplexity: {results_dms['perplexity']:.4f}")
    print(f"  Average NLL: {results_dms['avg_nll']:.4f}")
    print(f"  Total tokens: {results_dms['total_tokens']:,}")
    print(f"  Density (with window): {results_dms['density_with_window']*100:.2f}%")
    print(f"  Density (w/o window): {results_dms['density_without_window']*100:.2f}%")
    print(f"  CR (with window): {results_dms['cr_with_window']:.2f}x")
    print(f"  CR (w/o window): {results_dms['cr_without_window']:.2f}x (target: {results_dms['target_compression_ratio']}x)")

    # Free memory
    del model_dms
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Compute Deltas
    # =========================================================================
    print("\n[5/5] Computing comparison metrics...")
    print("=" * 70)

    nll_delta = results_dms['avg_nll'] - results_qwen3['avg_nll']
    ppl_delta = results_dms['perplexity'] - results_qwen3['perplexity']
    ppl_ratio = results_dms['perplexity'] / results_qwen3['perplexity']

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Qwen3-8B':>15} {'DMS-8x':>15} {'Delta':>15}")
    print("-" * 70)
    print(f"{'Perplexity':<30} {results_qwen3['perplexity']:>15.4f} {results_dms['perplexity']:>15.4f} {ppl_delta:>+15.4f}")
    print(f"{'Average NLL':<30} {results_qwen3['avg_nll']:>15.4f} {results_dms['avg_nll']:>15.4f} {nll_delta:>+15.4f}")
    print(f"{'Total Tokens':<30} {results_qwen3['total_tokens']:>15,} {results_dms['total_tokens']:>15,}")
    print("-" * 70)
    print(f"\n{'DMS Compression Stats':}")
    print(f"  Density (with window): {results_dms['density_with_window']*100:.2f}%")
    print(f"  Density (w/o window):  {results_dms['density_without_window']*100:.2f}%")
    print(f"  CR (with window):      {results_dms['cr_with_window']:.2f}x")
    print(f"  CR (w/o window):       {results_dms['cr_without_window']:.2f}x (target: {results_dms['target_compression_ratio']}x)")
    print(f"  DMS Window Size:       {results_dms['dms_window_size']}")
    print(f"  PPL Ratio (DMS/Base):  {ppl_ratio:.4f}")
    print("=" * 70)

    # Save results
    all_results = {
        "config": {
            "amaia_sources_config": args.amaia_sources_config,
            "max_samples": args.max_samples,
            "max_length": args.max_length,
        },
        "qwen3_8b": {
            "model": "Qwen/Qwen3-8B",
            "perplexity": results_qwen3["perplexity"],
            "avg_nll": results_qwen3["avg_nll"],
            "total_tokens": results_qwen3["total_tokens"],
            "num_samples": results_qwen3["num_samples"],
        },
        "dms_8x": {
            "model": "nvidia/Qwen3-8B-DMS-8x",
            "perplexity": results_dms["perplexity"],
            "avg_nll": results_dms["avg_nll"],
            "total_tokens": results_dms["total_tokens"],
            "num_samples": results_dms["num_samples"],
            "density_with_window": results_dms["density_with_window"],
            "density_without_window": results_dms["density_without_window"],
            "cr_with_window": results_dms["cr_with_window"],
            "cr_without_window": results_dms["cr_without_window"],
            "target_compression_ratio": results_dms["target_compression_ratio"],
            "dms_window_size": results_dms["dms_window_size"],
        },
        "comparison": {
            "nll_delta": nll_delta,
            "ppl_delta": ppl_delta,
            "ppl_ratio": ppl_ratio,
        },
    }

    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
