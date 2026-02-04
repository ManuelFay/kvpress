#!/usr/bin/env python3
"""
NLL Evaluation for Qwen3-8B-DMS-8x Model

Computes Negative Log-Likelihood (NLL) and Perplexity on the govreport validation set
using the nvidia Qwen3-8B-DMS-8x model with Dynamic Memory Sparsification.

Usage:
    python evaluate_nll_dms.py --max_samples 10 --max_length 4096
"""

# import os
# os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Import data loader
from loaders import create_loader, Sample


def compute_nll_full_sequence(
    model,
    tokenizer,
    samples: list,
    max_length: int = 4096,
    device: str = "cuda",
) -> dict:
    """
    Compute NLL by processing full sequences at once.

    This lets the DMS model handle its own internal compression.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The language model
    tokenizer : AutoTokenizer
        Tokenizer for the model
    samples : list[Sample]
        List of pre-tokenized samples
    max_length : int
        Maximum sequence length
    device : str
        Device to use

    Returns
    -------
    dict
        Results with perplexity, avg_nll, total_nll, total_tokens
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    nlls_per_sample = []

    with torch.inference_mode():
        for sample in tqdm(samples, desc="Computing NLL"):
            tokens = sample.tokens
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            input_ids = torch.tensor([tokens], device=device)
            seq_len = input_ids.size(1)

            if seq_len < 2:
                continue

            # Create attention mask (all ones - attending to all tokens)
            attention_mask = torch.ones_like(input_ids)

            # Process full sequence at once
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,  # Don't use cache for full sequence processing
                return_dict=True,
            )

            logits = outputs.logits

            # Compute NLL: predict token i+1 from position i
            # logits[:, :-1, :] contains predictions for tokens 1 to seq_len
            # input_ids[:, 1:] contains the actual tokens 1 to seq_len
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Cross entropy loss
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
        "nll_per_sample": nlls_per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLL for Qwen3-8B-DMS-8x")
    parser.add_argument("--data_path", type=str, default="/home/manufay/govreport.val.jsonl",
                        help="Path to data file")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--output_file", type=str, default="nll_results_dms.json",
                        help="Output file for results")
    args = parser.parse_args()

    print("=" * 60)
    print("NLL Evaluation: Qwen3-8B-DMS-8x")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B",
        # local_files_only=True
    )

    # Load config and set pad_token_id (missing from nvidia's config.json)
    print("Loading model config...")
    config = AutoConfig.from_pretrained(
        "nvidia/Qwen3-8B-DMS-8x",
        trust_remote_code=True,
        # local_files_only=True
    )
    config.pad_token_id = tokenizer.pad_token_id

    # Load model
    print("Loading DMS model...")
    model = AutoModelForCausalLM.from_pretrained(
        "nvidia/Qwen3-8B-DMS-8x",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # local_files_only=True
    )
    print(f"Model loaded on device: {model.device}")

    # Load data using the loader system
    print(f"\nLoading data from: {args.data_path}")
    loader = create_loader(
        loader_type="jsonl_text",
        file_path=args.data_path,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )
    samples = loader.load()
    print(f"Loaded {len(samples)} samples")

    # Print sample statistics
    lengths = [len(s.tokens) for s in samples]
    print(f"Token lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")

    # Compute NLL
    print(f"\nComputing NLL (full sequence processing)...")
    results = compute_nll_full_sequence(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        max_length=args.max_length,
        device="cuda",
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results: Qwen3-8B-DMS-8x")
    print("=" * 60)
    print(f"Perplexity: {results['perplexity']:.4f}")
    print(f"Average NLL: {results['avg_nll']:.4f}")
    print(f"Total NLL: {results['total_nll']:.4f}")
    print(f"Total tokens: {results['total_tokens']:,}")
    print(f"Number of samples: {results['num_samples']}")

    # Save results
    results_to_save = {
        "model": "nvidia/Qwen3-8B-DMS-8x",
        "data_path": args.data_path,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "perplexity": results["perplexity"],
        "avg_nll": results["avg_nll"],
        "total_nll": results["total_nll"],
        "total_tokens": results["total_tokens"],
        "num_samples": results["num_samples"],
    }

    with open(args.output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
