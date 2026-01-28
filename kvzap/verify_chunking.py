#!/usr/bin/env python3
"""
Verify that chunking does NOT affect NLL for the baseline (non-compressed) model.

This is a sanity check: the NLL should be identical regardless of chunk_size
when no compression is applied, since the KV cache accumulates all tokens
and each token attends to all previous tokens.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

def compute_nll_single_pass(model, tokenizer, text, device="cuda"):
    """Compute NLL in a single forward pass (standard approach)."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)
    
    with torch.inference_mode():
        outputs = model(input_ids, labels=input_ids)
    
    # HuggingFace loss is mean NLL over (seq_len - 1) predictions
    seq_len = input_ids.size(1)
    total_nll = outputs.loss.item() * (seq_len - 1)
    return total_nll, seq_len - 1


def compute_nll_chunked(model, tokenizer, text, chunk_size, device="cuda"):
    """Compute NLL using chunked processing with KV cache."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)
    
    total_nll = 0.0
    total_tokens = 0
    cache = DynamicCache()
    
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    with torch.inference_mode():
        for chunk_idx in range(num_chunks):
            start_pos = chunk_idx * chunk_size
            end_pos = min((chunk_idx + 1) * chunk_size, seq_len)
            chunk_ids = input_ids[:, start_pos:end_pos]
            chunk_len = chunk_ids.size(1)
            
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
            
            cache = outputs.past_key_values
            logits = outputs.logits
            
            # Compute loss for this chunk
            if end_pos < seq_len:
                chunk_labels = input_ids[:, start_pos + 1:end_pos + 1]
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                per_token_loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    chunk_labels.view(-1)
                )
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
            
            total_nll += chunk_nll
            total_tokens += chunk_tokens
    
    return total_nll, total_tokens


def main():
    print("=" * 70)
    print("VERIFYING: Chunking should NOT affect baseline NLL")
    print("=" * 70)
    
    # Use a small model for quick testing
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"\nLoading model: {model_name}")
    
    # Test with float32 to rule out precision issues
    dtype = torch.float32
    print(f"Using dtype: {dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    
    # Test text
    test_texts = [
        "The quick brown fox jumps over the lazy dog. This is a test.",
        "Machine learning models can be evaluated using perplexity metrics.",
    ]
    
    chunk_sizes = [1, 4, 16, 64, 128]
    
    for text in test_texts:
        print(f"\n{'='*70}")
        print(f"Text: {text[:50]}...")
        print(f"{'='*70}")
        
        # Single pass baseline
        nll_single, tokens_single = compute_nll_single_pass(model, tokenizer, text)
        avg_nll_single = nll_single / tokens_single
        print(f"\nSingle pass:     NLL={nll_single:.6f}, tokens={tokens_single}, avg={avg_nll_single:.6f}")
        
        # Chunked with different sizes
        for chunk_size in chunk_sizes:
            nll_chunked, tokens_chunked = compute_nll_chunked(model, tokenizer, text, chunk_size)
            avg_nll_chunked = nll_chunked / tokens_chunked
            diff = abs(avg_nll_chunked - avg_nll_single)
            status = "✓" if diff < 1e-4 else "✗"
            print(f"Chunk size {chunk_size:3d}:  NLL={nll_chunked:.6f}, tokens={tokens_chunked}, avg={avg_nll_chunked:.6f}  diff={diff:.2e} {status}")
    
    print(f"\n{'='*70}")
    print("If all diffs are < 1e-4, chunking is working correctly!")
    print("=" * 70)


if __name__ == "__main__":
    main()
