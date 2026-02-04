# No-Chunking Baseline

## Overview

The comparison script now evaluates **two baselines**:
1. **Baseline (No Chunking)** - Processes entire sequence in one forward pass (ideal baseline)
2. **Baseline (Chunked)** - Processes sequence in chunks as specified by `--chunk_size`

## Why Two Baselines?

### The Chunking Artifact Problem

When using chunked evaluation (e.g., `chunk_size=128`), even the baseline without compression has a small quality degradation:

**Without Chunking (chunk_size = seq_len):**
```
Token 0-999: All processed together in parallel
→ Each token can attend to ALL previous tokens with full precision
→ Best possible quality (ideal baseline)
```

**With Chunking (chunk_size = 128):**
```
Chunk 0 (tokens 0-127):   Processed together ✓
Chunk 1 (tokens 128-255): Can attend to chunk 0 + within chunk 1 ✓
Chunk 2 (tokens 256-383): Can attend to chunks 0-1 + within chunk 2 ✓
...
```

**BUT**: Each chunk is processed separately, which introduces minor numerical differences compared to processing everything at once, even without compression.

### Expected Chunking Artifact

On typical long documents with `chunk_size=128`:
- **ΔNLL**: +0.0001 to +0.0010 (very small increase)
- **ΔNLL %**: +0.005% to +0.05%

Example:
```
Baseline (No Chunking) NLL: 2.130456 | PPL: 8.4213
Baseline (Chunked)     NLL: 2.130512 | PPL: 8.4218
Chunking Artifact:     ΔNLL: +0.000056 (+0.003%)
```

This small artifact is the **cost of chunked evaluation** and affects all compression methods equally.

## When It's Evaluated

### Single-GPU Mode (Default)
The no-chunking baseline **is always evaluated** automatically.

```bash
python compare_compression_methods.py \
    --data_path data.jsonl \
    --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
    --chunk_size 128
```

Output shows both:
```
Baseline (No Chunking) NLL: 2.130456 | PPL: 8.4213
Baseline (Chunked)     NLL: 2.130512 | PPL: 8.4218
Chunking Artifact:     ΔNLL: +0.000056 (+0.003%)
```

### Multi-GPU Mode
The no-chunking baseline **is NOT evaluated** (would require separate code path).

```bash
python compare_compression_methods.py \
    --data_path data.jsonl \
    --num_gpus -1
```

Only shows chunked baseline.

## Comparison Table Format

The table shows ΔNLL **relative to Baseline (Chunked)**, not the no-chunking baseline:

```
Method                         NLL         ΔNLL      ΔNLL %    Dens(all)    Dens(comp)
--------------------------------------------------------------------------------------------
Note: ΔNLL is relative to Baseline (Chunked)
      Dens(all) = Overall density (all tokens)
      Dens(comp) = Density in compressible region (excluding window, may include sinks)
--------------------------------------------------------------------------------------------
Baseline (No Chunking)   2.130456    -0.000056      -0.00%      100.00%      100.00%
Baseline (Chunked)       2.130512    +0.000000      +0.00%      100.00%      100.00%
StreamingLLM             2.145321    +0.014809      +0.70%       65.23%        0.12%
ExpectedAttention        2.152134    +0.021622      +1.02%       72.15%       18.45%
```

### Interpreting the Table

**Baseline (No Chunking)**:
- Shows ΔNLL = **negative** (better than chunked baseline)
- This is the "ideal" quality without any artifacts
- All compression methods should be compared to this conceptually

**Baseline (Chunked)**:
- Reference point (ΔNLL = 0.0 by definition)
- Has small chunking artifact vs. no-chunking baseline
- **All ΔNLL values are relative to this**

**Compression Methods**:
- ΔNLL values show degradation vs. chunked baseline
- **Total degradation** = Method ΔNLL + Chunking artifact
- Example for StreamingLLM:
  - vs. Chunked: +0.014809 (+0.70%)
  - vs. No-chunking: +0.014865 (+0.70%)
  - Chunking artifact: +0.000056 (+0.003%)

## Why Compare to Chunked Baseline?

All compression methods are evaluated with the same chunking, so they all have the **same chunking artifact**. Therefore:

1. **Fair comparison**: All methods have identical chunking effects
2. **Isolates compression impact**: ΔNLL shows only the compression degradation
3. **Practical**: In real deployment, you'd use chunking for speed

The no-chunking baseline is mainly useful for:
- Understanding the chunking overhead
- Verifying the artifact is small (should be < 0.05%)
- Research purposes

## Implementation Details

### How It Works

The no-chunking baseline is evaluated by setting `chunk_size = max_length`:

```python
result_no_chunk = calculate_perplexity_chunked(
    model=model,
    tokenizer=tokenizer,
    texts=texts,
    max_length=max_length,
    chunk_size=max_length,  # Process entire sequence at once
    press=None,
    device=device,
)
```

This forces the evaluation function to process each sequence in a **single forward pass**, regardless of sequence length (up to `max_length`).

### Memory Considerations

**Warning**: The no-chunking baseline requires more memory than chunked evaluation:

- **Chunked (128 tokens)**: ~8 GB VRAM for Llama-3.1-8B
- **No-chunking (8192 tokens)**: ~16-20 GB VRAM for Llama-3.1-8B

If sequences are very long, you may run out of memory. In that case, the script will error on the no-chunking evaluation but continue with chunked methods.

### Evaluation Order

1. **Baseline (No Chunking)** - Evaluated first (separate call)
2. **Baseline (Chunked)** - Evaluated in main loop
3. **All compression methods** - Evaluated in main loop

## Expected Results

### Typical Values

On govreport validation set with Llama-3.1-8B:

```
Baseline (No Chunking) NLL: ~2.130
Baseline (Chunked, 128) NLL: ~2.131
Chunking Artifact: ~+0.001 (+0.047%)
```

### Sanity Checks

✅ **Good**: Chunking artifact < 0.1% (typical)
⚠️ **Warning**: Chunking artifact 0.1-0.5% (check chunk_size, may be too small)
❌ **Error**: Chunking artifact > 0.5% (bug or chunk_size = 1)

### Special Case: chunk_size = 1

With `chunk_size=1`, compression **does not work** (kvpress skips single-token processing). You'll see:
- All compression methods have **identical NLL** to chunked baseline
- Chunking artifact is **larger** than with chunk_size > 1
- Warnings will be printed

**Solution**: Use `chunk_size > 1` (e.g., 128) for compression to take effect.

## Troubleshooting

### "Out of memory" on no-chunking baseline
**Cause**: Sequences too long for single forward pass
**Solution**: This is expected. The script will continue with chunked evaluation.

### Chunking artifact is negative (no-chunking is worse)
**Cause**: This shouldn't happen - possible numerical instability
**Solution**: Check if sequences are very short or if there's a bug

### Chunking artifact is very large (> 1%)
**Possible causes**:
1. `chunk_size = 1` (compression disabled)
2. Numerical instability
3. Bug in chunked evaluation

**Debug**: Try increasing chunk_size to 128 or 256

## Summary

The no-chunking baseline provides:
- ✅ **Gold standard quality** for comparison
- ✅ **Quantifies chunking overhead**
- ✅ **Validates chunked evaluation** is working correctly

In practice:
- Use **chunked baseline** as reference for ΔNLL comparisons
- Check **chunking artifact** is small (< 0.1%)
- Use **no-chunking baseline** to understand total degradation
