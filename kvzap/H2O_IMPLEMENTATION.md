# H2O (Heavy-Hitter Oracle) Implementation

## Overview

I've implemented H2O (Heavy-Hitter Oracle) KV cache compression as described in the paper "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models" (https://arxiv.org/abs/2306.14048).

## What Was Added

### 1. New H2O Press Class (`/storage/home/manufay/kvpress/kvpress/presses/h2o_press.py`)

This is a faithful implementation of the H2O algorithm that combines:

**Local Window (Recent Tokens)**:
- Always preserves the last K tokens (default: 512)
- These tokens are never pruned, maintaining local context
- Assigned infinite score to guarantee retention

**Heavy Hitters (Important Historical Tokens)**:
- Older tokens (beyond local window) are scored using observed attention
- Tokens with highest cumulative attention weights are kept
- Implements the same scoring as `ObservedAttentionPress`

**Key Features**:
- Requires `attn_implementation="eager"` to access attention weights
- Compression ratio applies to the entire cache (local window + heavy hitters)
- More faithful to the H2O paper than plain `ObservedAttentionPress`

### 2. Integration into Comparison Script

Updated `compare_compression_methods.py` to support:
- H2O press factory function with configurable `local_window_size`
- Multi-GPU support for H2O evaluation
- Automatic model reloading with eager attention mode when needed
- Both single-GPU and multi-GPU evaluation paths

## Usage

### Basic H2O Evaluation

```bash
python compare_compression_methods.py \
  --data_path /path/to/data.jsonl \
  --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
  --include_observed_attention \
  --max_samples 100
```

This will evaluate:
- ObservedAttention (plain observed attention scoring)
- H2O (local window + heavy hitters)

### Customize H2O Parameters

```bash
python compare_compression_methods.py \
  --data_path /path/to/data.jsonl \
  --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
  --methods h2o kvzap baseline \
  --compression_ratio 0.6 \
  --h2o_local_window_size 1024 \
  --max_samples 100
```

**Parameters**:
- `--compression_ratio`: Fraction to evict (0.6 = remove 60% of tokens)
- `--h2o_local_window_size`: Number of recent tokens to always keep (default: 512)

### Multi-GPU H2O Evaluation

```bash
python compare_compression_methods.py \
  --data_path /path/to/data.jsonl \
  --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
  --include_observed_attention \
  --num_gpus -1 \
  --max_samples 500
```

Workers will automatically load models with `attn_implementation="eager"` for H2O and ObservedAttention.

## How It Works

### Scoring Logic

For a sequence of length N:
- **Recent tokens** (positions `N - local_window_size` to `N`): Score = ∞ (always kept)
- **Older tokens** (positions `0` to `N - local_window_size - 1`): Score = observed attention

Observed attention score for token `i`:
```python
score[i] = mean over all queries q of: attention_weight[q, i]
```

### Compression Process

1. **Compute observed attention scores** from attention weights
2. **Assign infinite scores** to local window tokens
3. **Sort all tokens** by score (local window will be at top)
4. **Keep top K tokens** where `K = N * (1 - compression_ratio)`
5. **Prune remaining tokens**

### Example

Sequence length = 1000, `compression_ratio = 0.5`, `local_window_size = 512`:
- Keep: 500 tokens total
- Local window: Last 512 tokens → **all 500 kept slots used by local window**
- Heavy hitters: 0 tokens (local window exceeded budget)

Better configuration: `compression_ratio = 0.3` (remove 30%):
- Keep: 700 tokens total
- Local window: Last 512 tokens → **512 kept**
- Heavy hitters: Top 188 tokens from positions 0-487 → **188 kept**
- Total: 700 kept, 300 removed

## Differences from ObservedAttentionPress

| Feature | ObservedAttentionPress | H2OPress |
|---------|----------------------|----------|
| Scoring | Observed attention only | Local window (∞) + Observed attention |
| Recent tokens | Scored like any other | Always preserved |
| Paper alignment | Generic attention-based | Faithful to H2O paper |
| Use case | Simple baseline | Production-ready compression |

## Performance Characteristics

**Advantages**:
- Better preservation of local context (recent tokens never pruned)
- More stable generation quality (local window acts as safety buffer)
- Aligns with H2O paper methodology

**Disadvantages**:
- Requires eager attention (slower than flash attention)
- More memory overhead during forward pass
- Not suitable for very long contexts if local window is large

## Architecture Comparison

### Plain ObservedAttention
```
[Score by attention] [Score by attention] [Score by attention]
├── Oldest tokens ──┼─── Middle tokens ───┼── Recent tokens ──┤
     (may prune)           (may prune)           (may prune)
```

### H2O (This Implementation)
```
[Score by attention] [Score by attention] [Always keep: score=∞]
├── Oldest tokens ──┼─── Middle tokens ───┼── Recent tokens ──┤
     (may prune)           (may prune)       (never prune)
```

## Files Modified/Created

1. **Created**: `/storage/home/manufay/kvpress/kvpress/presses/h2o_press.py`
   - H2OPress class implementation
   - Extends ScorerPress
   - Combines local window + observed attention scoring

2. **Modified**: `/storage/home/manufay/kvpress/kvzap/compare_compression_methods.py`
   - Added H2OPress import
   - Added `create_h2o_press()` factory function
   - Added `h2o_local_window_size` parameter to main function
   - Updated worker to handle H2O with eager attention
   - Added Phase 4 evaluation for ObservedAttention and H2O
   - Updated documentation and usage examples

## Testing

### Quick Test (Single-GPU)
```bash
python compare_compression_methods.py \
  --data_path /path/to/data.jsonl \
  --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
  --methods baseline h2o \
  --max_samples 10 \
  --chunk_size 128
```

### Full Comparison (Multi-GPU)
```bash
python compare_compression_methods.py \
  --data_path /path/to/data.jsonl \
  --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
  --include_observed_attention \
  --num_gpus -1 \
  --max_samples 100 \
  --chunk_size 1
```

## Expected Results

H2O should show:
- **Better PPL than ObservedAttention** (local window preservation helps)
- **Worse PPL than baseline** (compression always degrades quality)
- **Comparable to StreamingLLM** (both use local windows, but H2O is smarter about older tokens)
- **Trade-off with KVzap** (H2O is more accurate but slower; KVzap is faster but approximate)

## Next Steps

If you want to experiment further:
1. **Tune `local_window_size`**: Try 256, 512, 1024 to find sweet spot
2. **Compression sweep**: Test multiple `compression_ratio` values (0.3, 0.4, 0.5, 0.6)
3. **Compare with SnapKV**: Add SnapKV to the comparison (similar idea to H2O)
4. **Optimize for speed**: Consider implementing kernel-level optimizations for scoring

## References

- H2O Paper: https://arxiv.org/abs/2306.14048
- KVpress Documentation: https://github.com/IsaacRe/kvpress
- ObservedAttentionPress: `/storage/home/manufay/kvpress/kvpress/presses/observed_attention_press.py`
