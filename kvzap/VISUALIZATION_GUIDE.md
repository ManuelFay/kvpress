# Compression Visualization Feature

## Overview

The comparison script now automatically generates heatmap visualizations showing which tokens are kept vs. evicted for each compression method on the first sample.

## What Gets Visualized

For each compression method, the script creates a PNG file showing:
1. **Binary compression decisions** (first 1024 tokens)
   - Green = Token kept (score ≥ threshold)
   - Red = Token evicted (score < threshold)
2. **Protected regions**
   - Blue = Protected tokens (sinks + sliding window)
   - White = Compressible region

## How It Works

### Score Capture
After evaluating the first sample with each method, the script:
1. Captures the scores from `DMSPress.scores_buffer`
2. Averages scores across all layers and heads
3. Applies the threshold to create binary decisions
4. Marks protected regions (sinks + sliding window)

### Visualization
- **Top panel**: Shows kept (green) vs evicted (red) tokens
- **Bottom panel**: Shows protected (blue) vs compressible (white) regions
- **Vertical lines**: Mark boundaries of protected regions
  - Blue dashed line: End of sink tokens
  - Orange dashed line: Start of sliding window

### Statistics Printed
For each visualization:
```
Tokens visualized: 1024
Overall density: 65.23% (668/1024 kept)
Protected tokens: 132 (sinks=4, window=128)
Compressible region density: 0.12% (1/896 kept)
```

## Output Files

Visualizations are saved to `{output_dir}/` with filenames like:
- `StreamingLLM_compression_heatmap.png`
- `Random_threshold_1.0__compression_heatmap.png`
- `ExpectedAttention_compression_heatmap.png`
- `KVzap_compression_heatmap.png`

## Usage

### Enable (Default)
```bash
python compare_compression_methods.py \
    --data_path data.jsonl \
    --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
    --methods streaming_llm random_validation \
    --max_samples 10 \
    --chunk_size 128
# Visualizations will be generated automatically
```

### Disable
```python
# In Python
compare_compression_methods(
    data_path="data.jsonl",
    visualize_first_sample=False  # Disable visualizations
)
```

## Validation with Visualizations

The visualizations are particularly useful for validating implementation:

### Example: StreamingLLM vs Random(threshold=1.0)
These two methods should produce **identical visualizations**:

**Expected pattern:**
```
Position:    0   4                                        896            1024
            [====][................................................][============]
            Sinks           Compressible (all red)            Window (all green)
```

- First 4 tokens (sinks): Green
- Middle 892 tokens: Red (all evicted)
- Last 128 tokens (window): Green

If the patterns differ, there's a bug in sink token implementation!

### Example: ExpectedAttention
**Expected pattern:**
```
Position:    0   4                                        896            1024
            [====][................sparse green..........][============]
            Sinks      Compressible (some green)            Window
```

- First 4 tokens: Green (sinks protected)
- Middle region: Scattered green tokens (~15-20% density)
- Last 128 tokens: All green (window protected)

### Example: Random
**Expected pattern:**
```
Position:    0   4                                        896            1024
            [====][........uniform random green.........][============]
            Sinks      Compressible (~50% green)            Window
```

- First 4 tokens: Green (sinks)
- Middle region: Random distribution of green/red (~threshold% green)
- Last 128 tokens: All green (window)

## Interpreting the Visualizations

### Checking Sink Token Implementation
1. **Look at positions 0-3** (first 4 tokens)
2. Should be **green** for methods with `n_sink=4`:
   - RandomPress ✓
   - StreamingLLM ✓
   - ExpectedAttention ✓
   - KVzap ✓
3. Should be **red or green** (based on scores) for methods without sinks:
   - ObservedAttention
   - H2O (uses local window instead)

### Checking Sliding Window
1. **Look at the last 128 positions**
2. Should be **all green** (always protected by DMSPress)
3. Orange vertical line should mark the boundary

### Checking Compressible Region
1. **Look at the middle region** (between sinks and window)
2. Pattern should match method behavior:
   - **StreamingLLM / Random(t=1.0)**: All red (aggressive eviction)
   - **ExpectedAttention / KVzap**: Sparse green (selective)
   - **Random**: Uniform distribution (threshold-dependent)
   - **ObservedAttention**: Based on actual attention patterns

## Technical Details

### Score Aggregation
Scores are averaged across:
- All layers (typically 32 layers for Llama-3.1-8B)
- All heads (typically 8 KV heads for Llama-3.1-8B)

This gives a single score per token position.

### Protected Region Calculation
```python
# Sink tokens (absolute positions)
protected[0:n_sink] = 1

# Sliding window (relative to sequence end)
if seq_len > sliding_window_size:
    protected[-sliding_window_size:] = 1
```

### Compressible Region
```python
compressible_region = total_tokens - protected_tokens
density_compressible = kept_compressible / compressible_region
```

## Performance Impact

- **Negligible**: Only processes first sample once
- **Time**: Adds ~1-2 seconds per method
- **Memory**: PNG files are ~200-500 KB each

## Troubleshooting

### No visualization generated
**Cause**: Multi-GPU mode is enabled
**Solution**: Use single-GPU mode for visualizations
```bash
# Don't specify --num_gpus, or use --num_gpus 1
python compare_compression_methods.py --data_path data.jsonl ...
```

### "No scores to visualize"
**Cause**: Method is Baseline (no compression)
**Expected**: Baseline doesn't generate scores, so no visualization

### All green or all red
**Possible causes**:
1. Threshold too low/high
2. Scores not being captured correctly
3. Bug in scoring implementation

**Debug**: Check printed statistics for density values

### Missing sink protection
**Possible causes**:
1. Method doesn't implement sink tokens
2. `n_sink` parameter not passed correctly
3. Sink token logic bug

**Debug**: Check first 4 positions should be green for methods with sinks

## Example Output

Running comparison with 2 methods:
```
Evaluating: StreamingLLM
================================================================================
...
Generating compression visualization for StreamingLLM...
  Saved visualization to: comparison_results/StreamingLLM_compression_heatmap.png
    Tokens visualized: 1024
    Overall density: 12.89% (132/1024 kept)
    Protected tokens: 132 (sinks=4, window=128)
    Compressible region density: 0.00% (0/892 kept)

Evaluating: Random(threshold=1.0)
================================================================================
...
Generating compression visualization for Random(threshold=1.0)...
  Saved visualization to: comparison_results/Random_threshold_1.0__compression_heatmap.png
    Tokens visualized: 1024
    Overall density: 12.89% (132/1024 kept)
    Protected tokens: 132 (sinks=4, window=128)
    Compressible region density: 0.00% (0/892 kept)
```

Notice: Both methods have **identical statistics** ✓ Validation successful!
