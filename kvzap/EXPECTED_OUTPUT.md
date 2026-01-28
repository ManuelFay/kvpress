# Expected Output Format

## Comparison Table (New Format)

When you run the comparison script, you'll now see:

```
================================================================================
COMPARISON SUMMARY
================================================================================

Baseline NLL: 2.130456
Baseline PPL: 8.4213

Method                    NLL         ΔNLL      ΔNLL %    Dens(all)    Dens(comp)
--------------------------------------------------------------------------------------------
Note: Dens(all) = Overall density (all tokens)
      Dens(comp) = Density in compressible region (excluding window, may include sinks)
--------------------------------------------------------------------------------------------
Baseline              2.130456    +0.000000      +0.00%      100.00%      100.00%
StreamingLLM          2.145321    +0.014865      +0.70%       65.23%        0.12%
Random(threshold=1.0) 2.145321    +0.014865      +0.70%       65.23%        0.12%
ExpectedAttention     2.152134    +0.021678      +1.02%       72.15%       18.45%
KVzap                 2.156892    +0.026436      +1.24%       68.92%       12.33%
Random                2.189234    +0.058778      +2.76%       58.12%        5.67%
```

### Interpreting the Results

#### 1. **Validation Check**
The first thing to verify: **StreamingLLM and Random(threshold=1.0) should be IDENTICAL**
- Same NLL, same ΔNLL, same densities
- If they differ, there's a bug in sink token implementation

#### 2. **NLL Delta**
- Lower is better
- Shows the quality degradation from compression
- Example: ExpectedAttention has +1.02% NLL increase (very good!)

#### 3. **Dens(all) - Overall Density**
- Percentage of all keys kept in the cache
- Includes: sinks (first 4) + window (last 128) + compressible region
- Example: ExpectedAttention keeps 72.15% of all keys

#### 4. **Dens(comp) - Compressible Region Density**
- Percentage of keys kept in the region OUTSIDE the sliding window
- This is the "true" compression aggressiveness
- Example: ExpectedAttention keeps 18.45% in the compressible region

**Note**: StreamingLLM and Random(t=1.0) have ~0% compressible density because they evict EVERYTHING in the compressible region (keep only sinks + window).

## Per-Method Output

During evaluation, you'll see detailed stats for each method:

```
================================================================================
Evaluating: KVzap
================================================================================

KVzap Results:
  Perplexity: 8.6234
  Avg NLL: 2.156892
  Total tokens: 125,432

Compression Statistics:
  Overall Density (w/ window): 68.92%
  Compressible Region Density (w/o window): 12.33%
  Keys Kept: 2,145,678 / 3,112,456
  Sliding Window Size: 128
```

## Real-World Example

Suppose on a 1000-token sequence:

### StreamingLLM:
- **Protected**: 4 sinks + 128 window = 132 tokens
- **Compressible region**: 1000 - 132 = 868 tokens
- **Kept in compressible**: ~0 tokens (all evicted)
- **Total kept**: 132 tokens
- **Dens(all)**: 132/1000 = 13.2%
- **Dens(comp)**: 0/868 = 0%

### ExpectedAttention (threshold=0.3):
- **Protected**: 4 sinks (always) + 128 window (always) = 132 tokens
- **Compressible region**: 1000 - 128 = 872 tokens (includes sinks in numerator/denominator)
- **Kept in compressible**: ~160 tokens (18.45% of 872)
- **Total kept**: 132 + 160 = 292 tokens
- **Dens(all)**: 292/1000 = 29.2%
- **Dens(comp)**: ~18.45%

### KVzap (threshold=-6.0):
- **Protected**: 4 sinks + 128 window = 132 tokens
- **Compressible region**: 872 tokens
- **Kept in compressible**: ~107 tokens (12.33% of 872)
- **Total kept**: 132 + 107 = 239 tokens
- **Dens(all)**: 239/1000 = 23.9%
- **Dens(comp)**: ~12.33%

## Method Comparison Summary

| Method | Expected Dens(comp) | Quality (ΔNLL%) | Use Case |
|--------|---------------------|-----------------|----------|
| StreamingLLM | ~0% | Good baseline | Fixed structure, predictable memory |
| Random(t=1.0) | ~0% | Same as StreamingLLM | **Validation only** |
| ExpectedAttention | 15-25% | Excellent | Best quality, moderate memory savings |
| KVzap | 10-15% | Very good | Fast, good quality, better compression |
| Random | 5-10% | Fair | Baseline for random eviction |
| ObservedAttention | 20-30% | Excellent | Best quality, but slow (eager mode) |
| H2O | 25-35% | Excellent | High quality, but uses 512 window |

## Important Notes

### Sink Tokens
Methods with sinks (first 4 positions never evicted):
- ✓ RandomPress
- ✓ StreamingLLM
- ✓ ExpectedAttention
- ✓ KVzap
- ✗ ObservedAttention
- ✗ H2O

### Window Sizes
- **Most methods**: 128 tokens (DMSPress default)
- **H2O**: 512 tokens (H2O's own local_window) + 128 (DMSPress)
  - Effective window is 512
  - But metrics only exclude 128 from compressible region calculation
  - This makes H2O's Dens(comp) not directly comparable

### Validation Success Criteria
When running the validation test with `streaming_llm` and `random_validation`:

**Pass**: NLL values match to 6+ decimal places
```
StreamingLLM          2.145321    +0.014865      +0.70%       65.23%        0.12%
Random(threshold=1.0) 2.145321    +0.014865      +0.70%       65.23%        0.12%
```

**Fail**: NLL values differ
```
StreamingLLM          2.145321    ...
Random(threshold=1.0) 2.148567    ...  ← Different! Bug in sink implementation
```

## Files Generated

After running the comparison, you'll find:

1. **`comparison_results/detailed_results.json`**: Full results for all methods
2. **`comparison_results/comparison_summary.json`**: Summary statistics
3. **`DENSITY_METRICS_EXPLAINED.md`**: Detailed explanation of metrics
4. **`CHANGES_SUMMARY.md`**: Summary of code changes

## Running the Validation

```bash
# Quick validation test (should take ~1 minute)
python kvzap/compare_compression_methods.py \
    --data_path your_data.jsonl \
    --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
    --methods streaming_llm random_validation \
    --max_samples 5 \
    --chunk_size 128

# Expected output:
# StreamingLLM and Random(threshold=1.0) should have IDENTICAL NLL
```
