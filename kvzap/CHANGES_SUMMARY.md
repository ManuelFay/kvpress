# Summary of Changes to Comparison Script

## Overview
Modified `compare_compression_methods.py` to report **density** instead of compression ratio, focus on **NLL** instead of PPL, and added a **validation method** to verify sink token implementation.

## Key Changes

### 1. Density Reporting (Instead of Compression Ratio)
**Previous**: Reported "compression ratio" (which was actually sparsity)
**Now**: Reports **density** = 1 - sparsity

Two density metrics are shown:
- **Dens(all)**: Overall density including all tokens (sinks + window + compressible region)
- **Dens(comp)**: Density in the compressible region (excluding sliding window)

**Note**: The compressible density currently excludes the sliding window but may still include sink tokens in both numerator and denominator, since sinks are not separately tracked in `evaluate_ppl_chunked.py`.

### 2. NLL-Focused Comparison Table
**Previous table**:
```
Method               PPL      Δ PPL    Δ PPL %    Comp Ratio
```

**New table**:
```
Method                    NLL         ΔNLL      ΔNLL %    Dens(all)    Dens(comp)
```

- Sorted by NLL (lower is better)
- Shows absolute and relative NLL delta from baseline
- PPL still shown in the baseline summary line

### 3. Validation Method: Random with threshold=1.0
Added `random_validation` method that should produce **identical results to StreamingLLM**.

**Why they should match**:
- **RandomPress(threshold=1.0)**:
  - Sink tokens (first 4): `score=inf > 1.0` → kept
  - Sliding window (last 128): Protected by DMSPress → kept
  - All others: `score ∈ [0, 1) < 1.0` → evicted
  - **Result**: 4 sinks + 128 window = 132 tokens kept

- **StreamingLLM(threshold=0.5)**:
  - Sink tokens (first 4): `score=1.0 > 0.5` → kept
  - Sliding window (last 128): Protected by DMSPress → kept
  - All others: `score=0.0 < 0.5` → evicted
  - **Result**: 4 sinks + 128 window = 132 tokens kept

**Usage**:
```bash
python compare_compression_methods.py \
    --data_path data.jsonl \
    --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
    --methods streaming_llm random_validation \
    --max_samples 10
```

If implementation is correct, these two methods should produce:
- **Identical NLL**
- **Identical density metrics**
- **Same number of keys kept**

### 4. KVZap Model Loading Fix
**Issue**: The `create_kvzap_press()` function wasn't passing `n_sink=4` parameter.

**Fixed**:
```python
# Before
kvzap_press = CustomKVzapPress(
    model_type=kvzap_model_type,
    explicit_model_name=kvzap_scorer_model
)

# After
kvzap_press = CustomKVzapPress(
    model_type=kvzap_model_type,
    explicit_model_name=kvzap_scorer_model,
    n_sink=4  # ✓ Now explicitly passed
)
```

This ensures KVZap properly protects the first 4 sink tokens with `score=inf`.

### 5. Updated Documentation
- Added `DENSITY_METRICS_EXPLAINED.md` explaining what density means for each method
- Updated script docstring with validation method example
- Added inline comments explaining sink token handling

## Code Locations

### Modified Functions
1. **`compare_compression_methods()`** (line 897):
   - Updated comparison results to include `nll_delta_pct`, `density_overall`, `density_compressible`
   - Changed table output to focus on NLL
   - Updated sorting to sort by NLL instead of PPL

2. **`create_kvzap_press()`** (line 306):
   - Added `n_sink` parameter with default value 4
   - Passes `n_sink` to both `CustomKVzapPress` and `KVzapPress`

3. **`evaluate_with_press()`** (line 770):
   - Updated console output to show density instead of compression ratio

### New Configuration
- **DEFAULT_THRESHOLDS** (line 112):
  - Added `"random_validation": 1.0`

- **methods_config** (line 1070):
  - Added random_validation configuration block (line 1110)

## Implications for Each Method

| Method | Sink Tokens | Sliding Window | Compressible Region |
|--------|------------|----------------|---------------------|
| Baseline | None | None | All tokens (100% density) |
| Random | 4 (score=inf) | 128 (DMSPress) | [4, seq_len-128) |
| StreamingLLM | 4 (score=1.0) | 128 (DMSPress) | [4, seq_len-128) |
| Random(t=1.0) | 4 (score=inf) | 128 (DMSPress) | [4, seq_len-128) - **should match StreamingLLM** |
| ExpectedAttention | 4 (score=inf) | 128 (DMSPress) | [4, seq_len-128) |
| KVzap | 4 (score=inf) ✓ | 128 (DMSPress) | [4, seq_len-128) |
| ObservedAttention | None | 128 (DMSPress) | [0, seq_len-128) |
| H2O | None | 512 (H2O) + 128 (DMSPress) | [0, seq_len-512) * |

\* **Important**: H2O has two layers of protection:
- H2O assigns `score=inf` to last 512 tokens
- DMSPress protects last 128 tokens
- Effective window size is 512, not 128
- The "compressible density" metric only excludes 128 tokens, so it's not accurate for H2O

## Testing the Changes

### Validation Test
```bash
# Should show identical results for StreamingLLM and Random(threshold=1.0)
python compare_compression_methods.py \
    --data_path test.jsonl \
    --methods streaming_llm random_validation \
    --max_samples 10 \
    --chunk_size 128
```

Expected output:
```
Method                    NLL         ΔNLL      ΔNLL %    Dens(all)    Dens(comp)
----------------------------------------------------------------------------------------
Baseline              2.130000    +0.000000      +0.00%      100.00%      100.00%
StreamingLLM          2.145000    +0.015000      +0.70%       65.20%        0.05%
Random(threshold=1.0) 2.145000    +0.015000      +0.70%       65.20%        0.05%
```

If the NLL values don't match exactly, there's a bug in the sink token implementation!

### Full Comparison
```bash
# Compare all methods with density reporting
python compare_compression_methods.py \
    --data_path govreport.val.jsonl \
    --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
    --methods random streaming_llm expected_attention kvzap \
    --max_samples 100 \
    --chunk_size 128
```

## Next Steps (Potential Future Work)

1. **Track sink tokens separately in `evaluate_ppl_chunked.py`**:
   - Add `total_keys_in_sinks` tracking
   - Calculate true compressible region excluding BOTH sinks AND window
   - Would provide more accurate "Dens(comp)" metric

2. **Add H2O-specific window size handling**:
   - H2O uses 512-token window, not 128
   - Metrics should reflect the actual protected region size

3. **Add column for "effective compression"**:
   - Show actual memory reduction percentage
   - Account for fixed overhead of sinks + window

4. **Per-method window size reporting**:
   - Show which methods use which window sizes
   - Make it clear that H2O is different
