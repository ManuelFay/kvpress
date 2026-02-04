# Specs: KV Cache Compression Method Comparison

## Overview
Create a script that evaluates and compares different KV cache compression methods on the same validation dataset to understand their performance-compression tradeoffs.

## Goal
Compare the following compression methods on perplexity evaluation:
1. **Baseline** - No compression
2. **KVzap** - Fast approximation with learned surrogate models (current implementation)
3. **Random** - Random token eviction (baseline)
4. **StreamingLLM** - Sliding window + sink tokens
5. **ExpectedAttention** - Statistical prediction of future attention
6. **ObservedAttention** - Historical attention weights

## Input Parameters

### Model Configuration
- `model_name`: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
- `data_path`: str - Path to validation JSONL dataset
- `max_length`: int = 8192 - Maximum sequence length
- `max_samples`: int = None - Limit dataset size for testing

### Evaluation Configuration
- `chunk_size`: int = 1 - Chunk size for evaluation (1=exact, >1=approximate)
- `device`: str = "cuda:0" - Device to use
- `output_dir`: str = "./comparison_results" - Where to save results
- `num_gpus`: int = None - Multi-GPU support (optional for future)

### Compression Configuration
- `compression_ratio`: float = 0.5 - Target compression (fraction to remove)
  - Used by: Random, StreamingLLM, ExpectedAttention, ObservedAttention
- `threshold`: float = -7.0 - Threshold for score-based eviction
  - Used by: KVzap (via DMSPress)
- `sliding_window_size`: int = 128 - Recent tokens to protect from eviction
  - Used by: KVzap (via DMSPress)

### Method-Specific Parameters

**StreamingLLM:**
- `n_sink`: int = 4 - Number of initial tokens to always preserve

**ExpectedAttention:**
- `n_future_positions`: int = 512 - Future positions to consider
- `n_sink`: int = 4 - Sink tokens
- `use_covariance`: bool = True - Include covariance in computation
- `use_vnorm`: bool = True - Rescale by value norms

**ObservedAttention:**
- Requires: `attn_implementation="eager"` in model config

## Output Format

### Per-Method Results
For each compression method, compute:
```json
{
  "method_name": "KVzap",
  "perplexity": 12.34,
  "avg_nll": 2.51,
  "total_tokens": 1234567,
  "compression_stats": {
    "avg_compression_ratio": 0.45,
    "sparsity_percentage": 45.0,
    "total_keys_original": 1000000,
    "total_keys_kept": 550000,
    "total_keys_discarded": 450000
  }
}
```

### Comparison Summary
```json
{
  "baseline_perplexity": 12.00,
  "results": [
    {
      "method": "KVzap",
      "perplexity": 12.34,
      "ppl_delta": 0.34,
      "ppl_delta_pct": 2.83,
      "compression_ratio": 0.45
    },
    ...
  ]
}
```

## Implementation Structure

### 1. Core Evaluation Function
```python
def evaluate_with_press(
    model,
    tokenizer,
    texts,
    press,
    method_name,
    max_length,
    chunk_size,
    device
) -> dict
```
- Wraps `calculate_perplexity_chunked` from existing script
- Returns standardized results dictionary
- Handles compression statistics extraction

### 2. Press Factory Functions
```python
def create_random_press(compression_ratio: float, seed: int = 42) -> RandomPress
def create_streaming_llm_press(compression_ratio: float, n_sink: int = 4) -> StreamingLLMPress
def create_expected_attention_press(compression_ratio: float, **kwargs) -> ExpectedAttentionPress
def create_observed_attention_press(compression_ratio: float) -> ObservedAttentionPress
def create_kvzap_press(threshold: float, kvzap_model_type: str, kvzap_scorer_model: str) -> DMSPress
```

### 3. Main Comparison Function
```python
def compare_compression_methods(
    data_path: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    compression_ratio: float = 0.5,
    threshold: float = -7.0,
    kvzap_scorer_model: str = "nvidia/KVzap-mlp-Llama-3.1-8B-Instruct",
    max_samples: int = None,
    chunk_size: int = 1,
    device: str = "cuda:0",
    output_dir: str = "./comparison_results",
    methods: list[str] = None,  # If None, run all methods
) -> dict
```

### 4. Visualization/Reporting (Optional, Future)
- Generate comparison plots
- Create markdown report
- Export to CSV for external analysis

## Execution Flow

1. **Load model and dataset** (once, reused for all methods)
2. **Evaluate baseline** (no compression)
3. **For each compression method:**
   - Create press with appropriate configuration
   - Evaluate perplexity with chunked processing
   - Extract compression statistics
   - Store results
4. **Compute comparisons** (deltas relative to baseline)
5. **Save results** to JSON file
6. **Print summary table** to stdout

## Key Design Decisions

### 1. Reuse vs. Separate Implementation
- **Reuse** `calculate_perplexity_chunked` from existing script
- Import CustomKVzapPress and helper functions
- Avoid code duplication

### 2. Fair Comparison Strategy
- All methods use **same chunk_size** (1 for exact, >1 for approximate)
- All methods see **same dataset** in same order
- Baseline evaluated first to establish reference

### 3. Compression Ratio Equivalence
For KVzap (threshold-based):
- KVzap uses threshold, others use compression_ratio
- Report **actual** compression ratio achieved by KVzap
- Cannot directly match target compression_ratio

Strategy: Run both paradigms separately
- Fixed ratio methods: Random, StreamingLLM, ExpectedAttention, ObservedAttention
- Threshold-based: KVzap with specified threshold

### 4. ObservedAttention Special Handling
- Requires `attn_implementation="eager"`
- Load model separately with this config
- Skip if user doesn't want to run it (slower)

## File Structure
```
/storage/home/manufay/kvpress/kvzap/
├── evaluate_ppl_chunked.py          # Existing chunked evaluation
├── compare_compression_methods.py    # New comparison script
├── SPECS_compare_methods.md         # This document
└── comparison_results/              # Output directory
    ├── comparison_summary.json
    └── detailed_results.json
```

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Import required modules and existing functions
- [ ] Create press factory functions
- [ ] Create `evaluate_with_press` wrapper function
- [ ] Test with baseline (no press)

### Phase 2: Fixed-Ratio Methods
- [ ] Implement Random press evaluation
- [ ] Implement StreamingLLM press evaluation
- [ ] Implement ExpectedAttention press evaluation
- [ ] Test all three methods

### Phase 3: Threshold-Based Method
- [ ] Implement KVzap evaluation (using existing CustomKVzapPress)
- [ ] Handle DMSPress wrapping correctly

### Phase 4: Optional - ObservedAttention
- [ ] Load model with eager attention
- [ ] Implement ObservedAttention evaluation
- [ ] Add flag to skip if too slow

### Phase 5: Results & Reporting
- [ ] Aggregate all results
- [ ] Compute comparison statistics
- [ ] Save to JSON
- [ ] Print formatted table to stdout

### Phase 6: CLI Integration
- [ ] Add fire CLI
- [ ] Add parameter validation
- [ ] Add progress reporting

## Usage Examples

### Basic comparison (all methods except ObservedAttention)
```bash
python compare_compression_methods.py \
  --data_path /path/to/govreport.val.jsonl \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
  --compression_ratio 0.5 \
  --threshold -7.0 \
  --chunk_size 1 \
  --max_samples 100
```

### Specific methods only
```bash
python compare_compression_methods.py \
  --data_path /path/to/data.jsonl \
  --methods random,streaming_llm,kvzap \
  --compression_ratio 0.5 \
  --chunk_size 128  # Faster approximate evaluation
```

### Include ObservedAttention (slower)
```bash
python compare_compression_methods.py \
  --data_path /path/to/data.jsonl \
  --methods all \
  --include_observed_attention \
  --chunk_size 1
```

## Success Criteria
- Script successfully evaluates all methods on same dataset
- Results are reproducible (fixed random seeds)
- Compression ratios are reported accurately
- Output includes both raw metrics and deltas vs baseline
- Runtime is reasonable for typical use cases
