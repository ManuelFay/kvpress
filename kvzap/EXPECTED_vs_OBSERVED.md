# ExpectedAttention vs ObservedAttention

## TL;DR

- **ExpectedAttention**: Predicts which KV pairs will be important **in the future**
- **ObservedAttention**: Measures which KV pairs were actually important **in the past**

## Detailed Comparison

### ExpectedAttention (Predictive / Forward-looking)

**What it does:**
Statistically predicts how much attention future query tokens will pay to current key-value pairs.

**How it works:**
1. Analyzes the **current distribution of queries** (mean and covariance)
2. Applies **RoPE rotation** to predict future query patterns
3. Computes **expected attention** that future tokens will pay to current keys:
   ```
   E(Attention) = exp(K @ mean_query^T / sqrt(d) + 1/2 * K @ cov_query @ K^T / d)
   ```
4. Optionally rescales by value norms: `(score + ε) * ||V||₂`

**Key parameters:**
- `n_future_positions=512`: How many future positions to predict
- `use_covariance=True`: Include variance in prediction (more accurate but slower)
- `use_vnorm=True`: Weight by magnitude of value vectors
- `n_sink=4`: Always keep first 4 tokens

**Advantages:**
- ✓ Works with **flash attention** (doesn't need actual attention weights)
- ✓ Fast computation (no need for eager mode)
- ✓ Theoretically grounded (statistical prediction)
- ✓ Includes sink token protection

**Disadvantages:**
- ✗ Prediction may not match actual future attention patterns
- ✗ Assumes attention patterns follow the statistical model

**Performance:**
- Usually achieves **excellent quality** with moderate compression
- Expected Dens(comp): 15-25%
- Expected ΔNLL: +0.5% to +1.5%

---

### ObservedAttention (Reactive / Backward-looking)

**What it does:**
Tracks which KV pairs **actually received attention** during the forward pass, and keeps the ones that were most attended to.

**How it works:**
1. During forward pass, collects **actual attention weights** from the model
2. For each KV pair, computes **average attention** it received:
   ```python
   scores = attentions.sum(2)  # Sum attention from all queries
   scores = scores / n_tokens_in_sum  # Average over tokens
   ```
3. Higher score = more attended to = more important

**Key parameters:**
- `compression_ratio`: Only parameter (or use with DMSPress threshold)
- **No sink tokens** (doesn't have n_sink parameter)

**Advantages:**
- ✓ Uses **actual observed data** (not predictions)
- ✓ Directly measures what the model paid attention to
- ✓ Often achieves the **best quality** among all methods

**Disadvantages:**
- ✗ Requires `attn_implementation="eager"` (much slower)
- ✗ Cannot use flash attention (memory intensive)
- ✗ Past attention may not predict future importance
- ✗ No built-in sink token protection

**Performance:**
- Usually achieves **best quality** but at high computational cost
- Expected Dens(comp): 20-30%
- Expected ΔNLL: +0.3% to +1.0%

---

## Side-by-Side Example

Suppose we're processing this sequence:
```
[CLS] The cat sat on the mat . What did the cat do ?
```

### Current state (just processed "cat do"):
- Need to decide which earlier KV pairs to keep

### ExpectedAttention approach:
1. Looks at the **distribution of recent queries** (from "What", "did", "the", "cat", "do")
2. Predicts: "Future queries will likely attend to nouns and question words"
3. Computes **expected attention** for each past KV:
   - "The" (first): High score (sink token)
   - "cat": High score (noun, likely to be attended to)
   - "sat": Medium score (verb, less likely)
   - "on": Low score (preposition)
   - "the": Low score (article)
   - "mat": Medium score (noun)
   - "What": High score (question word)
4. Keeps tokens with highest expected future attention

### ObservedAttention approach:
1. Looks at **actual attention weights** from "What did the cat do"
2. Measures which past tokens were **actually attended to**:
   - "The" (first): Received attention from most tokens (sink phenomenon)
   - "cat": Received high attention from "do" (0.42)
   - "sat": Received low attention (0.08)
   - "on": Received low attention (0.05)
   - "the": Received low attention (0.03)
   - "mat": Received medium attention (0.15)
   - "What": Received high attention from "cat" and "do" (0.38)
3. Keeps tokens that received the most attention in the past

---

## When to Use Each

### Use ExpectedAttention when:
- ✓ You want **fast inference** with flash attention
- ✓ You need **reasonable quality** with good speed
- ✓ You want **sink token protection** built-in
- ✓ You're processing **long sequences** (>4K tokens)
- ✓ You trust statistical prediction of attention patterns

### Use ObservedAttention when:
- ✓ You prioritize **maximum quality** over speed
- ✓ You can afford **eager attention mode** (slower, more memory)
- ✓ You want to use **actual observed data** from the model
- ✓ You're doing **research** or **offline evaluation**
- ✓ You're processing **shorter sequences** (<2K tokens)

---

## Mathematical Comparison

### ExpectedAttention Score Formula
```
For each key K_i:
  μ = mean(Q_past)          # Mean of past queries (before RoPE)
  Σ = cov(Q_past)           # Covariance of past queries
  R = RoPE(future)          # Average RoPE rotation for future positions

  μ' = R @ μ                # Rotate mean to future
  Σ' = R @ Σ @ R^T          # Rotate covariance to future

  score_i = exp((K_i @ μ') / √d + 0.5 * (K_i @ Σ' @ K_i^T) / d)

  if use_vnorm:
    score_i = (score_i + ε) * ||V_i||₂
```

### ObservedAttention Score Formula
```
For each key K_i:
  # A[t, i] = attention weight from query t to key i
  score_i = (1/T) * Σ_{t=1}^{T} A[t, i]

  # Where T is the number of queries that have seen K_i
```

---

## Relationship to H2O

**H2O** is an extension of ObservedAttention:
- Uses **observed attention scores** (like ObservedAttention)
- Adds **local window protection**: Last N tokens always kept (score=∞)
- In this repo:
  - `ObservedAttention`: Pure observed attention, no local window
  - `H2OPress`: ObservedAttention + local_window_size=512

---

## Empirical Comparison (Expected Results)

On a typical long-document task with threshold tuned for ~70% overall density:

| Metric | ExpectedAttention | ObservedAttention |
|--------|-------------------|-------------------|
| **Quality (ΔNLL)** | +0.8% to +1.2% | +0.5% to +0.9% ← Better |
| **Speed** | Fast (flash attn) ← Better | Slow (eager mode) |
| **Memory** | Low ← Better | High |
| **Dens(comp)** | 15-20% | 20-25% ← Higher |
| **Has sinks?** | Yes ← Better | No |
| **Requires eager?** | No ← Better | Yes |

**Conclusion**:
- **ObservedAttention** usually achieves slightly **better quality** (lower NLL)
- **ExpectedAttention** is **much faster** and works with flash attention
- For production use, **ExpectedAttention** is typically preferred
- For research/maximum quality, **ObservedAttention** may be worth the cost

---

## Code Comparison

### ExpectedAttention
```python
# Works with flash attention (fast)
press = ExpectedAttentionPress(
    compression_ratio=0.0,
    n_future_positions=512,  # Predict 512 tokens ahead
    n_sink=4,                # Protect first 4 tokens
    use_covariance=True,     # Use full statistics
    use_vnorm=True,          # Weight by value norms
)
press = DMSPress(press, threshold=0.3, sliding_window_size=128)

# No special model loading needed
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### ObservedAttention
```python
# Requires eager attention (slow)
press = ObservedAttentionPress(compression_ratio=0.0)
press = DMSPress(press, threshold=0.0005, sliding_window_size=128)

# Must load with eager attention mode
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager"  # Required!
)
```

---

## Summary Table

| Aspect | ExpectedAttention | ObservedAttention |
|--------|-------------------|-------------------|
| **Philosophy** | Predict future importance | Measure past importance |
| **Data source** | Query statistics | Actual attention weights |
| **Computation** | Statistical prediction | Direct measurement |
| **Speed** | Fast | Slow |
| **Attn mode** | Flash (fast) | Eager (slow) |
| **Quality** | Very good | Excellent |
| **Sink tokens** | Yes (n_sink=4) | No |
| **Use case** | Production | Research |
