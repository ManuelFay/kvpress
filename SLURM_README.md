# Running KVPress Comparison on SLURM

Simple single-file SLURM script for evaluating KV compression methods on the GovReport validation set.

## Usage

```bash
# Submit the job
sbatch run_govreport.sh

# Monitor
squeue -u $USER
tail -f kvpress_*.out

# Cancel if needed
scancel <JOB_ID>
```

## Configuration

The script is configured to:
- **Data**: `/checkpoint/amaia/scale/manufay/ppl_data/govreport.val.jsonl`
- **Model**: Llama-3.1-8B-Instruct
- **GPUs**: 8 (all available with `--num_gpus -1`)
- **Time**: 11.5 hours
- **CPUs**: 16
- **Samples**: 100
- **Chunk size**: 128 (fast approximate mode)
- **Methods**: random, streaming_llm, expected_attention, kvzap
- **Partition**: learn
- **Account**: fair_amaia_cw_codegen
- **QoS**: scale

## Customization

Edit `run_govreport.sh` to change:

### Number of samples
```bash
--max_samples 500    # Increase for more thorough evaluation
```

### Chunk size (speed vs accuracy)
```bash
--chunk_size 1       # Exact NLL (very slow)
--chunk_size 128     # Fast approximate (100x faster)
```

### Methods to compare
```bash
--methods random streaming_llm kvzap    # Fewer methods = faster
```

### Add observed attention methods
```bash
--methods random streaming_llm expected_attention kvzap observed_attention h2o \
--include_observed_attention
```

### Custom thresholds
```bash
--thresholds '{"kvzap": -7.5, "expected_attention": 0.1}'
```
**Note**: Pass as a JSON string with single quotes around it.

### GPU count
```bash
--gpus-per-node=4    # Use fewer GPUs
--num_gpus 4         # Limit to 4 GPUs even if more available
```

### Partition (update for your cluster)
```bash
#SBATCH --partition=learn     # Current partition
#SBATCH --account=fair_amaia_cw_codegen
#SBATCH --qos=scale
```

## Output

Results are saved to `./kvzap/govreport_results_<JOB_ID>/`:
- `comparison_summary.json` - Summary table
- `detailed_results.json` - Full results
- `*_compression_heatmap.png` - Visualizations

## Expected Runtime

With 8 GPUs and 100 samples:
- **Actual time**: ~15-25 minutes
- **Speedup**: ~7x vs single GPU
- **Total forward passes**: ~100 samples × ~800 tokens/128 chunk_size = ~625 passes
- **Per GPU**: ~78 passes

## Troubleshooting

### OOM (Out of Memory)
Reduce samples:
```bash
--max_samples 50
```

### Job timeout
Increase time limit:
```bash
#SBATCH --time=16:00:00
```
