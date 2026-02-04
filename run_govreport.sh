#!/bin/bash
#SBATCH --job-name=kvpress_govreport
#SBATCH --output=kvpress_%j.out
#SBATCH --error=kvpress_%j.err
#SBATCH --time=11:30:00
#SBATCH --gpus-per-node=8
#SBATCH --partition=learn
#SBATCH --account=fair_amaia_cw_codegen
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=scale

echo "========================================="
echo "KVPress Multi-GPU Comparison - GovReport"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Started: $(date)"
echo ""

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/home/manufay/claude_conda_env

# Set Python path
export PYTHONPATH="/home/manufay/kvpress/kvzap:$PYTHONPATH"

# Force HuggingFace to use cached files only (no network calls)
export HF_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Navigate to working directory
cd /home/manufay/kvpress/kvzap

# Print environment info
echo "Environment:"
echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo "  Working dir: $(pwd)"
echo ""
nvidia-smi
echo ""

# Run comparison on all 8 GPUs
python compare_compression_methods.py \
    --data_path /checkpoint/amaia/scale/manufay/ppl_data/govreport.val.jsonl \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --kvzap_scorer_model nvidia/KVzap-mlp-Llama-3.1-8B-Instruct \
    --max_samples 100 \
    --chunk_size 128 \
    --methods "kvzap_no_sink,h2o" \
    --output_dir ./govreport_results_${SLURM_JOB_ID} \
    --num_gpus -1 \
    --data_loader jsonl_text

exit_code=$?

echo ""
echo "========================================="
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS - Results saved to: ./govreport_results_${SLURM_JOB_ID}"
else
    echo "FAILED - Check error log"
fi
echo "Finished: $(date)"
echo "========================================="

exit $exit_code
