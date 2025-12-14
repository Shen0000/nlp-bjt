#!/bin/bash
#SBATCH --job-name=detectgpt-test
#SBATCH --partition=gpu-a5000-q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=logs/detectgpt-test-%j.out
#SBATCH --error=logs/detectgpt-test-%j.err

# Quick test script to verify DetectGPT integration works on HPC
# Uses Mistral 7B with reduced parameters for fast validation

set -e

echo "========================================"
echo "DetectGPT Test Run on HPC - Mistral 7B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"
echo ""

# -------------------------------
# Module setup
# -------------------------------
module purge
module load cuda11.8/toolkit

# -------------------------------
# Initialize conda
# -------------------------------
source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate base

# -------------------------------
# Activate DetectGPT environment
# -------------------------------
DETECT_GPT_DIR="/home/common/nlp-bjt/baseline/detect-gpt"
cd "$DETECT_GPT_DIR"

if [ -d "${DETECT_GPT_DIR}/env" ]; then
    echo "Activating virtual environment..."
    source "${DETECT_GPT_DIR}/env/bin/activate"
fi

# -------------------------------
# Create logs directory
# -------------------------------
mkdir -p logs

# -------------------------------
# Set paths
# -------------------------------
DATA_DIR="/home/common/nlp-bjt/news_data/data"
CACHE_DIR="/home/common/nlp-bjt/baseline/detect-gpt/.cache"

echo "Data directory: ${DATA_DIR}"
echo "Cache directory: ${CACHE_DIR}"
echo ""

# -------------------------------
# Run test with minimal parameters
# -------------------------------
echo "Starting DetectGPT test run..."
echo "This test uses 50 samples with t5-base for quick validation"
echo ""

python run.py \
    --output_name "test_mistral_7b" \
    --base_model_name "mistralai/Mistral-7B-v0.1" \
    --mask_filling_model_name "t5-base" \
    --n_perturbation_list "10" \
    --n_samples 50 \
    --pct_words_masked 0.3 \
    --span_length 2 \
    --batch_size 5 \
    --dataset xsum \
    --paired_data_path "${DATA_DIR}/mistral_7B_train_filtered.jsonl" \
    --cache_dir "${CACHE_DIR}" \
    --skip_baselines \
    --base_half

echo ""
echo "Results are in: results/test_mistral_7b/"
echo ""
echo "If this test ran without errors, submit the full evaluation with:"
echo "  sbatch slurm_run_news_models.sh"
