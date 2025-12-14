#!/bin/bash
#SBATCH --job-name=detectgpt-mistral7b
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/detectgpt-mistral7b-%j.out
#SBATCH --error=logs/detectgpt-mistral7b-%j.err

# DetectGPT evaluation for Mistral 7B news dataset
# Runs on A100 GPU (40-80GB) - more memory for comfortable execution

set -e

echo "========================================"
echo "DetectGPT Evaluation - Mistral 7B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

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
# Setup environment
# -------------------------------
DETECT_GPT_DIR="/home/common/nlp-bjt/baseline/detect-gpt"
cd "$DETECT_GPT_DIR"

if [ -d "${DETECT_GPT_DIR}/env" ]; then
    source "${DETECT_GPT_DIR}/env/bin/activate"
fi

mkdir -p logs

# -------------------------------
# Run DetectGPT
# -------------------------------
DATA_DIR="/home/common/nlp-bjt/news_data/data/sample"
CACHE_DIR="/home/common/nlp-bjt/baseline/detect-gpt/.cache"

python run.py \
    --output_name "news_mistral_7b" \
    --base_model_name "mistralai/Mistral-7B-v0.1" \
    --mask_filling_model_name "t5-3b" \
    --n_perturbation_list "10,50,100" \
    --n_samples 200 \
    --pct_words_masked 0.15 \
    --span_length 2 \
    --batch_size 15 \
    --dataset xsum \
    --paired_data_path "${DATA_DIR}/mistral_7B_sampled.jsonl" \
    --cache_dir "${CACHE_DIR}" \
    --skip_baselines \
    --base_half

echo "Mistral 7B evaluation completed"
