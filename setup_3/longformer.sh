#! /bin/sh

#SBATCH --job-name=llama-7B
#SBATCH --partition=gpu-a5000-q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=80G
#SBATCH --output=llama-7B-%j.out
#SBATCH --error=llama-7B-%j.err

MODEL="llama_7B"

export OMP_NUM_THREADS=1

# -------------------------------
# Module setup
# -------------------------------
module purge
module load cuda11.8/toolkit

# -------------------------------
# Initialize conda
# -------------------------------
source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh

# -------------------------------
# Go to base working directory
# -------------------------------
cd /home/common/nlp-bjt/setup_3

# --------------------------
# Activate conda environment
# --------------------------
conda activate /home/common/nlp-bjt/setup_3/conda-setup_3

# ----------------
# Train Longformer
# ----------------
# python 5_train_longformer.py $MODEL --motif
python 5_train_longformer.py $MODEL

# -------------------
# Evaluate Longformer
# -------------------
# python 6_evaluate_longformer.py $MODEL --motif
python 6_evaluate_longformer.py $MODEL