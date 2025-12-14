#! /bin/sh

#SBATCH --job-name=llama-30B
#SBATCH --cpus-per-task=48
#SBATCH --mem=80G
#SBATCH --output=logs/llama-30B-%j.out
#SBATCH --error=logs/llama-30B-%j.err

MODEL="llama_30B"

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

mkdir -p data/${MODEL}/split/
mkdir -p data/${MODEL}/unsplit/
mkdir -p data/motifs/

# ----------------------------------------
# Extract and save motifs from the dataset
# ----------------------------------------
python 1_extract_and_save_motifs.py $MODEL

# ------------------------------------------
# Add ALL motif distributions to the dataset
# ------------------------------------------
python 2_add_motif_dists_to_datasets.py $MODEL

# -----------------
# Select top motifs
# -----------------
python 3_select_motifs.py $MODEL

# -------------------------------------------
# Add SELECTED motif distributions to dataset
# -------------------------------------------
python 2_add_motif_dists_to_datasets.py $MODEL --selected

# --------------------------
# Split dataset for training
# --------------------------
python 4_split_datasets.py $MODEL