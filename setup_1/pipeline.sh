#! /bin/sh

#SBATCH --job-name=llama-13b
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/llama-13b-%j.out
#SBATCH --error=logs/llama-13b-%j.err

REPHRASER="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL="llama_13B"
REPHRASER_NAME="Qwen3-30B"

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
# Go to base working directory
# -------------------------------
cd /home/common/nlp-bjt/setup_1

# -------------------------------
# Activate environment for rephraser and sentence parser
# -------------------------------
conda activate /home/common/nlp-bjt/setup_1/conda-setup_1

# -------------------------------
# Run rephraser scripts
# -------------------------------
python rephraser.py "$MODEL" 1 "$REPHRASER" "$REPHRASER_NAME" 
python rephraser.py "$MODEL" 2 "$REPHRASER" "$REPHRASER_NAME"
echo "Rephrased data successfully!"

# -------------------------------
# Run sentence parser script
# -------------------------------
python sentence_parser.py "$MODEL" 1 "$REPHRASER_NAME" --original
python sentence_parser.py "$MODEL" 1 "$REPHRASER_NAME"
python sentence_parser.py "$MODEL" 2 "$REPHRASER_NAME"
echo "Splitted data into sentences successfully!"

# -------------------------------
# Switch to AMR parser environment
# -------------------------------
conda deactivate
conda activate /home/common/nlp-bjt/setup_1/transition-amr-parser/cenv_x86

# -------------------------------
# Run sentence parser for AMR
# -------------------------------
cd /home/common/nlp-bjt/setup_1/transition-amr-parser
python parsing.py "$MODEL" 1 "$REPHRASER_NAME" --original
python parsing.py "$MODEL" 1 "$REPHRASER_NAME"
python parsing.py "$MODEL" 2 "$REPHRASER_NAME"
echo "Parsed data into AMR graphs successfully!"

# -------------------------------
# Compute Semantic Gap
# -------------------------------
cd /home/common/nlp-bjt/setup_1/transition-amr-parser/docAMR
python docSmatch/smatch.py -f /home/common/nlp-bjt/setup_1/$MODEL/$REPHRASER_NAME/amrs/human_amr_1.txt /home/common/nlp-bjt/setup_1/human_amr_original.txt --source "$MODEL" --rephraser "$REPHRASER_NAME" --generation human --iteration 1
python docSmatch/smatch.py -f /home/common/nlp-bjt/setup_1/$MODEL/$REPHRASER_NAME/amrs/human_amr_2.txt /home/common/nlp-bjt/setup_1/$MODEL/$REPHRASER_NAME/amrs/human_amr_1.txt --source "$MODEL" --rephraser "$REPHRASER_NAME" --generation human --iteration 2
python docSmatch/smatch.py -f /home/common/nlp-bjt/setup_1/$MODEL/$REPHRASER_NAME/amrs/llm_amr_1.txt /home/common/nlp-bjt/setup_1/$MODEL/$REPHRASER_NAME/amrs/llm_amr_original.txt --source "$MODEL" --rephraser "$REPHRASER_NAME" --generation llm --iteration 1
python docSmatch/smatch.py -f /home/common/nlp-bjt/setup_1/$MODEL/$REPHRASER_NAME/amrs/llm_amr_2.txt /home/common/nlp-bjt/setup_1/$MODEL/$REPHRASER_NAME/amrs/llm_amr_1.txt --source "$MODEL" --rephraser "$REPHRASER_NAME" --generation llm --iteration 2
echo "Computed semantic gap successfully!"

# -------------------------------
# Align the Smatch scores
# -------------------------------
conda deactivate
conda activate /home/common/nlp-bjt/setup_1/conda-setup_1
cd /home/common/nlp-bjt/setup_1
python align_scores.py --source "$MODEL" --rephraser "$REPHRASER_NAME"
echo "Aligned Smatch scores successfully!"

# -------------------------------
# Train Decision Tree
# -------------------------------
cd /home/common/nlp-bjt/setup_1
python train_dt.py "$MODEL"
echo "Trained decision tree!"

# -------------------------------
# Train XGBoost
# -------------------------------
cd /home/common/nlp-bjt/setup_1
python train_xg.py "$MODEL"
echo "Trained XGBoost!"

# -------------------------------
# Done
# -------------------------------
echo "Pipeline finished successfully!"
