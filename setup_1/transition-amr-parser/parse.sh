#! /bin/sh

#SBATCH --job-name=amr_parse
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output amr-%j.out
#SBATCH --error amr-%j.err

module purge
module load cuda11.8/toolkit

# Activate conda
source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate /home/common/nlp-bjt/setup_1/transition-amr-parser/cenv_x86

echo "Which python:"
which python
python --version

python -m pip list | grep transition


# cd /home/common/nlp-bjt/setup_1/transition-amr-parser
# export PYTHONPATH="${PYTHONPATH}:/home/common/nlp-bjt/setup_1/transition-amr-parser/cenv_x86"

# python parsing.py

# # cd /home/common/nlp-bjt/setup_1/transition-amr-parser

# # conda run -p /home/common/nlp-bjt/setup_1/transition-amr-parser/cenv_x86 python parsing.py