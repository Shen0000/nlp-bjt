#! /bin/sh

#SBATCH --job-name=llama-13b
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --output=llama-13b-%j.out
#SBATCH --error=llama-13b-%j.err

MODEL="llama_13B"

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
cd /home/common/nlp-bjt/setup_2

# -------------------------------
# Activate DRS parser environment
# -------------------------------
conda activate /home/common/nlp-bjt/setup_2/conda-setup_2

# ----------------------
# Run text-to-DRS parser
# ----------------------
cd TreeDRSparsing/workspace/gd_sys1

for ITERATION in 0 1 2; do
    python main.py easy \
        --model-path-base models \
        --pretrain-path embeddings/sskip.100.vectors \
        --action-dict-path data/dict \
        --const \
        --beam-size 1 \
        --source $MODEL \
        --i $ITERATION
    echo "Successfully parsed DRS for T${ITERATION}"
done

# -------------------------------
# Compute semantic gap
# -------------------------------
mkdir -p ${MODEL}_eval_outputs
cd ${MODEL}_eval_outputs

ln -s ../../../scripts/tree2tuple.py
ln -s ../../../scripts/counter_gmb.py

for ITERATION in 1 2; do
    for AUTHOR in human llm; do
        python tree2tuple.py --input /home/common/nlp-bjt/setup_2/$MODEL/drs/${AUTHOR}_drs_${ITERATION}.txt \
            > ${AUTHOR}_drs_${ITERATION}.tuple

        if [ $ITERATION -eq 1 ]; then
            python tree2tuple.py --input /home/common/nlp-bjt/setup_2/$MODEL/drs/${AUTHOR}_drs_$((ITERATION-1)).txt \
                > ${AUTHOR}_drs_$((ITERATION-1)).tuple
        fi

        python counter_gmb.py \
            -f1 ${AUTHOR}_drs_${ITERATION}.tuple \
            -f2 ${AUTHOR}_drs_$((ITERATION-1)).tuple \
            -msf ${AUTHOR}_${ITERATION}_score.jsonl \
            -pr -r 20 -p 20

        echo "Successfully calculated ${AUTHOR}_${ITERATION}_score.jsonl"
    done
done

echo "Successfully finished pipeline"