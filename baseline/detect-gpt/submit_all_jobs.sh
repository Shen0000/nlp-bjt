#!/bin/bash
# Master script to submit all DetectGPT evaluation jobs to SLURM
# Submits jobs with delays to be respectful of HPC resources

set -e

DETECT_GPT_DIR="/home/common/nlp-bjt/baseline/detect-gpt"
cd "$DETECT_GPT_DIR"

# Create logs directory
mkdir -p logs

echo "========================================"
echo "Submitting DetectGPT Evaluation Jobs"
echo "========================================"
echo ""
echo "This will submit 4 jobs (one per model) to the HPC queue."
echo "Each job will use 1 A100 GPU with reasonable resource limits."
echo ""
echo "Jobs will be submitted with delays to avoid overwhelming the queue."
echo "========================================"
echo ""

# Function to submit a job with confirmation
submit_job() {
    local script=$1
    local model_name=$2
    
    if [ -f "$script" ]; then
        echo "Submitting: $model_name"
        job_id=$(sbatch "$script" | awk '{print $4}')
        echo "Job ID: $job_id"
        echo ""
        sleep 2  # Small delay between submissions
    else
        echo "$script not found, skipping $model_name"
        echo ""
    fi
}

# Submit jobs
submit_job "slurm_mistral_7b.sh" "Mistral 7B"
submit_job "slurm_llama_7b.sh" "Llama 7B"
submit_job "slurm_llama_13b.sh" "Llama 13B"
submit_job "slurm_falcon_7b.sh" "Falcon 7B"

echo "========================================"
echo "All jobs submitted!"
echo "========================================"
echo ""
echo "Check job status with: squeue -u $USER"
echo "Check job details with: scontrol show job <job_id>"
echo "Cancel a job with: scancel <job_id>"
echo ""
echo "Results will be saved in: results/"
echo "Logs will be saved in: logs/"
