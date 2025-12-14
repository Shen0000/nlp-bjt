# Baseline Models for Human vs. LLM-Generated Text Classification

This repository contains DetectGPT implementation for detecting LLM-generated news articles using zero-shot probability curvature methods.

## Repository Structure

- **`detect-gpt/`** - DetectGPT implementation for news article detection 

## DetectGPT Setup

### 1. Create and activate environment

```bash
cd detect-gpt
python -m venv env
source env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your data

DetectGPT has been modified to work with pre-paired human/LLM datasets in JSONL format:

**Expected format:**
```json
{"human": "Human-written text here...", "llm": "LLM-generated text here..."}
{"human": "Another human text...", "llm": "Another LLM text..."}
```

Place your JSONL files in `/home/common/nlp-bjt/news_data/data/`:
- `mistral_7B_train_sample.jsonl`
- `llama_7B_train_sample.jsonl`
- `llama_13B_train_sample.jsonl`
- `llama_30B_train_sample.jsonl`
- `llama_65B_train_sample.jsonl`
- `falcon_7B_train_sample.jsonl`

## Running DetectGPT

### Quick Test (Recommended First)

Test with 50 samples to verify everything works:

```bash
cd detect-gpt
sbatch slurm_test_run.sh

# Monitor job
squeue -u $USER

# Check results when complete
tail logs/detectgpt-test-*.out
```

### Full Evaluation on HPC

Submit all model evaluations at once:

```bash
cd detect-gpt
./submit_all_jobs.sh
```

This submits jobs for:
- Mistral-7B (8 hours, 50GB GPU memory)
- Llama-7B (8 hours, 50GB GPU memory)
- Llama-13B (10 hours, 60GB GPU memory)
- Llama-65B (12 hours, 80GB GPU memory)
- Falcon-7B (8 hours, 50GB GPU memory)

**Or submit individual models:**

```bash
sbatch slurm_mistral_7b.sh
sbatch slurm_llama_7b.sh
sbatch slurm_llama_13b.sh
sbatch slurm_llama_30b.sh
sbatch slurm_llama_65b.sh
sbatch slurm_falcon_7b.sh
```

### Monitor Jobs

```bash
# Check job queue
squeue -u $USER

# Check recent job history
sacct -u $USER --starttime=today --format=JobID,JobName%30,State,ExitCode,Elapsed

# View logs in real-time
tail -f logs/detectgpt-mistral7b-<JOBID>.out
tail -f logs/detectgpt-mistral7b-<JOBID>.err
```

### Results Location

Results are saved to:
```
detect-gpt/results/<output_name>/
```

Each result includes:
- ROC-AUC scores for different perturbation counts
- Precision-Recall metrics
- Raw prediction data

## Understanding DetectGPT Results

DetectGPT outputs metrics like:
- `perturbation_10_d`: 10 perturbations, difference criterion
- `perturbation_10_z`: 10 perturbations, z-score criterion
- `perturbation_50_d`: 50 perturbations, difference criterion
- `perturbation_50_z`: 50 perturbations, z-score criterion (usually best)
- `perturbation_{n}_d`: n perturbations if set in batch script
- `perturbation_{n}_z`: n perturbations if set in batch script

**ROC-AUC interpretation:**
- 0.5 = Random guessing
- 0.7-0.8 = Moderate detection
- 0.9+ = Strong detection

**Note:** Short texts (<100 words) typically show lower ROC-AUC. The DetectGPT paper achieved 0.95 AUROC on longer news articles (100+ words).

## Troubleshooting

### Package Version Conflicts

If you see errors about `pyarrow` or `numpy`:
```bash
pip install "numpy<2.0.0" "pyarrow>=12.0.0,<15.0.0" --force-reinstall
```

### Cache Permission Errors

If you see "PermissionError" for `.cache/.locks/`:
```bash
cd detect-gpt
rm -rf .cache/.locks/
# Or use your own cache:
export HF_HOME=~/.cache/huggingface
```

### OOM (Out of Memory) Errors

- Use `--base_half` flag (already enabled in scripts) for FP16 precision
- Reduce `--batch_size` if still getting OOM
- Use smaller models (7B/13B) instead of 70B

### Job Fails Immediately

Check the error log:
```bash
cat logs/detectgpt-<model>-<JOBID>.err
```

Common issues:
- Environment not activated correctly
- Data files not found
- GPU memory insufficient

## Acknowledgements
Our code is adapted from the [detect-gpt code repository](https://github.com/eric-mitchell/detect-gpt/tree/main) associated with the paper *“DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature”* (2023). We thank the authors for making their code publicly available and enabling our project.