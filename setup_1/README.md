# Setup 1: DART Framework Using AMR

Code for rephrasing and parsing texts into AMR graphs to classify human-generated vs. LLM-generated texts.

## Requirements

1. **Create and activate conda environment**
   ```shell script
   conda create -p ./conda-setup_1
   conda activate conda-setup_1
   ```

2. **Install dependencies**
   ```shell script
   pip install -r requirements.txt
   ```

## Data

- Obtain the data from NYT Developer Portal API between the dates October 2023 and January 2024. Match the ``lead paragraph'' texts with the LLM-generated data from the Muñoz‑Ortiz et al. (2024) work. The files should be in jsonl with items formatted as {"human": "", "llm": ""}.
- Obtain 1000 samples from the filtered, paired dataset using random seed 42 for all models.

## Pipeline

The complete pipeline is automated in `pipeline.sh`.

**To run the pipeline on the HPC**
```shell script
sbatch pipeline.sh
```
This will:
- Split the outputs into sentences to prepare for AMR parsing.
- Parse all sentences into AMR graphs using the state-of-the-art document-level text-to-AMR parser. In our experiments, we used [docAMR](https://github.com/IBM/docAMR). Download the pretrained checkpoints from [Transition-based Neural Parser](
transition-amr-parser) (see setup_1 for the AMR parsing pipeline). We should obtain 6 files: human_amr_orignal.txt (AMRs of original human text), human_amr_1.txt (AMRs of rephrased human text), human_amr_2.txt (AMRs of rewritten texts from human rephrased text), llm_amr_orignal.txt (AMRs of original llm text), llm_amr_1.txt (AMRs of rephrased llm text), llm_amr_2.txt (AMRs of rewritten texts from llm rephrased text)
- Compute document-level Smatch scores between: human_amr_original & human_amr_1, human_amr_1 & human_amr_2, llm_amr_original & llm_amr_1, llm_amr_1 & llm_amr_2.
- Run the align_scores.py to filter out the the text pairs that do not successfully have a Smatch score due to ill-formatted AMR graphs.
- Run the train_test_dev_split.py to obtain the dataset splits to prepare for model training.
- Run the train_dt.py and train_xg.py to train a Decision Tree model and XGBoost model, respectively.

**Note:**
Before running, edit `pipeline.sh` to set the `MODEL` variable to match the dataset (e.g., `llama_30B`), the `REPHRASER` variable to match the rephraser model you are using (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507), and the `REPHRASER_NAME` variable that you want to call the rephraser model (e.g., Qwen3-30B).
