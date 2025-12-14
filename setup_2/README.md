# Setup 1: DART Framework Using DRS

Code for rephrasing and parsing texts into DRS graphs to classify human-generated vs. LLM-generated texts.

## Requirements

1. **Create and activate conda environment**
   ```shell script
   conda create -p ./conda-setup_2
   conda activate conda-setup_2
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
- Pass all paired texts into rephraser model (e.g., Qwen3-30B) to rewrite them. After obtaining the rephrase of the original text, rephrase the outputs again.
- Split the outputs into sentences to prepare for DRS parsing.
- Parse all sentences into DRS graphs using the state-of-the-art document-level text-to-DRS parser. In our experiments, we download [TreeDRSparsing](https://github.com/LeonCrashCode/TreeDRSparsing/tree/bs_sattn_drssup). We should obtain 6 files: human_drs_0.txt (DRS of original human text), human_drs_1.txt (DRS of rephrased human text), human_drs_2.txt (DRS of rewritten texts from human rephrased text), llm_drs_orignal.txt (DRS of original llm text), llm_drs_1.txt (DRS of rephrased llm text), llm_drs_2.txt (DRS of rewritten texts from llm rephrased text)
- Compute document-level Counter scores between: human_drs_0 & human_drs_1, human_drs_1 & human_drs_2, llm_drs_0 & llm_drs_1, llm_dts_1 & llm_drs_2.
- Run the train_test_dev_split.py to obtain the dataset splits to prepare for model training.
- Run the train_dt.py and train_xg.py to train a Decision Tree model and XGBoost model, respectively.

**Note:**
- Please download the TreeDRSparsing repository locally.
- This setup relies on rephrased texts generated in setup 1. Make sure all rephrased files are placed in the appropriate `data/{model}` directory.
- Before running the pipeline, process these files with `sentence_parser.py` to prepare them for the text-to-DRS parser.
- Before running, edit `pipeline.sh` to set the `MODEL` variable to match the dataset (e.g., `llama_30B`)