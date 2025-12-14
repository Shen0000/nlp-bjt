# Setup 3: Detecting LLM-Generated Texts Using AMR Motifs

Code for extracting and incorporating AMR Motifs into binary classifiers

## Requirements

1. **Create and activate conda environment**  
   ```shell script
   conda create -n setup_3
   conda activate setup_3
   ```

2. **Install dependencies**  
   ```shell script
   pip install -r requirements.txt
   ```

## Data

- Parse all texts into AMR graphs using a state-of-the-art parser. In our experiments, we used [docAMR](https://github.com/IBM/docAMR). Download the pretrained checkpoints from [Transition-based Neural Parser](
transition-amr-parser) (see setup_1 for the AMR parsing pipeline).
- Place human data in the `data/` directory. Refer to `human_amr_original.txt` for the expected file format.
- Place LLM-generated data (from one of the six models) in the appropriate `data/<model>/unsplit/` folder
- Run the `0_prepare_amr_datasets.py` script to combine human and LLM-generated data into a `jsonl` file and assign labels. The resulting data file is in the appropriate `data/<model>/unsplit/`.

## Pipeline

The complete pipeline for extracting AMR motifs and incorporating them into datasets is automated in `pipeline.sh`.

**To run the pipeline on the HPC**
```shell script
sbatch pipeline.sh
```
This will:
- Extract motifs from AMR graphs
- Add motif distributions to the dataset
- Select top motifs
- Split the dataset for training/testing

**Note:**  
Before running, edit `pipeline.sh` to set the `MODEL` variable to match the dataset (e.g., `llama_30B`)

## Training & Evaluation

### Decision Tree
```shell script
python train_dt.py <model>
```

### XGBoost
```shell script
python train_xg.py <model>
```

### Longformer
See `longformer.sh` for training and evaluation commands:
```shell script
sbatch longformer.sh
```
Before running, edit `longformer.sh` to set the `MODEL` variable to match the dataset and indicate whether to include motifs

## Acknowledgements
Our code is adapted from the [Threads of Subtlety code repository](https://github.com/minnesotanlp/threads-of-subtlety) associated with the paper *“Threads of Subtlety: Detecting Machine-Generated Texts Through Discourse Motifs”* (ACL 2024). We thank the authors for making their code publicly available and enabling our project.