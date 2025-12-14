# What Do You Mean

This repository contains code for our final project.

## Directory Structure

- `baseline/`: Code for running baseline models.
- `news_data/`: Code for extracting NYT news data and preprocessing
- `setup_1/`: Code for DART framework using [Abstract Meaning Representation](https://aclanthology.org/W13-2322.pdf) (AMR).
- `setup_2/`: Code for DART framework using Discourse Representation Structure (DRS).
- `setup_3/`: Code for incorporating AMR motifs, based on [discourse motifs](https://aclanthology.org/2024.acl-long.298.pdf)

Each folder contains its own README with instructions.

## Data

We use 1000 pairs of human and LLM-generated news texts sampled from the [Contrasting Linguistic Patterns in Human and LLM-Generated News Text](https://arxiv.org/abs/2308.09067) dataset. 

The dataset includes data from six LLMs:
- Mistral 7B
- Falcon 7B
- LLaMA 7B
- LLaMA 13B
- LLaMA 30B
- LLaMA 65B

Each setup folder contains subfolders for each model’s data.

## Acknowledgements
This project is supported by the Amherst College HPC, which is funded by NSF Award 2117377.