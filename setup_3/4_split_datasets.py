import os
import sys
import json
import random
from typing import List, Dict

"""
Split the dataset into train, validation, and test sets.

Train: split/{model}_train.jsonl
Validation: split/{model}_valid.jsonl
Test: split/{model}_test.jsonl
"""

random.seed(42)

def save_dataset_as_jsonl(dataset: List[Dict], output_path: str):
    with open(output_path, 'w') as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    model = sys.argv[1]

    dataset = []

    with open(f"data/{model}/unsplit/{model}_amr_data.motif_dists_selected.jsonl", 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)

    random.shuffle(dataset)

    total = len(dataset)
    train_size = int(total * 0.8)
    valid_size = int(total * 0.1)

    # Split
    train = dataset[:train_size]
    valid = dataset[train_size:train_size + valid_size]
    test = dataset[train_size + valid_size:]

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    print(f"train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

    os.makedirs(f'data/{model}/split/', exist_ok=True)

    save_dataset_as_jsonl(train, f"data/{model}/split/{model}_train.jsonl")
    save_dataset_as_jsonl(valid, f"data/{model}/split/{model}_valid.jsonl")
    save_dataset_as_jsonl(test, f"data/{model}/split/{model}_test.jsonl")