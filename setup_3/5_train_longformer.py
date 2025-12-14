import random
import numpy as np
import torch
import os
import sys
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    LongformerForSequenceClassification,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)
from transformers.tokenization_utils_base import PaddingStrategy

from tos.tos_models import LongformerWithMotifsForSequenceClassification
from tos.tos_utils import compute_metrics

"""
Train Longformer WITH or WITHOUT motifs. Save model to results/longformer_{model}_motif.
"""

random.seed(42)

class LongformerDataset(Dataset):
    def __init__(self, file_path: str, shuffle: bool = False):
        self.dataset = []
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.dataset.append(item)
        
        if shuffle:
            import random
            random.shuffle(self.dataset)
        
        print(f"Dataset loaded from {file_path}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

@dataclass
class LongformerDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    add_motif: bool = False

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        text_data = []
        labels = []
        motif_dists = []

        for data in features:
            text_data.append(data["text"])
            labels.append(data["label"])
            if self.add_motif:
                m3_mf = np.array(data["motif_dists"]["m3"]["mf"], dtype=np.float32)
                m3_wad = np.array(data["motif_dists"]["m3"]["wad"], dtype=np.float32)
                m3_feats = np.zeros(m3_mf.shape[0] * 2, dtype=np.float32)
                m3_feats[::2] = m3_mf
                m3_feats[1::2] = m3_wad
                motif_dists.append(m3_feats)

        batch = self.tokenizer(
            text_data,
            padding=self.padding,
            return_tensors="pt",
            truncation=True,
        )
        batch["labels"] = torch.tensor(labels)
        if self.add_motif:
            batch["motif_dists"] = torch.tensor(
                motif_dists, dtype=torch.float
            )
        return batch

def train_longformer_plain(
    model_name: str,
    train_file: str,
    valid_file: str,
    base_model_path: str = "allenai/longformer-base-4096",
    num_labels: int = 2,
):

    with open(train_file, 'r') as f:
        sample = json.loads(f.readline())
        m3_mf = np.array(sample["motif_dists"]["m3"]["mf"])
        motif_dims = m3_mf.shape[0] * 2
    # print(f"Motif dimension: {motif_dims}")

    model = LongformerForSequenceClassification.from_pretrained(
        base_model_path, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    train_dataset = LongformerDataset(train_file, shuffle=True)
    valid_dataset = LongformerDataset(valid_file, shuffle=False)
        
    training_args = TrainingArguments(
        # use_mps_device=True,
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        metric_for_best_model="f1",
        greater_is_better=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        load_best_model_at_end=True,
        logging_dir=f"./models/{model_name}",  # directory for storing logs
        logging_strategy="steps",
        num_train_epochs=4,  # total number of training epochs
        output_dir=f"./results/{model_name}",  # output directory
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        remove_unused_columns=False,
        save_total_limit=3,
        weight_decay=4e-5,
    )
    
    trainer = Trainer(
        args=training_args,
        data_collator=LongformerDataCollator(
            tokenizer=tokenizer, padding="longest", add_motif=False
        ),
        eval_dataset=valid_dataset,
        model=model,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model

def train_longformer_motif(
    model_name: str,
    train_file: str,
    valid_file: str,
    base_model_path: str = "allenai/longformer-base-4096",
    num_labels: int = 2,
):
    with open(train_file, 'r') as f:
        sample = json.loads(f.readline())
        m3_mf = np.array(sample["motif_dists"]["m3"]["mf"])
        motif_dims = m3_mf.shape[0] * 2
    # print(f"Motif dimension: {motif_dims}")

    model = LongformerWithMotifsForSequenceClassification(
        base_model_path, num_labels=num_labels, motif_dims=motif_dims
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=True
    )
    
    train_dataset = LongformerDataset(train_file, shuffle=True)
    valid_dataset = LongformerDataset(valid_file, shuffle=False)
        
    training_args = TrainingArguments(
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        metric_for_best_model="f1",
        greater_is_better=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        load_best_model_at_end=True,
        logging_dir=f"./models/{model_name}",  # directory for storing logs
        logging_strategy="steps",
        num_train_epochs=4,  # total number of training epochs
        output_dir=f"./results/{model_name}",  # output directory
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        remove_unused_columns=False,
        save_total_limit=3,
        weight_decay=4e-5,
    )

    trainer = Trainer(
        args=training_args,
        data_collator=LongformerDataCollator(
            tokenizer=tokenizer, padding="longest", add_motif=True
        ),
        eval_dataset=valid_dataset,
        model=model,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model


if __name__ == "__main__":    
    model = sys.argv[1]
    use_motifs = len(sys.argv) > 2 and sys.argv[2] == "--motif"
    
    train_file = f"data/{model}/split/{model}_train.jsonl"
    valid_file = f"data/{model}/split/{model}_valid.jsonl"
    
    if use_motifs:
        print(f"train_longformer_motif for {model}")
        model_output_name = f"longformer_{model}_motif"
        train_longformer_motif(
            model_name=model_output_name,
            train_file=train_file,
            valid_file=valid_file
        )
    else:
        print(f"train_longformer_plain for {model}")
        model_output_name = f"longformer_{model}_plain"
        train_longformer_plain(
            model_name=model_output_name,
            train_file=train_file,
            valid_file=valid_file
        )