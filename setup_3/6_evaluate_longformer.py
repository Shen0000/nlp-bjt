import random
import numpy as np
import torch
import os
import sys
import json
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import evaluate
import torch
from accelerate import Accelerator
from rich.progress import track
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LongformerForSequenceClassification, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy

from tos.tos_models import LongformerWithMotifsForSequenceClassification

"""
Evaluate Longformer WITH or WITHOUT motifs on test set.
"""

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

def evaluate_longformer(test_file: str, model_path: str, add_motif: bool):
    accelerator = Accelerator()
    # accelerator.print(f"\n\n------{test_file}-------")

    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096", use_fast=True
    )

    metric = evaluate.combine(
        ["accuracy", "f1", "precision", "recall", "BucketHeadP65/confusion_matrix"]
    )

    if add_motif:
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))

        dense_weight_shape = state_dict['classifier.dense.weight'].shape
        hidden_size = dense_weight_shape[0]
        input_size = dense_weight_shape[1]
        motif_dims = input_size - hidden_size

        model = LongformerWithMotifsForSequenceClassification(motif_dims=motif_dims)
        model.load_state_dict(state_dict)
    else:
        model = LongformerForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )

    test_dataset = LongformerDataset(test_file, shuffle=True)

    data_loader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=LongformerDataCollator(
            tokenizer=tokenizer, padding="longest", add_motif=add_motif
        ),
    )

    model, data_loader = accelerator.prepare(model, data_loader)

    for data in track(data_loader, total=len(data_loader), description="Evaluating..."):
        targets = data["labels"]
        predictions = model(**data).logits

        predictions = torch.argmax(predictions, dim=1)
        all_predictions, all_targets = accelerator.gather_for_metrics(
            (predictions, targets)
        )
        metric.add_batch(predictions=all_predictions, references=all_targets)

    accelerator.print(metric.evaluation_modules[0].__len__())
    accelerator.print(metric.compute())
    # accelerator.print("-----------------------")


if __name__ == "__main__":
    model = sys.argv[1]
    use_motifs = len(sys.argv) > 2 and sys.argv[2] == "--motif"
    test_file = f"data/{model}/split/{model}_test.jsonl"

    if use_motifs:
        # print(f"longformer_{model}_motif")
        model_path = f"./results/longformer_{model}_motif/checkpoint-100"
    else:
        # print(f"longformer_{model}_plain")
        model_path = f"./results/longformer_{model}_plain/checkpoint-100"

    evaluate_longformer(
        test_file,
        model_path=model_path,
        add_motif=use_motifs,
    )