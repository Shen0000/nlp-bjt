import json
import numpy as np
from typing import Dict, List, Tuple
import sys

"""
Select motifs with MF-IDF scores surpassing at least one standard deviation.
Save the selected motif hashes to a dictionary.
"""

def compute_mfidf_distribution(data_file):
    human_mf = []
    human_df = []
    llm_mf = []
    llm_df = []

    with open(data_file) as f:
        for line in f:
            document = json.loads(line.strip())
            if document['label'] == 0:
                human_mf.append(document['motif_dists']['m3']['mf'])
                human_df.append(document['motif_dists']['m3']['raw'])
            elif document['label'] == 1:
                llm_mf.append(document['motif_dists']['m3']['mf'])
                llm_df.append(document['motif_dists']['m3']['raw'])
    
    human_mf = np.array(human_mf)
    human_mf = np.mean(human_mf, axis=0)

    human_df = np.array(human_df)
    human_df = (human_df > 0).astype(int)
    human_df = np.mean(human_df, axis=0)

    llm_mf = np.array(llm_mf)
    llm_mf = np.mean(llm_mf, axis=0)

    llm_df = np.array(llm_df)
    llm_df = (llm_df > 0).astype(int)
    llm_df = np.mean(llm_df, axis=0)

    df = human_df + llm_df
    idf = np.log((2 + 1) / (df + 1))

    human_mfidf = np.multiply(human_mf, idf)
    llm_mfidf = np.multiply(llm_mf, idf)

    # Filter scores surpassing at least one standard deviation
    human_mean = np.mean(human_mfidf)
    human_std = np.std(human_mfidf)
    human_threshold = human_mean + human_std
    
    llm_mean = np.mean(llm_mfidf)
    llm_std = np.std(llm_mfidf)
    llm_threshold = llm_mean + llm_std
    
    human_high_indices = np.where(human_mfidf > human_threshold)[0]
    llm_high_indices = np.where(llm_mfidf > llm_threshold)[0]

    # Keep motifs that exceed threshold in either corpus
    selected_indices = np.union1d(human_high_indices, llm_high_indices)

    print(f"Number of selected motifs: {len(selected_indices)}/{len(human_mfidf)}")

    return selected_indices

def main():
    model = sys.argv[1]
    data_file = f"data/{model}/unsplit/{model}_amr_data.motif_dists_all.jsonl"
    selected_indices = compute_mfidf_distribution(data_file)

    with open(f"data/motifs/{model}_motifs.json", "r") as f:
        motifs = json.load(f)
        motif_hashes = list(motifs.keys())
        selected_hashes = [motif_hashes[i] for i in selected_indices]

    with open(f"data/motifs/{model}_selected-motif-hashes.json", 'w') as file:
        json.dump(selected_hashes, file)

main()