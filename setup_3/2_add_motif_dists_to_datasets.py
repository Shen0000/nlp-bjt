import json
import numpy as np
import os
import random
from glob import glob
from typing import Any, Dict, List
import sys

import networkx as nx
from tqdm.contrib.concurrent import process_map

"""
Add motif distributions to the dataset.

If ALL motifs
Output: {model}_amr_data.motif_dists_all.jsonl
Format: {'text':__, 'amr':__, 'graph_dict':__, 'motif_dists':__, 'label':__}

If SELECTED motifs 
Output: {model}_amr_data.motif_dists_selected.jsonl
Format: {'text':__, 'amr':__, 'graph_dict':__, 'motif_dists':__, 'label':__} 
"""

random.seed(42)

def load_motifs(motif_path: str, selected_hashes: List[str] = None) -> List[nx.DiGraph]:
    motif_graphs = []
    with open(motif_path, "r") as f:
        _motifs = json.load(f)
        if selected_hashes:
            motif_graphs.extend(
                [
                    nx.json_graph.node_link_graph(_motifs[hash])
                    for hash in selected_hashes
                ]
            )
        else:
            motif_graphs.extend(
                [nx.json_graph.node_link_graph(v) for v in _motifs.values()]
            )
    # print(f"loaded motif graphs: {len(motif_graphs)}")
    return motif_graphs

m3_motifs = None

def calculate_motif_distribution(G: nx.DiGraph, graph_motifs: List[nx.DiGraph], root_label: str) -> Dict[str, np.ndarray]:
    G_diameter = nx.diameter(G.to_undirected())
    hist = np.zeros(len(graph_motifs), dtype=float)
    wad = np.zeros(len(graph_motifs), dtype=float)

    for index, motif in enumerate(graph_motifs):
        if motif.number_of_nodes() == 1:
            hist[index] = G.number_of_nodes()
            continue
        if motif.number_of_nodes() == 2:
            hist[index] = G.number_of_edges()
            continue

        DiGM = nx.algorithms.isomorphism.DiGraphMatcher(
            G, motif, edge_match=lambda e1, e2: e1["label"] == e2["label"]
        )

        counts_per_depth = {}

        for subgraph in DiGM.subgraph_isomorphisms_iter():
            motif_nodes = subgraph.keys()
            motif_depth = np.mean(
                [
                    nx.shortest_path_length(
                        G.to_undirected(), source=root_label, target=node_label
                    )
                    for node_label in motif_nodes
                ]
            )
            if motif_depth not in counts_per_depth:
                counts_per_depth[motif_depth] = 1
            else:
                counts_per_depth[motif_depth] += 1

            hist[index] += 1

        counts_x_depths = np.sum(
            [depth * counts for depth, counts in counts_per_depth.items()]
        )

        # sum(depth x count) / sum(count)
        wad[index] = counts_x_depths / hist[index] if hist[index] > 0 else -1

    num_of_motifs = np.sum(hist)
    motif_freqs = hist / num_of_motifs if num_of_motifs > 0 else hist
    wad = wad / G_diameter
    # -1 means that the motif does not exist in the graph
    wad[wad < 0] = -1
    return {"raw": hist.tolist(), "mf": motif_freqs.tolist(), "wad": wad.tolist()}

def find_amr_root(G: nx.DiGraph) -> str:
    for node in G.nodes():
        if G.in_degree(node) == 0:
            return node
    return max(G.nodes(), key=lambda n: G.out_degree(n))

def add_motif_dist_to_document(document: Dict) -> Dict:
    graph = document['graph_networkx']
    root_label = find_amr_root(graph)
    m3_dists = calculate_motif_distribution(graph, m3_motifs, root_label=root_label)
    document['motif_dists'] = {"m3": m3_dists}
    return document

def load_document_corpus(file_path: str) -> List[Dict]:
    dataset = []
    with open(file_path) as f:
        for line in f:
            document = json.loads(line.strip())
            document['graph_networkx'] = nx.json_graph.node_link_graph(document['graph_dict'])
            dataset.append(document)
    return dataset

def save_dataset_as_jsonl(dataset: List[Dict], output_path: str):
    with open(output_path, 'w') as f:
        for data in dataset:
            output = {
                'text': data['text'],
                'amr': data['amr'],
                'graph_dict': data['graph_dict'],
                'motif_dists': data['motif_dists'],
                'label': data['label']
            }
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    model = sys.argv[1]
    selected = len(sys.argv) > 2 and sys.argv[2] == "--selected"
    file_path = f"data/{model}/unsplit/{model}_amr_data.graph_added.jsonl"

    if selected:
        with open(f'data/motifs/{model}_selected-motif-hashes.json', 'r') as file:
            selected_motif_hashes = json.load(file)
        m3_motifs = load_motifs(f"data/motifs/{model}_motifs.json", selected_motif_hashes)
    else:
        m3_motifs = load_motifs(f"data/motifs/{model}_motifs.json")

    dataset = load_document_corpus(file_path)

    dataset = process_map(
        add_motif_dist_to_document, dataset, max_workers=14, chunksize=1
    )

    if selected:
        save_dataset_as_jsonl(dataset, f"data/{model}/unsplit/{model}_amr_data.motif_dists_selected.jsonl")
    else:
        save_dataset_as_jsonl(dataset, f"data/{model}/unsplit/{model}_amr_data.motif_dists_all.jsonl")