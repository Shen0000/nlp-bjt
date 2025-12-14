import re
import os
import json

"""
Prepare datasets before running pipeline.

Output: {model}_amr_data.jsonl
Format: {'text':__, 'amr':__, 'label':__}
"""

def preprocess(file):
    with open(file, 'r', encoding='utf-8') as infile:
        blocks = infile.read().split('\n\n')
        documents = []
        graphs = []
        for block in blocks:
            graph = []
            for line in block.split('\n'):
                if not line.startswith('# ::tok'):
                    graph.append(line)
                else:
                    documents.append(line[8:])
            graphs.append('\n'.join(graph))
    return documents, graphs

models = ["falcon_7B", "llama_7B", "llama_13B", "llama_30B", "llama_65B", "mistral_7B"]

for model in models:
    os.makedirs(f'data/{model}/unsplit/', exist_ok=True)

    with open(f'data/{model}/unsplit/{model}_amr_data.jsonl', 'w') as outfile:
        human_documents, human_graphs = preprocess("data/human_amr_original.txt")
        for i in range(len(human_documents)):
            graph = re.sub(r'\s+', ' ', human_graphs[i])
            json.dump({'text': human_documents[i], 'amr': graph, 'label': 0}, outfile)
            outfile.write('\n')
        
        llm_documents, llm_graphs = preprocess(f'data/{model}/unsplit/{model}_amr_data.jsonl')
        for i in range(len(llm_documents)):
            graph = re.sub(r'\s+', ' ', llm_graphs[i])
            json.dump({'text': llm_documents[i], 'amr': graph, 'label': 1}, outfile)
            outfile.write('\n')