import json
import penman
import spacy
import re
import os
import networkx as nx
from collections import Counter
from itertools import combinations
from multiprocessing import Manager, Pool
from tqdm import tqdm
import sys
import hashlib

"""
Extract all motifs and save them to a dictionary. Also add graphs to the dataset.

Output: {model}_amr_data.graph_added.jsonl
Format: {'text':__, 'amr':__, 'graph_dict':__, 'label':__}
"""

nlp = spacy.load("en_core_web_sm")

def get_pos_tag(text, concept):
    # Special AMR concepts
    if concept == "multi-sentence":
        return "MULTI-SENTENCE"
    if concept == "amr-unknown":
        return "?"
    if concept == "truth-value":
        return "TRUTH-VALUE"
    if concept == "amr-choice":
        return "OR"
    
    # Entities
    if concept == "person":
        return "PERSON"
    if concept == "name":
        return "NAME"
    if concept in ["organization", "company", "government-organization", "military", "criminal-organization", "political-party", "market-sector", "school", "university", "research-institute", "team", "league"]:
        return "ORGANIZATION"
    if concept in ["location", "city", "city-district", "county", "state", "province", "territory", "country", "local-region", "country-region", "world-region", "continent", "ocean", "sea", "lake", "river", "gulf", "bay", "strait", "canal", "peninsula", "mountain", "volcano", "valley", "canyon", "island", "desert", "forest", "moon", "planet", "star", "constellation"]:
        return "LOCATION"
    if concept in ["facility", "airport", "station", "port", "tunnel", "bridge", "road", "railway-line", "canal", "building", "theater", "museum", "palace", "hotel", "worship-place", "market", "sports-facility", "park", "zoo", "amusement-park"]:
        return "FACILITY"
    if concept in ["event", "incident", "natural-disaster", "earthquake", "war", "conference", "game", "festival"]:
        return "EVENT"
    if concept in ["product", "vehicle", "ship", "aircraft", "aircraft-type", "spaceship", "car-make", "work-of-art", "picture", "music", "show", "broadcast-program"]:
        return "PRODUCT"
    if concept in ["publication", "book", "newspaper", "magazine", "journal"]:
        return "PUBLICATION"

    # Other entities
    if concept == "date-entity":
        return "DATE"
    if concept == "ordinal-entity":
        return "ORDINAL"
    if concept == "temporal-quantity":
        return "TEMPORAL-QUANTITY"
    if concept in ["date-interval", "year", "month", "day", "weekday", "season", "decade", "century", "era", "timezone"]:
        return "TIME"
    if concept in ["monetary-quantity", "distance-quantity", "area-quantity", "volume-quantity", "temporal-quantity", "frequency-quantity", "speed-quantity", "acceleration-quantity", "mass-quantity", "force-quantity", "pressure-quantity", "energy-quantity", "power-quantity", "voltage-quantity", "charge-quantity", "potential-quantity", "resistance-quantity", "inductance-quantity", "magnetic-field-quantity", "magnetic-flux-quantity", "radiation-quantity", "concentration-quantity", "temperature-quantity", "score-quantity", "fuel-consumption-quantity", "seismic-quantity"]:
        return "QUANTITY"
    if concept in ["percentage-entity", "phone-number-entity", "email-address-entity", "url-entity", "byline-91", "hyperlink-91"]:
        return "IDENTIFIER"

    # POS tags
    base = re.sub(r'-\d+$', '', concept)

    context = nlp(text.lower())
    for token in context:
        if token.lemma_.lower() == base.lower():
            return token.pos_

    no_context = nlp(base)
    return no_context[0].pos_

def amr_to_graph(text, amr):
    amr = penman.decode(amr)
    instances = {source: target for source, role, target in amr.instances()}
    
    # Convert concepts into indexed part-of-speech tags
    pos_counter = Counter(get_pos_tag(text, concept) for concept in instances.values())
    pos_indices = Counter()
    nodes = {}

    for i, (variable, concept) in enumerate(instances.items()):
        # POS tags
        pos = get_pos_tag(text, concept)

        # Manual tags
        if i == 0 and concept == "document":
            pos = "DOCUMENT"
        
        if pos_counter[pos] > 1:
            pos_indices[pos] += 1
            nodes[variable] = f"{pos}-{pos_indices[pos]}"
        else:
            nodes[variable] = pos
    
    # print("Variable\t\tConcept\t\tTag")
    # print("-" * 50)
    # for variable in instances.keys():
    #     print(f"{variable}\t{instances[variable]:<20}\t{nodes[variable]}")     

    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes.values())

    for source, role, target in amr.edges(): 
        if source in nodes and target in nodes:
            G.add_edge(nodes[source], nodes[target], label=role)
    
    return G

def is_isomorphic_multiple(graphs, candidate_graph) -> bool:
    for motif in graphs:
        DiGM = nx.algorithms.isomorphism.DiGraphMatcher(
            motif,
            candidate_graph,
            edge_match=lambda e1, e2: e1["label"] == e2["label"],
        )
        if DiGM.is_isomorphic():
            return True
    return False

def save_graph_motifs(graphs, model, show_tracking=False):
    non_iso_graphs = []
    for graph in tqdm(graphs, desc="Checking isomorphism", disable=not show_tracking):
        if not is_isomorphic_multiple(non_iso_graphs, graph):
            non_iso_graphs.append(graph)

    non_iso_dict = {}
    for G in tqdm(non_iso_graphs, desc="Converting to dict", disable=not show_tracking):
        iso_hash = nx.weisfeiler_lehman_graph_hash(G, edge_attr="label")
        G_dict = nx.json_graph.node_link_data(G)
        non_iso_dict[iso_hash] = G_dict

    with open(f"data/motifs/{model}_motifs.json", "w") as f:
        json.dump(non_iso_dict, f, indent=2)

shared_list = None

def init_globals(manager_list):
    """Initializer for each child process to set the global shared_list."""
    global shared_list
    shared_list = manager_list

def worker_function(G):
    """
    Checks if 'item' is in the shared_list.
    If not found, appends it.
    Returns a message for demonstration.
    """
    for SG in (G.subgraph(s).copy() for s in combinations(G, 3)):
        if len(list(nx.isolates(SG))):
            continue
        if is_isomorphic_multiple(shared_list, SG):
            continue
        shared_list.append(SG)
    return "P"

if __name__ == "__main__":
    model = sys.argv[1]
    file = f"data/{model}/unsplit/{model}_amr_data.jsonl"

    all_graphs = []
    dataset = []

    with open(file) as f:
        for line in tqdm(f):
            data = json.loads(line)
            G = amr_to_graph(data['text'], data['amr'])
            all_graphs.append(G)

            data['graph_dict'] = nx.json_graph.node_link_data(G)
            dataset.append(data)

        with Manager() as manager:
            manager_list = manager.list()
            with Pool(initializer=init_globals, initargs=(manager_list,)) as pool:
                results = list(
                    tqdm(pool.imap(worker_function, all_graphs), total=len(all_graphs))
                )
            motifs = list(manager_list)
            # print("Shared list:", motifs)
            # print("Number of all motifs:", len(motifs))

    save_graph_motifs(motifs, model)

    # Save dataset with NetworkX graphs
    with open(f"data/{model}/unsplit/{model}_amr_data.graph_added.jsonl", "w") as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')