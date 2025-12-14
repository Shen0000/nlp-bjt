import json
from transition_amr_parser.parse import AMRParser
import os

# --- Initialize the parser once ---
parser = AMRParser.from_pretrained('doc-sen-conll-amr-seed42')

def parse(doc_sentences):
    """
    Parse a list of tokenized sentences with the AMR parser.
    
    Args:
        doc_sentences (List[str]): List of sentences (strings).
        
    Returns:
        str: AMR annotation in Penman notation for the document.
    """
    tok_sentences = []
    for sen in doc_sentences:
        tokens, positions = parser.tokenize(sen)
        tok_sentences.append(tokens)

    try:
        annotations, _ = parser.parse_docs([tok_sentences])
    except Exception as e:
        print(f"[WARN] Failed to parse doc: {doc_sentences}\n  Error: {e}")
        return None  # skip this document

    if annotations[0] is None:
        print(f"[WARN] Parser returned None for doc: {doc_sentences}")
        return None  # skip this document

    return annotations[0]  # Penman notation

def main(source, i, model_used, original=False):

    # Parsing original text
    if original:
        # --- Input JSONL file ---
        data_path = "/home/common/nlp-bjt/setup_1/sentences_original.jsonl"

        # --- Output TXT files ---
        human_out_path = "/home/common/nlp-bjt/setup_1/human_amr_original.txt"
        llm_out_path   = f"/home/common/nlp-bjt/setup_1/{source}/{model_used}/amrs/llm_amr_original.txt"

        # Make sure directories exist
        os.makedirs(os.path.dirname(llm_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(human_out_path), exist_ok=True)

        # Load JSONL once
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]

        # Write human AMR only if it doesn't exist
        if not os.path.exists(human_out_path):
            with open(human_out_path, "w") as human_f:
                for line in data:
                    human_amr = parse(line["human"])
                    if human_amr is not None:
                        human_f.write(human_amr.strip() + "\n\n")

        # Write LLM AMR
        with open(llm_out_path, "w") as llm_f:
            for line in data:
                llm_amr = parse(line["llm"])
                if llm_amr is not None:
                    llm_f.write(llm_amr.strip() + "\n\n")

    else:
        # --- Input JSONL file ---
        data_path = f"/home/common/nlp-bjt/setup_1/{source}/{model_used}/data/sentences_{i}.jsonl"
        
        # --- Output TXT files ---
        human_out_path = f"/home/common/nlp-bjt/setup_1/{source}/{model_used}/amrs/human_amr_{i}.txt"
        llm_out_path   = f"/home/common/nlp-bjt/setup_1/{source}/{model_used}/amrs/llm_amr_{i}.txt"
        os.makedirs(os.path.dirname(human_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(llm_out_path), exist_ok=True)
        
        # Load data
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
        
        # Write output (human and LLM)
        with open(human_out_path, "w") as human_f, \
             open(llm_out_path, "w") as llm_f:

            for line in data:
                human_amr = parse(line["human"])
                llm_amr   = parse(line["llm"])

                # Write each AMR with a blank line separator
                if human_amr != None and llm_amr != None:
                    human_f.write(human_amr.strip() + "\n\n")
                    llm_f.write(llm_amr.strip() + "\n\n")

    print("Done! Saved to:", human_out_path, "and", llm_out_path)

# --- Execution ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python parsing.py <data_source> <iteration> <rephraser_name> <--original>")
        sys.exit(1)
    source_name = sys.argv[1]
    iteration = int(sys.argv[2])
    model_used = sys.argv[3]
    original = "--original" in sys.argv
    main(source_name, iteration, model_used, original)
