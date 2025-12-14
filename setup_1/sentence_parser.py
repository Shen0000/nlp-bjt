import json
import spacy
import os

# --- Load SpaCy model once ---
nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    """Split a paragraph into sentences using SpaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def main(source, i, model_used, original=False):
    if original:
        data_path = f"/home/common/nlp-bjt/news_data/data/sample/{source}_sampled.jsonl"

        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]

        # --- Output file ---
        out_path = f"/home/common/nlp-bjt/setup_1/sentences_original.jsonl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True) 

        with open(out_path, "w") as out_f:
            for line in data:
                human_sent = split_sentences(line["human"])
                llm_sent   = split_sentences(line["llm"])

                out_obj = {
                    "human": human_sent,
                    "llm": llm_sent
                }
                out_f.write(json.dumps(out_obj) + "\n")
    else:
        # --- Input file ---
        data_path = f"/home/common/nlp-bjt/setup_1/{source}/{model_used}/data/rephrased_{i}.jsonl"
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]

        # --- Output file ---
        out_path = f"/home/common/nlp-bjt/setup_1/{source}/{model_used}/data/sentences_{i}.jsonl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w") as out_f:
            for line in data:
                human_sent = split_sentences(line["human"])
                llm_sent   = split_sentences(line["llm"])

                out_obj = {
                    "human": human_sent,
                    "llm": llm_sent
                }
                out_f.write(json.dumps(out_obj) + "\n")

    print("Done! Saved to:", out_path)

# --- Example usage ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python sentence_parser.py <data_source> <iteration> <rephraser_name> <--original>")
        sys.exit(1)
    source_name = sys.argv[1]
    iteration = int(sys.argv[2])
    model_used = sys.argv[3]
    original = "--original" in sys.argv
    main(source_name,iteration, model_used, original)
