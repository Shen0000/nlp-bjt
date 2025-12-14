import json
import spacy
import os

"""
This function splits a multi-sentence string into a list of sentences (for the parser)

For example:
Input: "Thumbs up. Five stars. You'll love it. It's very easy to use. It will never break!"
Output: ["Thumbs up.", "Five stars.", "You'll love it.", "It's very easy to use.", "It will never break!"]
"""

nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    """Split a paragraph into sentences using SpaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def main(model_used, i):
    # --- Input file ---
    data_path = "/home/common/nlp-bjt/setup_2/{}/data/rephrased_{}.jsonl".format(model_used, i)
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    # --- Output file ---
    out_path = "/home/common/nlp-bjt/setup_2/{}/data/sentences_{}.jsonl".format(model_used, i)

    with open(out_path, "w") as out_f:
        for line in data:
            human_sent = split_sentences(line["human"])
            llm_sent   = split_sentences(line["llm"])

            out_obj = {
                "human": human_sent,
                "llm": llm_sent
            }
            out_f.write(json.dumps(out_obj) + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sentence_parser.py <model used> <iteration>")
        sys.exit(1)
    model_used = sys.argv[1]
    iteration = int(sys.argv[2])
    main(model_used, iteration)