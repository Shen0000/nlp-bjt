import json
import os

def convert_to_jsonl(source, method):
    # Input and output paths
    input_path = f"/home/common/nlp-bjt/news_data/data/{source}_{method}_filtered.json"
    output_path = f"/home/common/nlp-bjt/news_data/data/{source}_{method}_filtered.jsonl"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load JSON array
    with open(input_path, "r") as f:
        data = json.load(f)

    # Write JSONL (one object per line)
    with open(output_path, "w") as out_f:
        for item in data:
            out_f.write(json.dumps(item) + "\n")

    print(f"Converted {len(data)} entries from {input_path} → {output_path}")

# --- Execution ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python json_to_jsonl.py <source> <method>")
        sys.exit(1)

    source_name = sys.argv[1]
    method_name = sys.argv[2]

    convert_to_jsonl(source_name, method_name)
