import json
import os

def filter_json(source, method):
    input_path = f"/home/common/nlp-bjt/news_data/data/{source}_{method}.json"
    output_path = f"/home/common/nlp-bjt/news_data/data/{source}_{method}_filtered.json"

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    with open(input_path, "r") as f:
        data = json.load(f)

    filtered = [
        d for d in data
        if d.get("human", "").strip() != "" and d.get("llm", "").strip() != ""
    ]

    removed = len(data) - len(filtered)
    print(f"Removed {removed} empty entries out of {len(data)} total.")

    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Saved filtered file to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python filter.py <source> <method>")
        sys.exit(1)

    source_name = sys.argv[1]
    method_name = sys.argv[2]

    filter_json(source_name, method_name)
