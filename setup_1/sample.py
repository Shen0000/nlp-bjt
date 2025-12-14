import json
import random
import os

def sample_json(source, method, n_samples=1000, original=False):
    random.seed(42)

    if original:
        input_path = "/home/common/nlp-bjt/news_data/"
        raise NotImplementedError("Original=True behavior not implemented.")
    else:
        input_path = f"/home/common/nlp-bjt/news_data/data/{source}_{method}.json"
        output_path = f"/home/common/nlp-bjt/news_data/data/{source}_{method}_sample_tos.json"

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load full dataset
        with open(input_path, "r") as f:
            data = json.load(f)

        # Filter out entries with empty human or llm
        filtered = [
            d for d in data
            if d.get("human", "").strip() != "" and d.get("llm", "").strip() != ""
        ]

        num_removed = len(data) - len(filtered)
        print(f"Filtered out {num_removed} empty or incomplete entries.")

        if len(filtered) == 0:
            raise ValueError("No valid entries found after filtering!")

        # Adjust sample size if needed
        if n_samples > len(filtered):
            print(f"Requested {n_samples} samples, but only {len(filtered)} valid entries.")
            n_samples = len(filtered)

        sampled_data = random.sample(filtered, n_samples)

        # Save sampled JSON array
        with open(output_path, "w") as out_f:
            json.dump(sampled_data, out_f, indent=2)

        print(f"Saved {n_samples} samples to {output_path}")



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python sample.py <source> <method> [n_samples]")
        sys.exit(1)

    source_name = sys.argv[1]
    method_name = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    sample_json(source_name, method_name, n)
