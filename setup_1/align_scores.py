import json
import argparse
from pathlib import Path


def align_multiple_jsonl(files, output_dir):
    """
    Align JSONL files by sent_num and keep only those that appear in ALL files.
    """
    data = {}

    # Load data from each file
    for f in files:
        d = {}
        with open(f, "r") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                d[obj["sent_num"]] = obj
        data[f] = d

    # Load GLOBAL sentence IDs
    with open("/home/common/nlp-bjt/setup_1/global_sent_nums.json") as f:
        global_common = set(json.load(f))

    # Compute intersection with this model’s files
    model_common = None
    for f in files:
        s = set(data[f].keys())
        model_common = s if model_common is None else (model_common & s)

    common = model_common & global_common

    print(f"Found {len(common)} matching sent_nums across ALL files!")

    # Create output directory if needed
    Path(output_dir).mkdir(exist_ok=True)

    # Write aligned outputs
    for f in files:
        out_path = Path(output_dir) / (Path(f).stem + "_aligned.jsonl")
        with open(out_path, "w") as out:
            for sn in sorted(common):
                out.write(json.dumps(data[f][sn], ensure_ascii=False) + "\n")
        print(f"→ Wrote aligned file: {out_path}")

    print("Alignment complete!")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True, help="data_source (e.g., llama_7B)")
    parser.add_argument("--rephraser", required=True, help="rephraser_model_name (e.g., Qwen3-30B)")

    args = parser.parse_args()

    # Construct 4 paths
    base = Path(f"/home/common/nlp-bjt/setup_1/{args.source}/{args.rephraser}/amrs")

    files = [
        base / f"human_1_score.jsonl",
        base / f"human_2_score.jsonl",
        base / f"llm_1_score.jsonl",
        base / f"llm_2_score.jsonl",
    ]

    # Validate existence
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print("ERROR: The following required files do NOT exist:")
        for m in missing:
            print("  -", m)
        return

    output_dir = f"/home/common/nlp-bjt/setup_1/{args.source}/{args.rephraser}/amrs/aligned"
    align_multiple_jsonl(files, output_dir)


if __name__ == "__main__":
    main()

