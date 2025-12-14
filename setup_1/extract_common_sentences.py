import json
from pathlib import Path
import argparse

def load_sent_nums(path):
    s = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                s.add(json.loads(line)["sent_num"])
    return s


def main(models, rephraser):
    base_root = Path("/home/common/nlp-bjt/setup_1")

    global_common = None

    for model in models:
        base = base_root / model / rephraser / "amrs"

        files = [
            base / "human_1_score.jsonl",
            base / "human_2_score.jsonl",
            base / "llm_1_score.jsonl",
            base / "llm_2_score.jsonl",
        ]

        model_common = None
        for f in files:
            s = load_sent_nums(f)
            model_common = s if model_common is None else (model_common & s)

        print(f"{model}: {len(model_common)} sentences")

        global_common = (
            model_common if global_common is None
            else (global_common & model_common)
        )

    print(f"\nGLOBAL common sentences: {len(global_common)}")

    out_path = base_root / "global_sent_nums.json"
    with open(out_path, "w") as f:
        json.dump(sorted(global_common), f)

    print(f"Saved global sent_nums to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--rephraser", required=True)
    args = parser.parse_args()

    main(args.models, args.rephraser)

