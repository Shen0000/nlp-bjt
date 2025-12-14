import json
import os
import random
import sys

SEED = 42

FILES = [
    "human_1_score_aligned.jsonl",
    "human_2_score_aligned.jsonl",
    "llm_1_score_aligned.jsonl",
    "llm_2_score_aligned.jsonl",
]

SPLIT_DIR = "/home/common/nlp-bjt/setup_1/sentence_splits"


# ----------------------------
# Utilities
# ----------------------------
def load_sent_nums(path):
    sent_nums = []
    with open(path) as f:
        for line in f:
            sent_nums.append(json.loads(line)["sent_num"])
    return sent_nums


def save_splits(train_ids, dev_ids, test_ids):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    for name, ids in [
        ("train", train_ids),
        ("dev", dev_ids),
        ("test", test_ids),
    ]:
        with open(os.path.join(SPLIT_DIR, f"{name}_sent_nums.json"), "w") as f:
            json.dump(sorted(ids), f)


def load_splits():
    splits = {}
    for name in ["train", "dev", "test"]:
        path = os.path.join(SPLIT_DIR, f"{name}_sent_nums.json")
        with open(path) as f:
            splits[name] = set(json.load(f))
    return splits["train"], splits["dev"], splits["test"]


# ----------------------------
# Main
# ----------------------------
def main(model, rephraser):
    base_dir = f"/home/common/nlp-bjt/setup_1/{model}/{rephraser}/amrs/aligned"
    out_dir = f"/home/common/nlp-bjt/setup_1/train_data/{rephraser}/{model}"
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------
    # 1. Load & validate sentence IDs
    # --------------------------------
    sent_num_lists = []
    for fname in FILES:
        path = os.path.join(base_dir, fname)
        sent_nums = load_sent_nums(path)
        sent_num_lists.append(sent_nums)

    lengths = [len(s) for s in sent_num_lists]
    if len(set(lengths)) != 1:
        raise ValueError(f"File length mismatch: {lengths}")

    num_sents = lengths[0]
    print(f"All files contain {num_sents} sentences")

    # Check sent_num consistency
    ref = sent_num_lists[0]
    for s in sent_num_lists[1:]:
        if s != ref:
            raise ValueError("sent_num mismatch or ordering difference across files")

    sent_nums = list(ref)

    # --------------------------------
    # 2. Create OR load splits
    # --------------------------------
    split_files_exist = all(
        os.path.exists(os.path.join(SPLIT_DIR, f"{name}_sent_nums.json"))
        for name in ["train", "dev", "test"]
    )

    if split_files_exist:
        print("Loading existing sentence splits")
        train_ids, dev_ids, test_ids = load_splits()
    else:
        print("Creating new sentence splits")
        random.seed(SEED)
        random.shuffle(sent_nums)

        n_train = int(0.8 * num_sents)
        n_dev   = int(0.1 * num_sents)

        train_ids = set(sent_nums[:n_train])
        dev_ids   = set(sent_nums[n_train:n_train + n_dev])
        test_ids  = set(sent_nums[n_train + n_dev:])

        save_splits(train_ids, dev_ids, test_ids)

    # Safety checks
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(test_ids)
    assert dev_ids.isdisjoint(test_ids)

    print(
        f"Train: {len(train_ids)} | "
        f"Dev: {len(dev_ids)} | "
        f"Test: {len(test_ids)}"
    )

    # --------------------------------
    # 3. Write split JSONLs
    # --------------------------------
    for fname in FILES:
        in_path = os.path.join(base_dir, fname)

        writers = {
            "train": open(os.path.join(out_dir, f"train_{fname}"), "w"),
            "dev":   open(os.path.join(out_dir, f"dev_{fname}"), "w"),
            "test":  open(os.path.join(out_dir, f"test_{fname}"), "w"),
        }

        with open(in_path) as f:
            for line in f:
                rec = json.loads(line)
                sid = rec["sent_num"]

                if sid in train_ids:
                    writers["train"].write(json.dumps(rec) + "\n")
                elif sid in dev_ids:
                    writers["dev"].write(json.dumps(rec) + "\n")
                elif sid in test_ids:
                    writers["test"].write(json.dumps(rec) + "\n")

        for w in writers.values():
            w.close()

    print(f"Split files written to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_sentence_splits.py <model> <rephraser>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])

