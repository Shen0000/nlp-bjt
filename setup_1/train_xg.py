import json
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from xgboost import XGBClassifier
import joblib
import numpy as np

FILES = ["human_1", "human_2", "llm_1", "llm_2"]

def load_pair(file1, file2, label):
    """Load two JSONL files and return X, y."""
    d1, d2 = {}, {}
    with open(file1) as f:
        for line in f:
            rec = json.loads(line)
            d1[rec["sent_num"]] = rec
    with open(file2) as f:
        for line in f:
            rec = json.loads(line)
            d2[rec["sent_num"]] = rec

    X, y = [], []
    for sent_num in sorted(d1.keys()):
        if sent_num not in d2:
            continue
        X.append([
            d1[sent_num].get("precision", 0.0),
            d2[sent_num].get("precision", 0.0),
            d1[sent_num].get("recall", 0.0),
            d2[sent_num].get("recall", 0.0),
        ])
        y.append(label)
    return X, y

def load_split(base_dir, split):
    """Load X, y directly from pre-made split files in train_data."""
    X_h, y_h = load_pair(
        os.path.join(base_dir, f"{split}_human_1_score_aligned.jsonl"),
        os.path.join(base_dir, f"{split}_human_2_score_aligned.jsonl"),
        label=0
    )
    X_l, y_l = load_pair(
        os.path.join(base_dir, f"{split}_llm_1_score_aligned.jsonl"),
        os.path.join(base_dir, f"{split}_llm_2_score_aligned.jsonl"),
        label=1
    )
    return X_h + X_l, y_h + y_l

def eval_split(name, clf, X, y):
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]  # positive class = LLM
    print(f"\n{name} metrics:")
    print(f"  F1:      {f1_score(y, preds):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y, probs):.4f}")
    print(f"  PR-AUC:  {average_precision_score(y, probs):.4f}")
    if name.lower() == "test":
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, preds))

def main(model, rephraser):
    base = f"/home/common/nlp-bjt/setup_1/train_data/{rephraser}/{model}"

    X_train, y_train = load_split(base, "train")
    X_dev,   y_dev   = load_split(base, "dev")
    X_test,  y_test  = load_split(base, "test")

    X_train = np.array(X_train)
    X_dev = np.array(X_dev)
    X_test = np.array(X_test)

    print("Train: {}  Dev: {}  Test: {}".format(len(X_train), len(X_dev), len(X_test)))

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=8,
        use_label_encoder=False,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate
    eval_split("Dev", clf, X_dev, y_dev)
    eval_split("Test", clf, X_test, y_test)

    save_dir = f"trained_models/{rephraser}"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"decision_tree_{model}.joblib")
    joblib.dump(clf, out_path)
    print(f"\nSaved model to {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python train_xg.py <model_name> <rephraser_name>")
        exit(1)
    main(sys.argv[1], sys.argv[2])
