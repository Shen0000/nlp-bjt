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
        os.path.join(base_dir, "{}_human_1_score.jsonl".format(split)),
        os.path.join(base_dir, "{}_human_2_score.jsonl".format(split)),
        label=0
    )
    X_l, y_l = load_pair(
        os.path.join(base_dir, "{}_llm_1_score.jsonl".format(split)),
        os.path.join(base_dir, "{}_llm_2_score.jsonl".format(split)),
        label=1
    )
    return X_h + X_l, y_h + y_l

def eval_split(name, clf, X, y):
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]  # positive class = LLM
    print("\n{} metrics:".format(name))
    print("  F1:      {:.4f}".format(f1_score(y, preds)))
    print("  ROC-AUC: {:.4f}".format(roc_auc_score(y, probs)))
    print("  PR-AUC:  {:.4f}".format(average_precision_score(y, probs)))
    if name.lower() == "test":
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, preds))

def main(model):
    base = "/home/common/nlp-bjt/setup_2/train_data/{}".format(model)

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

    save_dir = "trained_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out_path = os.path.join(save_dir, "xgboost_{}.joblib".format(model))
    joblib.dump(clf, out_path)
    print("\nSaved model to {}".format(out_path))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_xg.py <model_name>")
        exit(1)
    main(sys.argv[1])