import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from vishing_detection.pipeline.train import load_data

THRESHOLD = 0.5


def agg(s, how, k=3):
    s = np.sort(np.asarray(s))[::-1]
    if how == "max":
        return float(s[0])
    if how == "topk":
        return float(s[:k].mean())
    return float(s.mean())


def metrics(y, score, thr=THRESHOLD):
    y = np.asarray(y)
    score = np.asarray(score)
    pred = (score >= thr).astype(int)
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if len(set(y.tolist())) > 1 else None,
        "fn": fn,
        "fp": fp,
        "fnr": float(fn / max(int((y == 1).sum()), 1)),
        "fpr": float(fp / max(int((y == 0).sum()), 1)),
    }


def main():
    data, cols = load_data()
    model = joblib.load("models/latest/model.joblib")
    te = data[data["split"] == "test"].copy()
    te["score"] = model.predict_proba(te[cols])[:, 1]

    res = {"threshold": THRESHOLD, "niveau": "appel (agregation des segments)"}
    for how in ("mean", "max", "topk"):
        calls = te.groupby("sample_id").agg(
            score=("score", lambda s, h=how: agg(s, h)),
            label=("label", "first"),
            language=("language", "first"),
        ).reset_index()
        res[how] = metrics(calls["label"], calls["score"])
        if how == "mean":
            res["par_langue_mean"] = {
                lang: metrics(g["label"], g["score"]) for lang, g in calls.groupby("language")
            }

    with open("reports/baselines.json", encoding="utf-8") as f:
        base = json.load(f)
    res["plancher_majorite_appels"] = base["appels"]["majority_acc_test"]
    res["plancher_duree_appels"] = base["appels"]["duration_sec"]["test_acc"]

    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print("Planchers (appels): majorite =", res["plancher_majorite_appels"], "| duree =", res["plancher_duree_appels"])
    for how in ("mean", "max", "topk"):
        m = res[how]
        print(f"{how:5s} acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']} FN={m['fn']} FP={m['fp']}")
    print("\nPar langue (mean):")
    for lang, m in res["par_langue_mean"].items():
        print(f"  {lang:16s} n={m['n']:2d} acc={m['accuracy']:.2f} f1={m['f1']:.2f}")


if __name__ == "__main__":
    main()
