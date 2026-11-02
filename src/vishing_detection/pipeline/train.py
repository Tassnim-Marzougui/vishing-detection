import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FORBIDDEN = {
    "n_turns", "speaker_role", "theme", "request_type", "label", "label_original",
    "category", "dialect", "urgency_score", "sensitive_info_requests",
    "financial_pressure_score", "threat_score", "impersonation_score",
    "conversation_length", "word_count", "label_binary", "duration_sec", "seg_duration",
}


def load_data():
    seg = pd.read_csv("data/processed/segments.csv")
    feats = pd.read_csv("data/features/handcrafted.csv").drop(columns=["sample_id"])
    cols = [c for c in feats.columns if c != "segment_id"]
    bad = FORBIDDEN & set(cols)
    assert not bad, f"colonnes d'annotation interdites dans les features: {bad}"
    return seg.merge(feats, on="segment_id"), cols


def make_model(head, seed):
    if head == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


def main():
    with open("params.yaml", encoding="utf-8") as f:
        seed = yaml.safe_load(f)["seed"]
    data, cols = load_data()
    tr = data[data["split"] == "train"]
    va = data[data["split"] == "validation"]

    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("vishing-detection")
    except ImportError:
        mlflow = None

    best = None
    for head in ("logistic", "rf"):
        model = make_model(head, seed).fit(tr[cols], tr["label"])
        auc = roc_auc_score(va["label"], model.predict_proba(va[cols])[:, 1])
        print(f"{head}: AUC validation (segment) = {auc:.3f}")
        if mlflow:
            with mlflow.start_run(run_name=f"acoustic_{head}"):
                mlflow.set_tags({"train_corpus": "DS07", "author": "Tassnim", "type": "candidate"})
                mlflow.log_params({"head": head, "seed": seed, "n_features": len(cols)})
                mlflow.log_metric("val_auc_segment", auc)
        if best is None or auc > best[0]:
            best = (auc, head, model)

    Path("models/latest").mkdir(parents=True, exist_ok=True)
    joblib.dump(best[2], "models/latest/model.joblib")
    with open("models/latest/meta.json", "w", encoding="utf-8") as f:
        json.dump({"head": best[1], "val_auc_segment": best[0], "seed": seed, "features": cols}, f, indent=2)
    print("modele retenu:", best[1])


if __name__ == "__main__":
    main()
