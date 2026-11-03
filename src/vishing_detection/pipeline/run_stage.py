import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MANIFEST = Path("data/metadata/audio_manifest.csv")
INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")


def _params():
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read(path):
    return pd.read_csv(path, keep_default_na=False)


def validate():
    df = _read(MANIFEST)
    req = ["sample_id", "audio_path", "label", "duration_sec", "split"]
    q = pd.DataFrame({"sample_id": df["sample_id"]})
    q["duplicate_id"] = df["sample_id"].duplicated(keep=False)
    q["missing_field"] = (df[req].astype(str) == "").any(axis=1)
    dur = pd.to_numeric(df["duration_sec"], errors="coerce")
    q["bad_duration"] = ~(dur > 0)
    q["bad_split"] = ~df["split"].isin(["train", "validation", "test"])
    q["bad_label"] = ~df["label"].isin([0, 1])
    flags = ["duplicate_id", "missing_field", "bad_duration", "bad_split", "bad_label"]
    q["ok"] = ~q[flags].any(axis=1)
    INTERIM.mkdir(parents=True, exist_ok=True)
    q.to_csv(INTERIM / "quality.csv", index=False)
    print(f"validate: {int(q['ok'].sum())}/{len(q)} echantillons valides")
    print(q[flags].sum().to_string())


def prepare():
    p = _params()["segmentation"]
    win, hop = p["window_sec"], p["hop_sec"]
    df = _read(MANIFEST)
    q = _read(INTERIM / "quality.csv")
    df = df[df["sample_id"].isin(q.loc[q["ok"], "sample_id"])].copy()

    n = df.groupby("scenario_id")["split"].nunique()
    if (n > 1).any():
        raise ValueError(f"groupes presents dans plusieurs splits: {list(n[n > 1].index)}")

    rows = []
    for r in df.itertuples():
        d = float(r.duration_sec)
        nseg = math.floor((d - win) / hop) + 1 if d > win else 1
        for i in range(nseg):
            s = i * hop
            e = min(s + win, d)
            rows.append({
                "segment_id": f"{r.sample_id}_{i:03d}",
                "sample_id": r.sample_id,
                "audio_path": r.audio_path,
                "start_sec": float(s),
                "end_sec": round(e, 3),
                "seg_duration": round(e - s, 3),
                "label": r.label,
                "split": r.split,
                "scenario_id": r.scenario_id,
                "language": r.language,
            })
    seg = pd.DataFrame(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "manifest.csv", index=False)
    seg.to_csv(PROCESSED / "segments.csv", index=False)
    print(f"prepare: {len(df)} appels -> {len(seg)} segments")
    print(seg.groupby(["split", "label"]).size().unstack())


def fit_threshold_rule(x, y):
    best = (0.0, None, 1)
    for t in np.unique(x):
        for sign in (1, -1):
            acc = ((sign * (x - t) > 0).astype(int) == y).mean()
            if acc > best[0]:
                best = (acc, t, sign)
    return best


def baseline_report(df, cols):
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    maj = int(tr["label"].mode()[0])
    out = {"majority_class": maj, "majority_acc_test": float((te["label"] == maj).mean()), "n_test": int(len(te))}
    for c in cols:
        acc, t, s = fit_threshold_rule(tr[c].to_numpy(float), tr["label"].to_numpy())
        pred = (s * (te[c].to_numpy(float) - t) > 0).astype(int)
        out[c] = {
            "threshold": float(t),
            "sign": int(s),
            "train_acc": float(acc),
            "test_acc": float((pred == te["label"].to_numpy()).mean()),
        }
    return out


def baselines():
    calls = _read(PROCESSED / "manifest.csv")
    seg = _read(PROCESSED / "segments.csv")
    res = {
        "appels": baseline_report(calls, ["duration_sec"]),
        "segments": baseline_report(seg, ["seg_duration"]),
        "limite": "n_turns indisponible (audit DS04 seulement): majorite + duree uniquement",
    }
    REPORTS.mkdir(exist_ok=True)
    import json
    with open(REPORTS / "baselines.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


STAGES = {"validate": validate, "prepare": prepare, "baselines": baselines}

if __name__ == "__main__":
    STAGES[sys.argv[1]]()
