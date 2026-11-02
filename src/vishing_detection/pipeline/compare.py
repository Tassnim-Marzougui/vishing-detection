import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from vishing_detection.pipeline.evaluate import metrics
from vishing_detection.pipeline.train import load_data


def clean(t):
    t = re.sub(r"[\u064B-\u0652\u0640]", "", str(t))
    return re.sub(r"\s+", " ", t).strip()


def calls_table():
    m = pd.read_csv("data/processed/manifest.csv", keep_default_na=False)
    txt = pd.read_excel("data/raw/DS07_aslc448/aslc448.xlsx")[["conversation_id", "full_conversation"]]
    m = m.merge(txt, left_on="scenario_id", right_on="conversation_id")
    m["text"] = m["full_conversation"].map(clean)
    return m.set_index("sample_id")


def acoustic_fit(tr_seg, cols, seed=42):
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    ).fit(tr_seg[cols], tr_seg["label"])


def acoustic_call_scores(model, seg, cols):
    d = seg.copy()
    d["score"] = model.predict_proba(d[cols])[:, 1]
    return d.groupby("sample_id")["score"].mean()


def text_fit(df, seed=42):
    m = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    return m.fit(df["text"], df["label"])


def text_call_scores(model, df):
    return pd.Series(model.predict_proba(df["text"])[:, 1], index=df.index)


def best_threshold(y, s):
    grid = np.linspace(0.2, 0.8, 61)
    bal = [(((s >= t)[y == 1]).mean() + ((s < t)[y == 0]).mean()) / 2 for t in grid]
    return float(grid[int(np.argmax(bal))])


def fmt(x):
    return "  n/a" if x is None else f"{x:.2f}"


def main():
    data, cols = load_data()
    calls = calls_table()
    res = {"split_officiel": {}, "lodo": {"acoustique_rf": {}, "texte_tfidf": {}}}

    tr_s, va_s, te_s = (data[data["split"] == s] for s in ("train", "validation", "test"))
    tr_c, va_c, te_c = (calls[calls["split"] == s] for s in ("train", "validation", "test"))
    rf = acoustic_fit(tr_s, cols)
    tm = text_fit(tr_c)
    pairs = {
        "acoustique_rf": (acoustic_call_scores(rf, va_s, cols), acoustic_call_scores(rf, te_s, cols)),
        "texte_tfidf": (text_call_scores(tm, va_c), text_call_scores(tm, te_c)),
    }
    for name, (sv, st) in pairs.items():
        yv = calls.loc[sv.index, "label"].to_numpy()
        yt = calls.loc[st.index, "label"].to_numpy()
        thr = best_threshold(yv, sv.to_numpy())
        res["split_officiel"][name] = {
            "seuil_val": thr,
            "seuil_0.5": metrics(yt, st.to_numpy(), 0.5),
            "seuil_valide": metrics(yt, st.to_numpy(), thr),
        }

    langs = sorted(calls["language"].unique())
    for L in langs:
        te_cl, tr_cl = calls[calls["language"] == L], calls[calls["language"] != L]
        te_sl, tr_sl = data[data["language"] == L], data[data["language"] != L]
        sa = acoustic_call_scores(acoustic_fit(tr_sl, cols), te_sl, cols)
        st = text_call_scores(text_fit(tr_cl), te_cl)
        for name, s in (("acoustique_rf", sa), ("texte_tfidf", st)):
            res["lodo"][name][L] = metrics(calls.loc[s.index, "label"].to_numpy(), s.to_numpy(), 0.5)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/compare.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print("== Split officiel (test, niveau appel, 64 appels) ==")
    for name, r in res["split_officiel"].items():
        for k in ("seuil_0.5", "seuil_valide"):
            m = r[k]
            print(f"{name:14s} {k:13s} acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={fmt(m['roc_auc'])} FN={m['fn']} FP={m['fp']}")
        print(f"{'':14s} seuil choisi sur validation = {r['seuil_val']:.2f}")
    print("\n== Leave-one-dialect-out (seuil 0.5) ==")
    for L in langs:
        a, t = res["lodo"]["acoustique_rf"][L], res["lodo"]["texte_tfidf"][L]
        print(f"{L:14s} n={a['n']:3d} | audio acc={a['accuracy']:.2f} auc={fmt(a['roc_auc'])} | texte acc={t['accuracy']:.2f} auc={fmt(t['roc_auc'])}")
    for name in ("acoustique_rf", "texte_tfidf"):
        accs = [res["lodo"][name][L]["accuracy"] for L in langs]
        print(f"moyenne LODO {name}: acc={np.mean(accs):.3f}")


if __name__ == "__main__":
    main()
