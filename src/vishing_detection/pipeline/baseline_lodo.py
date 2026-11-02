import numpy as np
import pandas as pd

from vishing_detection.pipeline.run_stage import fit_threshold_rule

m = pd.read_csv("data/processed/manifest.csv", keep_default_na=False)
accs = []
for L in sorted(m["language"].unique()):
    tr, te = m[m["language"] != L], m[m["language"] == L]
    _, t, s = fit_threshold_rule(tr["duration_sec"].to_numpy(float), tr["label"].to_numpy())
    pred = (s * (te["duration_sec"].to_numpy(float) - t) > 0).astype(int)
    a = float((pred == te["label"].to_numpy()).mean())
    accs.append(a)
    print(f"{L:14s} n={len(te):3d} duree acc={a:.2f}")
print(f"moyenne LODO duree: {np.mean(accs):.3f}")
