from math import gcd
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
N_MFCC = 20


def load_segment(path, start, end):
    info = sf.info(path)
    sr0 = info.samplerate
    y, _ = sf.read(path, start=int(start * sr0), stop=int(end * sr0), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr0 != SR:
        g = gcd(sr0, SR)
        y = resample_poly(y, SR // g, sr0 // g)
    return y


def acoustic_features(y, sr=SR, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    f = {}
    for i in range(n_mfcc):
        f[f"mfcc{i + 1}_mean"] = float(mfcc[i].mean())
        f[f"mfcc{i + 1}_std"] = float(mfcc[i].std())
    f["rms_mean"] = float(rms.mean())
    f["rms_std"] = float(rms.std())
    f["zcr_mean"] = float(zcr.mean())
    db = 20 * np.log10(rms + 1e-10)
    f["silence_ratio"] = float((db < -40).mean())
    return f


def features_handcrafted():
    seg = pd.read_csv("data/processed/segments.csv")
    rows = []
    for k, r in enumerate(seg.itertuples(), 1):
        y = load_segment(r.audio_path, r.start_sec, r.end_sec)
        rows.append({"segment_id": r.segment_id, "sample_id": r.sample_id, **acoustic_features(y)})
        if k % 200 == 0:
            print(f"{k}/{len(seg)}")
    out = pd.DataFrame(rows)
    Path("data/features").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/features/handcrafted.csv", index=False)
    print("features:", out.shape, "(attendu: 2 + 44 colonnes)")


if __name__ == "__main__":
    features_handcrafted()
