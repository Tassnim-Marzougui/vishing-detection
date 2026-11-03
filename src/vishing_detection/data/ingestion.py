from pathlib import Path

import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit

from vishing_detection.data.schema import MANIFEST_COLUMNS


def _grouped_split(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """70/15/15, groupe par scenario_id (GroupShuffleSplit, seed 42)."""
    g1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    _, rest = next(g1.split(df, groups=df["scenario_id"]))
    rest_df = df.iloc[rest]
    g2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    va, te = next(g2.split(rest_df, groups=rest_df["scenario_id"]))
    split = pd.Series("train", index=df.index)
    split.iloc[rest[va]] = "validation"
    split.iloc[rest[te]] = "test"
    return split


def ingest_ds07(raw_dir: str) -> pd.DataFrame:
    """DS07 (ASLC-448 + audio TTS) -> manifest commun."""
    raw = Path(raw_dir).resolve()
    root = raw.parents[2]  # racine du projet (data/raw/DS07_x -> racine)
    txt = pd.read_excel(raw / "aslc448.xlsx")
    wavs = {p.name: p for p in (raw / "audio").rglob("*.wav")}
    txt["wav"] = txt["conversation_id"].map(lambda c: wavs.get(f"{c}.wav"))
    txt = txt[txt["wav"].notna()].reset_index(drop=True)
    infos = [sf.info(p) for p in txt["wav"]]

    out = pd.DataFrame({
        "sample_id": "DS07_" + txt["conversation_id"].astype(str),
        "audio_path": [p.relative_to(root).as_posix() for p in txt["wav"]],
        "label": txt["label_binary"].astype(int),
        "label_original": txt["label"],
        "language": "ar-" + txt["dialect"].astype(str),
        "source_dataset": "DS07_ASLC448",
        "speaker_id": "",
        "scenario_id": txt["conversation_id"].astype(str),
        "is_synthetic": True,
        "duration_sec": [round(i.duration, 3) for i in infos],
        "sample_rate": [i.samplerate for i in infos],
        "license": "A verifier",
    })
    out["split"] = _grouped_split(out)
    return out[MANIFEST_COLUMNS]
