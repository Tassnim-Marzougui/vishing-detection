import pandas as pd

from vishing_detection.data.audit import group_leakage
from vishing_detection.data.schema import MANIFEST_COLUMNS, validate_manifest


def _sample():
    return pd.DataFrame({
        "sample_id": ["DS07_A", "DS07_B", "DS07_C"],
        "audio_path": ["data/raw/x/a.wav", "data/raw/x/b.wav", "data/raw/x/c.wav"],
        "label": [1, 0, 1],
        "label_original": ["scam", "not_scam", "scam"],
        "language": ["ar-MSA"] * 3,
        "source_dataset": ["DS07_ASLC448"] * 3,
        "speaker_id": [""] * 3,
        "scenario_id": ["A", "B", "C"],
        "is_synthetic": [True] * 3,
        "duration_sec": [12.5, 20.0, 30.1],
        "sample_rate": [24000] * 3,
        "split": ["train", "validation", "test"],
        "license": ["A verifier"] * 3,
    })[MANIFEST_COLUMNS]


def test_valid_manifest():
    assert validate_manifest(_sample()) == []


def test_bad_label():
    df = _sample()
    df.loc[0, "label"] = 2
    assert validate_manifest(df) != []


def test_sample_id_prefix():
    df = _sample()
    df.loc[0, "sample_id"] = "A"
    assert validate_manifest(df) != []


def test_duplicate_sample_id():
    df = _sample()
    df.loc[1, "sample_id"] = "DS07_A"
    assert validate_manifest(df) != []


def test_group_leakage():
    df = _sample()
    assert group_leakage(df) == 0
    df.loc[1, "scenario_id"] = "A"
    assert group_leakage(df) == 1
