import pandas as pd

MANIFEST_COLUMNS = [
    "sample_id", "audio_path", "label", "label_original", "language",
    "source_dataset", "speaker_id", "scenario_id", "is_synthetic",
    "duration_sec", "sample_rate", "split", "license",
]


def validate_manifest(df: pd.DataFrame) -> list:
    """Retourne une liste d'erreurs ([] = manifest valide)."""
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        return [f"colonnes manquantes: {missing}"]
    errors = []
    if df["sample_id"].duplicated().any():
        errors.append("sample_id dupliques")
    if not df["sample_id"].astype(str).str.match(r"^DS\d{2}_").all():
        errors.append("sample_id sans prefixe DSxx_")
    if not df["label"].isin([0, 1]).all():
        errors.append("label hors {0,1}")
    if not df["split"].isin(["train", "validation", "test"]).all():
        errors.append("split invalide")
    if not (df["duration_sec"] > 0).all():
        errors.append("duration_sec <= 0")
    if df["audio_path"].astype(str).str.contains("\\", regex=False).any():
        errors.append("audio_path contient des backslash (utiliser /)")
    return errors
