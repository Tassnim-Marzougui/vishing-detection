import pandas as pd


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["split"], df["label"])


def group_leakage(df: pd.DataFrame, group_col: str = "scenario_id") -> int:
    """Nombre de groupes presents dans plusieurs splits. Attendu: 0."""
    return int((df.groupby(group_col)["split"].nunique() > 1).sum())
