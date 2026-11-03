from vishing_detection.data.ingestion import ingest_ds07
from vishing_detection.data.schema import validate_manifest
from vishing_detection.data.audit import class_balance, group_leakage

df = ingest_ds07("data/raw/DS07_aslc448")
errs = validate_manifest(df)
assert errs == [], errs
print("shape:", df.shape)
print("sample_rate:", df["sample_rate"].unique(), "| duree min/max:", df["duration_sec"].min(), df["duration_sec"].max())
print(class_balance(df))
leak = group_leakage(df)
print("group_leakage:", leak)
assert leak == 0
df.to_csv("data/metadata/manifests/DS07_manifest.csv", index=False)
print("OK -> data/metadata/manifests/DS07_manifest.csv")
