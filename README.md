# Vishing Detection - DS07 (ASLC-448, version audio TTS)

Projet academique (Tek-Up). Dataset DS07 : 448 conversations arabes (9 dialectes),
422 avec audio TTS (dialecte Yemeni sans audio). Suit le guide d'equipe.

## Rejouer le pipeline
@'
# Vishing Detection - DS07 (ASLC-448, version audio TTS)

Projet academique (Tek-Up). Dataset DS07 : 448 conversations arabes (9 dialectes),
422 avec audio TTS (dialecte Yemeni sans audio). Suit le guide d'equipe.

## Rejouer le pipeline

    python -m venv .venv
    .venv\Scripts\activate
    pip install pandas scikit-learn openpyxl soundfile librosa scipy mlflow pytest pyyaml dvc
    pip install -e .
    pytest -q
    python make_manifest_ds07.py
    copy data\metadata\manifests\DS07_manifest.csv data\metadata\audio_manifest.csv
    python -m vishing_detection.pipeline.run_stage validate
    python -m vishing_detection.pipeline.run_stage prepare
    python -m vishing_detection.pipeline.run_stage baselines
    python -m vishing_detection.pipeline.features
    python -m vishing_detection.pipeline.train
    python -m vishing_detection.pipeline.evaluate
    python -m vishing_detection.pipeline.compare
    python -m vishing_detection.pipeline.baseline_lodo

## Resultats (corpus synthetique, intra-corpus)

- Planchers (test, 64 appels): majorite 0.50, regle sur la duree 0.625.
- Leave-one-dialect-out (422 appels): regle duree 0.637.
- Audio (MFCC/RMS/ZCR + RF): 0.69 (test), 0.71 en LODO (+7 points sur la duree).
- Texte (TF-IDF + LogReg): 1.00 / 0.99. Non generalisable : scam et legitime
  sont separes a 100 % par la categorie dans ce corpus.

## Ecarts par rapport au guide

- Spark remplace par pandas (422 appels) ; sorties en CSV (pas parquet).
- Pas d'embeddings wav2vec2 (GPU Kaggle) : a faire.
- Pas de baseline n_turns (DS04 uniquement).
- Pas de test cross-corpus : remplace par leave-one-dialect-out.
- Licence a verifier (colonne license = "A verifier").
- Audio TTS a voix unique : scores gonfles.
