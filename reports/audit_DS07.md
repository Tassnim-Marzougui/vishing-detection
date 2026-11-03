# Audit DS07 — ASLC-448 (conversations arabes, version audio)

## Accès
- Source texte : ASLC-448 (Mendeley Data). Fichier local : `data/raw/DS07_aslc448/aslc448.xlsx`
- Audio : 422 WAV générés par TTS par l'équipe (dossier `DATASET/audio`, métadonnées `audio_metadata.csv`)
- DOI / MD5 : à compléter

## Licence
- À vérifier sur la page Mendeley du dataset avant toute intégration (colonne `license` du manifest).

## Volumétrie
- Texte : 448 conversations, 9 dialectes (MSA, Egyptian, Gulf, Jordanian, Saudi, Yemeni, Sudanese, Iraqi, Syrian), 22 catégories
- Audio : 422 WAV, durée 12,6 à 55,9 s (moyenne 27,3 s). Dialecte Yemeni : aucun audio (26 conversations).
- Format / fréquence d'échantillonnage : à compléter (cellule 7 de 01_eda.ipynb)

## Labels
- `label` : scam / not_scam ; `label_binary` : 1 = scam, 0 = not_scam
- Audio : scam 233, not_scam 189 (majorité 55,3 %). Deux classes bien présentes.
- Catégories : scam et légitime sont totalement séparées par `category` (aucun recouvrement).

## Nature
- Synthétique : texte généré, audio TTS (voix unique ou peu variée). Pas d'appels réels.
- Identifiants : `conversation_id` (1 audio par conversation). Pas d'identifiant de locuteur.

## Doublons / manquants
- 0 doublon exact, 0 valeur manquante dans le fichier texte.
- 26 conversations sans audio (Yemeni).

## Raccourcis détectés (EDA)
- Règle « durée > 18,8 s => scam » : accuracy test 0,729 (majorité 0,553).
- Règle « word_count > 24 » : 0,682 ; « conversation_length > 152 » : 0,647.
- Tout modèle doit battre nettement 0,729. La segmentation en fenêtres de 10 s neutralise le raccourci durée.

## Limites
- Corpus synthétique à voix unique : les scores sont gonflés (cf. guide, section 7).
- Pas de locuteur : split groupé par `conversation_id`.
- Pas de `n_turns` d'audit (baseline n_turns indisponible).
- Pas de corpus externe : test « cross-corpus » remplacé par leave-one-dialect-out.

## Compléments (étape 1)
- Format audio : WAV, 24 kHz (une seule fréquence d'échantillonnage dans le manifest).
- Durée totale : environ 11 527 s (3 h 12 min), moyenne 27,3 s, min 12,6 s, max 55,9 s.
- DOI / MD5 : à compléter (page Mendeley du dataset).
- Licence : à vérifier sur Mendeley avant intégration (colonne license du manifest = "A verifier").
- Contrôle de sécurité avant dvc push : aucun matériel sensible attendu (conversations synthétiques); à confirmer avec la licence.
