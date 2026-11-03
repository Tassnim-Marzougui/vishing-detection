# 0001 - Pandas au lieu de Spark pour DS07

## Contexte
Le guide decrit validate, prepare et features_handcrafted avec PySpark (Java 17+, Python 3.11/3.12).
DS07 contient 422 appels (1676 segments) et l'environnement local tourne sous Python 3.14.

## Decision
Ces etapes sont implementees en pandas (run_stage.py, features.py) avec le meme schema d'entree et de sortie
(5 drapeaux de validation, fenetres 10 s / hop 5 s, 44 features). Les sorties sont des CSV.

## Consequences
- Aucune dependance a Java ni a Spark pour rejouer DS07.
- Migration possible vers PySpark sans changer les fonctions metier.
- Pour un corpus beaucoup plus gros, repasser sur Spark.
