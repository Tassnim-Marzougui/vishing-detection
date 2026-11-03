# 0002 - Seuil de decision choisi sur la validation

## Contexte
Le seuil fixe a 0,5 donne beaucoup de fausses alertes sur DS07 (matrice de confusion : 22 FP pour 3 FN sur 64 appels de test).
Les segments des scams sont plus nombreux (appels plus longs), ce qui desequilibre l'entrainement.

## Decision
evaluate.py calcule en plus le seuil qui maximise l'accuracy equilibree sur le split validation
(au niveau appel, pour chaque agregation) et l'applique au test. Le seuil 0,5 reste rapporte.

## Consequences
- Le seuil ne regarde jamais le test.
- Les rapports contiennent deux jeux de metriques (seuil 0,5 et seuil validation).
- Le jeu de validation est petit (63 appels) : le seuil reste instable.
