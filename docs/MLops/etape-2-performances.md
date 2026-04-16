---
layout: default
title: "Étape 2 — Tests de performance"
parent: "Monitoring / MLOps"
nav_order: 2
---



## Tests de performance du modèle sélectionné

K-fold stratified cross-validation est une technique d’évaluation de la performance d’un modèle de machine learning qui consiste à diviser le dataset en K sous-ensembles (ou "folds") de manière stratifiée, c’est-à-dire en respectant la distribution des classes dans chaque fold. Le modèle est ensuite entrainé K fois, chaque fois en utilisant K-1 folds pour l’entrainement et le fold restant pour le test. Les performances du modèle sont ensuite évaluées en calculant les métriques d’évaluation (précision, rappel, F1-score) pour chaque fold, et en calculant la moyenne de ces métriques sur les K folds.


![Kfold statifié](figures/Kfold_strat_all.png)



Rerun avec un split train/validation/test plus classique, pour voir si les résultats sont similaires à ceux obtenus avec la K-fold stratified cross-validation. Cela permettra de vérifier la robustesse des résultats et de s’assurer que le modèle sélectionné est performant sur un split plus classique du dataset.

Cette technique permet d’obtenir une estimation plus fiable de la performance du modèle, en réduisant le risque de surapprentissage (overfitting) et en fournissant une évaluation plus robuste de la performance du modèle sur des données non vues. En utilisant K-fold stratified cross-validation, j’ai pu évaluer les performances du modèle sélectionné de manière plus fiable, en prenant en compte la variabilité des données et en fournissant une évaluation plus robuste de la performance du modèle sur des données non vues.


![Kfold stratifié split 70-15-15](figures/Kfold_strat_all_split70_15_15.png)

# Run Summary (detailed)

| Metric                     | Score     | Interpretation |
|----------------------------|-----------|----------------|
| **Test Accuracy**          | **0.94627** | Très haute précision globale |
| **F1‑Score Macro**         | 0.94478   | Bon équilibre entre classes |
| **Precision Macro**        | 0.94521   | Peu de faux positifs |
| **Recall Macro**           | 0.94480   | Peu de faux négatifs |
| **Val–Test Gap**           | 0.00945   | Excellente généralisation |