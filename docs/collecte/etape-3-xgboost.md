---
layout: default
title: "Étape 3 — XGBoost multi-classe"
parent: "Collecte des données"
nav_order: 4
---

# 🧩 Étape 4 — XGBoost multi‑classe sur embeddings

Pour le filtrage automatique, j'ai mis en place un classifieur XGBoost basé sur les embeddings. L'objectif n'était pas de classifier les images avec précision, mais de déterminer si une nouvelle image est cohérente avec les patterns visuels des 8 000 images filtrées manuellement — même si elle n'appartient pas exactement à la bonne classe. J'ai donc utilisé un classifieur multi-classe (35 classes) plutôt qu'un classifieur binaire par classe. Si la confiance de la prédiction dépassait un seuil défini, les distances aux 4 clusters calculés précédemment étaient alors utilisées pour filtrer les images incohérentes.

Pourquoi XGBoost et cette approche ?

- il gère très bien les données tabulaires (comme les embeddings),
- il est robuste aux classes déséquilibrées,
- il est rapide à entraîner,
- il permet de tester rapidement la qualité des embeddings,
- il fournit des probabilités de prédiction utiles pour le filtrage.

Ce modèle n’est pas destiné à remplacer les modèles finaux qui seront fine‑tuned pour classifier les plantes, mais il constitue un excellent outil d’analyse et de pré‑filtrage pour pouvoir enrichir le dataset avec des images de qualité.

Voici la matrice de confusion obtenue pour ce modèle de classification multi‑classe basé sur les embeddings pour chaque classe, ainsi que les différentes métriques d’évaluation (précision, rappel, F1‑score) pour chaque classe :

| Classe | Precision | Recall | F1‑Score | Support |
|-------:|-----------:|--------:|---------:|---------:|
| 0  | 0.86 | 0.86 | 0.86 | 37 |
| 1  | 0.59 | 0.69 | 0.64 | 32 |
| 2  | 0.71 | 0.60 | 0.65 | 50 |
| 3  | 0.96 | 0.96 | 0.96 | 51 |
| 4  | 0.68 | 0.54 | 0.60 | 52 |
| 5  | 0.68 | 0.78 | 0.73 | 51 |
| 6  | 0.70 | 0.51 | 0.59 | 37 |
| 7  | 0.85 | 0.85 | 0.85 | 41 |
| 8  | 0.80 | 0.83 | 0.82 | 64 |
| 9  | 0.79 | 0.69 | 0.74 | 39 |
| 10 | 0.89 | 0.85 | 0.87 | 46 |
| 11 | 0.75 | 0.77 | 0.76 | 60 |
| 12 | 0.93 | 0.91 | 0.92 | 44 |
| 13 | 0.73 | 0.81 | 0.77 | 47 |
| 14 | 0.90 | 0.93 | 0.92 | 60 |
| 15 | 0.73 | 0.78 | 0.75 | 58 |
| 16 | 0.84 | 0.97 | 0.90 | 32 |
| 17 | 0.89 | 0.61 | 0.72 | 28 |
| 18 | 0.90 | 0.76 | 0.83 | 25 |
| 19 | 0.85 | 0.92 | 0.89 | 38 |
| 20 | 0.63 | 0.70 | 0.67 | 47 |
| 21 | 0.62 | 0.70 | 0.66 | 50 |
| 22 | 0.86 | 0.86 | 0.86 | 44 |
| 23 | 0.88 | 0.88 | 0.88 | 64 |
| 24 | 0.66 | 0.71 | 0.68 | 41 |
| 25 | 0.82 | 0.82 | 0.82 | 49 |
| 26 | 0.66 | 0.51 | 0.58 | 37 |
| 27 | 0.44 | 0.24 | 0.31 | 17 |
| 28 | 0.90 | 0.98 | 0.94 | 63 |
| 29 | 0.84 | 0.85 | 0.85 | 48 |
| 30 | 0.56 | 0.60 | 0.58 | 53 |
| 31 | 0.77 | 0.91 | 0.84 | 56 |
| 32 | 0.94 | 0.94 | 0.94 | 49 |
| 33 | 0.82 | 0.93 | 0.87 | 29 |
| 34 | 0.91 | 0.79 | 0.84 | 62 |

<br><br>

Voici le résumé global des métriques d’évaluation pour le modèle de classification multi‑classe basé sur les embeddings :

| Metric        | Precision | Recall | F1‑Score | Support |
|---------------|-----------|--------|----------|---------|
| Accuracy      | —         | —      | 0.79     | 1601    |
| Macro Avg     | 0.78      | 0.77   | 0.77     | 1601    |
| Weighted Avg  | 0.79      | 0.79   | 0.79     | 1601    |

<br><br>

### Interprétation des résultats

Même si le but n'était pas d'obtenir un modèle de classification de plantes performant, on observe que ce modèle de classification multi‑classe basé sur les embeddings a obtenu une précision globale de 79% sur le jeu de test, ce qui est un bon résultat pour un modèle de pré‑filtrage. Cependant, il y a clairement des classes pour lesquelles les performances sont plus faibles que pour d'autres, comme par exemple la classe 27 qui a une précision de seulement 44%, ce qui peut être dû à plusieurs facteurs tels que la qualité des images, la complexité de la classe ou encore l'imbalance des classes dans le dataset. Cependant, dans l'ensemble, ce modèle de classification multi‑classe basé sur les embeddings a montré une bonne capacité à généraliser à de nouvelles données, ce qui est un bon indicateur de sa capacité à être utilisé pour le filtrage automatique des images incohérentes.

<br><br>

Pour plus de détails, vous pouvez visiter le notebook Jupyter correspondant à cette étape : [notebook XGBoost](/home/jouell/code/jouell3/plant_detect2/notebooks/02_clustering_xgboost.ipynb).
