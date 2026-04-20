---
layout: default
title: "Étape 4 — XGBoost multi-classe"
parent: "Collecte des données"
nav_order: 4
---

# 🧩 Étape 4 — XGBoost multi‑classe sur embeddings

Pour le filtrage autamatique, j'ai voulu mettre en place un arbre de désision basé sur les embeddings. La raison ici n'était pas de faire une classification des images à filtrer mais pour seulement déterminer si ces images étaient cohérentes avec les patterns appris par XGBoost des 8000 premières images filtrées manuellement, même si elles n'étaient pas classifiées dans la bonne classe. C'est pour cela que j'ai utilisé un classifieur multi-classe (35 classes) et pas un classifieur par classe (2 classes : dans la classe ou hors classe). Ensuite, si cette prédition avait un confience plus élevée que la limite qui avait été définie, alors les distances avec les 4 clusters calculés précédament étaient utilisées pour filtrer les images incohérentes.

Pourquoi XGBoost et cette approche ?

- il gère très bien les données tabulaires (comme les embeddings),
- il est robuste aux classes déséquilibrées,
- il est rapide à entraîner,
- il permet de tester rapidement la qualité des embeddings,
- il fournit des probabilités de prédiction utiles pour le filtrage.

Ce modèle n’est pas destiné à remplacer le modèle final (EfficientNet fine‑tuned), mais il constitue un excellent outil d’analyse et de pré‑filtrage pour pouvoir enrichir le dataser avec des images de qualité.


Pour plus de détails, vous pouvez visiter le notebook Jupyter correspondant à cette étape : [notebook XGBoost](/home/jouell/code/jouell3/plant_detect2/notebooks/02_clustering_xgboost.ipynb).
