---
layout: default
title: "Étape 4 — XGBoost multi-classe"
parent: "Collecte des données"
nav_order: 4
---

# 🧩 Étape 4 — XGBoost multi‑classe sur embeddings

Les embeddings sont ensuite utilisés comme features pour entraîner un modèle XGBoost multi‑classe.

Pourquoi XGBoost ?

- il gère très bien les données tabulaires,
- il est robuste aux classes déséquilibrées,
- il est rapide à entraîner,
- il permet de tester rapidement la qualité des embeddings,
- il fournit des probabilités de prédiction utiles pour le filtrage.

Ce modèle n’est pas destiné à remplacer le modèle final (EfficientNet fine‑tuned),
mais il constitue un excellent outil d’analyse et de pré‑filtrage.
