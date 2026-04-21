---
layout: default
title: "Collecte des données"
nav_order: 2
has_children: true
has_toc: false
---


## Première étape: Collecte de données

## *Objectif du module*

<br>

**Un model de computer vision performant nécessite un dataset de qualité.**

<br>
La collecte de données est une étape cruciale, mais elle peut être longue et fastidieuse, surtout lorsque les classes sont nombreuses et hétérogènes. Les images ont été obtenues à partir de l'API d'iNaturalist, qui est un site collaboratif qui permet au utilisateurs de contribuer et de partager des observations de différentes espèces d'être vivant (plantes, arbres, insects, animaux etc). Ces images sont libres de droit et la plus part ont été labellisées par des experts. Cependant, pas toutes ces images sont de qualité pour un entrainement d'un modèle de "computer vision". Il a donc fallu que je fasse un triage des images pour ne sélectionner que les meilleures.

Pour m'aider dans cette tâche, j'ai développé un outil de visualisation sur Streamlit qui m'a permis de sélectionner les premières images pour chacune de ces nouvelles classes (cette première passe a permis de sélectionner plus de 8 000 images dans ces 33 catégories). J'ai ensuite utilisé ces images pour développer un pipeline de collecte automatisé (détaillé plus loin) qui utilise l'API d'iNaturalist pour récupérer des images supplémentaires afin d'enrichir le dataset. Pour ce faire, j'ai utilisé une approche de filtrage automatique basée sur les embeddings d'images, en appliquant des méthodes de classification non supervisées, ce qui a permis de détecter les images hors distribution sans introduire de biais de confirmation.

L’objectif de ce module de collect des données est d'avoir la plus haute qualité des images avant l’entraînement du modèle final, en utilisant cette approche :

- Téléchargement de 500 images par classe via l’API d’iNaturalist,
- Filtrage manuel de ces images pour ne garder que les meilleures (environ 8 000 images pour les 33 classes),
- Entraînement d’un pipeline de filtrage automatique basé sur les embeddings d’images, pour enrichir le dataset sans introduire de biais.
- Application du pipeline de filtrage automatique aux nouvelles images collectées via l’API d’iNaturalist, basé sur les embeddings et un classifieur XGBoost.

Cette approche permet d’automatiser une partie du nettoyage du dataset **sans introduire le biais du self‑training**, contrairement à l’utilisation du modèle final comme filtre.

<br>


Pour plus de détails sur les différentes étapes, voici le menu de navigation (aussi présent dans la barre latérale gauche) pour cette section de collecte des données:

- [Étape 1 — Extraction des embeddings](collecte/etape-1-embeddings.html)
- [Étape 2 — Clustering par classe](collecte/etape-2-clustering.html)
- [Étape 3 — Visualisation PCA](collecte/etape-3-pca.html)
- [Étape 4 — XGBoost multi-classe](collecte/etape-4-xgboost.html)
- [Étape 5 — Filtrage automatique](collecte/etape-5-filtrage.html)
- [Résultats et avantages](collecte/resultats.html)

---