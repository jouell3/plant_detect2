---
layout: default
title: "Étape 2 — Clustering par classe"
parent: "Collecte des données"
nav_order: 2
---

# 🧩 Étape 2 — Clustering par classe (KMeans)

Pour chaque classe (ex : *cosmos*, *fig*, *zinnia*…), j’ai appliqué un clustering KMeans sur les embeddings correspondants de cette classe.

Initiallement, je n'avais mis qu'un seul cluster par classe, mais cela ne capturait pas toute la diversité d'une classe (ex : différence entre fleurs et feuilles, angles de vue, conditions lumineuses).
J'ai donc décidé d'utiliser **4 clusters par classe**, ce qui permet de mieux modéliser la distribution interne de chaque classe (nombre un arbitraire mais pas vu de différence entre 3 et 4 clusters).

Pourquoi par classe ?

- les images d’une même classe forment naturellement un cluster compact,
- cela permet de repérer les images atypiques (angles étranges, objets parasites, mauvaises plantes),
- cela permet de définir un **seuil automatique** pour filtrer les futures images.

Pour chaque classe, je calcule :

- le centroïde des clusters,
- la distance de chaque nouvelle image de ces centroïdes,
- un seuil basé sur le percentile 99 * 1.2 de ces distances (vu le faible nombre d'images par classe, je n'ai pas voulu un seuil trop strict).

Pour toute nouvelle image, ce seuil a servi a sélectionner les images pertinentes et eliminer les images trop différentes pour ce cluster.
