---
layout: default
title: "Étape 1 — Clustering par classe"
parent: "Collecte des données"
nav_order: 2
---

# Étape 1 — Clustering par classe (KMeans)

---

## **1. Extraction des embeddings**

Chaque image est transformée en un vecteur numérique (embedding) à l’aide d’EfficientNet‑B3 pré‑entraîné sur ImageNet.

L’embedding capture :

- la texture,
- la forme,
- les couleurs dominantes,
- la structure visuelle globale.

Les embeddings sont ensuite **normalisés** (StandardScaler) afin de rendre les distances comparables entre les différentes classes.

## Clustering par classe

Pour chaque classe (ex : *cosmos*, *fig*, *zinnia*…), j’ai appliqué un clustering KMeans sur les embeddings extraits des images sélectionnées manuellement, un clustering par classe.

Initialement, j'avais utilisé un seul cluster par classe, mais cela ne capturait pas toute la diversité interne d'une classe (ex : différence entre fleurs et feuilles, angles de vue, conditions lumineuses). J'ai donc opté pour **4 clusters par classe**, ce qui permet de mieux modéliser cette distribution interne (nombre arbitraire — aucune différence significative observée entre 3 et 4 clusters).

Pourquoi par classe ?

- les images d’une même classe forment naturellement un cluster compact,
- cela permet de repérer les images atypiques (angles étranges, objets parasites, mauvaises plantes),
- cela permet de définir un **seuil automatique** pour filtrer les futures images.

Pour chaque classe, je calcule :

- le centroïde des clusters,
- la distance de chaque nouvelle image de ces centroïdes,
- un seuil basé sur le percentile 99 * 1.2 de ces distances (vu le faible nombre d'images par classe, je n'ai pas voulu un seuil trop strict).

Pour toute nouvelle image, ce seuil sert à sélectionner les images pertinentes et à éliminer celles trop éloignées du cluster.

