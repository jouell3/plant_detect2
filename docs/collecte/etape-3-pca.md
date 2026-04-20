---
layout: default
title: "Étape 3 — Visualisation PCA"
parent: "Collecte des données"
nav_order: 3
---

# 🧩 Étape 3 — Visualisation PCA

Pour valider visuellement la cohérence des clusters, j’ai appliqué une réduction de dimension (PCA) sur l’ensemble des embeddings.

Cela permet :

- de vérifier qu'on est cappable de voir différent classes et qu'elles sont relativement bien séparées,
- d’identifier les classes qui se chevauchent,
- de déterminer si certaines images, même sélectionnées manuellement, semblent être à l'extérieur de la distribution.

<br>

![PCA des embedings](../figures/PCA.png)

#### Figure 1 : Visualisation des embeddings avec PCA, colorés par classe. On observe une bonne séparation entre les classes, avec quelques chevauchements.
