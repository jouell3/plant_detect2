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

## **Interprétation**

Comme il est posisble de voir sur cette visualisation, les classes sont relativement bien séparées, ce qui confirme que les embeddings capturent des caractéristiques visuelles pertinentes. Cependant, on peut aussi observer quelques chevauchements entre certaines classes, ce qui suggère que certaines images pourraient être ambiguës ou que les classes partagent des caractéristiques visuelles similaires. Il pourrait être possible de vérifier que les images qui se trouvent dans ces zones de chevauchement sont effectivement des images de mauvaise qualité ou d'angles atypiques, ce qui confirmera la pertinence du filtrage automatique mais par manque de temps, cette étape n'a pas été réalisée. Il pourrait être aussi intéressant de faire une analyse plus approfondie de ces classes chevauchantes pour comprendre les raisons de ce chevauchement (ex : classes visuellement similaires, images mal étiquetées, etc.) et éventuellement ajuster le processus de collecte ou de filtrage en conséquence. Nous allons revenir sur ces classes qui sont plus difficile visuellement à différencier dans la partie 5 du projet, lors de l'analyse des erreurs du modèle final (matrice de confusion).
