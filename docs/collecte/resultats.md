---
layout: default
title: "Résultats et avantages"
parent: "Collecte des données"
nav_order: 6
---

# 🧩 Avantages de cette approche

### ✔ Pas de biais de confirmation
Contrairement au self‑training, le modèle final n’est jamais utilisé pour filtrer les données.

# Résultats

Après avoir appliqué ce pipeline de filtrage automatique, et avoir combiné toutes les images du dataset "aromates" avec les nouvelles images de fleurs et d'arbres fruitiers, j'ai obtenu un dataset final de 58 000 images réparties sur 58 classes différentes (23 classes d'aromates, 20 classes de fleurs et 15 classes d'arbres fruitiers). Ce dataset est de haute qualité, avec des images variées et représentatives de chaque classe, ce qui a permis d'entraîner un modèle de classification performant par la suite.

Voici la distribution finale des classes dans le dataset après le filtrage automatique :

![Distribution des classes dans le dataset final](../figures/distributions.png)
