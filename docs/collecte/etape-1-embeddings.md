---
layout: default
title: "Étape 1 — Extraction des embeddings"
parent: "Collecte des données"
nav_order: 1
---

# 🧩 Étape 1 — Extraction des embeddings (EfficientNet‑B3)

(Je vais revenir sur ce choix de modèle pré-entrainé dans la prochaine section - entrainement du modèle)

Chaque image est passée dans EfficientNet‑B3 (pré‑entraîné ImageNet).
On récupère la sortie du dernier bloc convolutionnel, qui est un vecteur de dimension fixe (1536 features dans le cas de EfficientNet-B3).

Ce vecteur représente l’image dans un espace latent où des images similaires sont proches les unes des autres.

**Résultat :**

- un tableau `embeddings` de taille `(8001, 1536)`
- un tableau `labels` de taille `(8001,)`

Ces embeddings servent de base à toutes les étapes suivantes.
