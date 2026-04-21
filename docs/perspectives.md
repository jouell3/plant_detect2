---
layout: default
title: "Perspectives"
nav_order: 5
---

# Perspectives et améliorations futures

Cette section présente les axes d'amélioration identifiés au cours du projet, qui n'ont pas pu être implémentés dans le cadre de cette certification mais qui constitueraient des évolutions naturelles du système.

---

## 1. Convergence vers un modèle unique en production

L'architecture actuelle charge **5 modèles en parallèle** au démarrage, ce qui requiert 4 Gi de RAM sur Cloud Run et un temps de démarrage à froid d'environ 60 secondes. Pour une mise en production à plus grande échelle, la convergence vers **un seul modèle** (ConvNeXt-Tiny) permettrait de :

- Réduire l'empreinte mémoire à ~1 Gi
- Diviser le coût Cloud Run par ~4
- Ramener le démarrage à froid à ~15 secondes

La page de comparaison côte-à-côte du frontend pourrait être conservée à des fins de démonstration en utilisant une page Streamlit dédiée simulant plusieurs modèles via un seul backend.

---

## 2. Réentraînement continu

Le pipeline de collecte et de filtrage automatique développé dans ce projet (embeddings + KMeans + XGBoost) est réutilisable pour enrichir le dataset avec de nouvelles images. Des améliorations envisageables :

- **Active learning** : prioriser la collecte sur les classes les plus faibles (`chrysanthemum`, `hydrangea`)
- **Réentraînement automatique** déclenché lorsque la confiance moyenne en production descend sous un seuil défini dans wandb
- **Extension à de nouvelles espèces** sans repartir de zéro grâce au transfer learning

---

## 3. Détection de dérive (Data Drift)

Le monitoring actuel suit la confiance et la latence en production, mais ne détecte pas encore de **dérive de distribution** — c'est-à-dire un changement dans le type d'images soumises par les utilisateurs par rapport au dataset d'entraînement. Des outils comme [Evidently AI](https://www.evidentlyai.com/) ou les alertes wandb permettraient d'automatiser cette détection et de déclencher un réentraînement si nécessaire.

---

## 4. Déploiement mobile

**MobileNetV3-Large** a été inclus dans le benchmark précisément pour cette perspective. Avec seulement 5.4 M de paramètres et une résolution d'entrée de 224×224, il est candidat à un déploiement **on-device** via TensorFlow Lite ou ONNX, permettant une prédiction sans connexion internet — utile pour les botanistes ou agriculteurs en zone rurale.
