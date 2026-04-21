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

## 2. Vote doux pondéré sur la distribution complète

L'agrégation actuelle repose sur un **vote majoritaire pondéré** appliqué uniquement au top-1 de chaque modèle. La méthode standard en apprentissage automatique est le **vote doux pondéré** (*weighted soft voting*) : on moyenne les distributions de probabilité complètes de chaque modèle, pondérées par leur précision de validation. Le résultat reste une vraie probabilité interprétable au sens bayésien.

Cette amélioration nécessiterait deux changements :

- **Backend** : modifier l'endpoint `/predict` pour retourner les 58 scores de confiance (un par classe) au lieu du top-3 uniquement.
- **Frontend** : remplacer le vote majoritaire par la somme pondérée des vecteurs de probabilité complets.

```python
# Avec les distributions complètes disponibles
def weighted_soft_vote(predictions):
    combined = defaultdict(float)
    for item in predictions:
        w = WEIGHTS[item["model"]]
        for class_name, prob in item["full_distribution"].items():
            combined[class_name] += w * prob
    total_weight = sum(WEIGHTS.values())
    return {c: s / total_weight for c, s in combined.items()}
```

Cela permettrait également d'envisager une **évaluation systématique de l'ensemble** sur un jeu de validation étiqueté : en soumettant un ensemble d'images dont la classe est connue, on pourrait comparer les méthodes d'agrégation (top-1 pondéré, soft voting, vote simple) via des métriques classiques (F1, matrice de confusion, test de McNemar) et valider statistiquement le gain apporté par l'ensemble par rapport à ConvNeXt-Tiny seul.

---

## 3. Réentraînement continu

Le pipeline de collecte et de filtrage automatique développé dans ce projet (embeddings + KMeans + XGBoost) est réutilisable pour enrichir le dataset avec de nouvelles images. Des améliorations envisageables :

- **Active learning** : prioriser la collecte sur les classes les plus faibles (`chrysanthemum`, `hydrangea`)
- **Réentraînement automatique** déclenché lorsque la confiance moyenne en production descend sous un seuil défini dans wandb
- **Extension à de nouvelles espèces** sans repartir de zéro grâce au transfer learning

---

## 4. Détection de dérive (Data Drift)

Le monitoring actuel suit la confiance et la latence en production, mais ne détecte pas encore de **dérive de distribution** — c'est-à-dire un changement dans le type d'images soumises par les utilisateurs par rapport au dataset d'entraînement. Des outils comme [Evidently AI](https://www.evidentlyai.com/) ou les alertes wandb permettraient d'automatiser cette détection et de déclencher un réentraînement si nécessaire.

---

## 5. Déploiement mobile

**MobileNetV3-Large** a été inclus dans le benchmark précisément pour cette perspective. Avec seulement 5.4 M de paramètres et une résolution d'entrée de 224×224, il est candidat à un déploiement **on-device** via TensorFlow Lite ou ONNX, permettant une prédiction sans connexion internet — utile pour les botanistes ou agriculteurs en zone rurale.
