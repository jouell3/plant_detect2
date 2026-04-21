---
layout: default
title: "index"
nav_order: 1
---



# Détection automatique de plantes par computer vision

> **API backend :** [https://plantdetectapi-2a5a6b4c0e-uc.a.run.app](https://plantdetectapi-2a5a6b4c0e-uc.a.run.app) — **Application :** [https://plantpredict.streamlit.app](https://plantpredict.streamlit.app) — **Wandb :** [wandb.ai/certification](https://wandb.ai/home)

## Introduction et Expression du besoin

### **Contexte du projet**

Pour cette certification, j'ai choisi de construire un classifieur d'images de plantes, avec un focus particulier sur les **fleurs, les aromates et les arbres fruitiers**. Ce travail a été initié lors du projet final de la formation Data Science & AI donnée par Artiefact School of Data. Avec 3 autres collègues de promotion, nous avons collecté un dataset de plus de 27 000 images réparties sur 23 classes différentes d'aromates, que nous avons utilisées pour entraîner 4 modèles différents de reconnaissance d'aromates.


### **Expression du besoin**

Bien que cette première application fonctionnait plutôt bien, elle était limitée : seulement 23 espèces de plantes étaient reconnues par ce premier modèle et la plupart des herbes aromatiques reconnues étaient bien connues du grand public. J'ai donc décidé d'étendre le périmètre du projet pour la certification, en enrichissant ce dataset avec des classes de fleurs et d'arbres fruitiers, afin de permettre au modèle d'identifier un plus grand nombre d'espèces. Cet outil de classification d'images de plantes pourrait être très utile pour les jardiniers amateurs, les botanistes, les agriculteurs, ou encore les curieux souhaitant identifier une plante trouvée dans la nature.

J'ai également identifié un problème méthodologique lors de la collecte des images pour le projet de groupe. L'approche utilisée n'était probablement pas idéale : un premier modèle avait été entraîné sur les premières images triées manuellement (environ 8 000 images), puis utilisé pour filtrer les images suivantes du dataset (en ne conservant que les images dont la classe prédite correspondait à la classe cible). Cette approche a probablement introduit un biais de confirmation : les images sélectionnées pour l'entraînement influençaient directement la sélection des données suivantes, ce qui peut conduire à une sur-représentation de certains types d'images et à une sous-représentation d'autres.

Les objectifs de ce projet de certification sont donc les suivants :

- Enrichir le dataset de plantes avec des classes de fleurs et d'arbres fruitiers, pour permettre au modèle d'identifier un plus grand nombre d'espèces.
- Mettre en place un processus de filtrage automatique des données pour éviter les biais liés au self-training, et garantir la qualité et la pertinence des images collectées pour l'entraînement du modèle final.
- Entraîner plusieurs modèles de classification de plantes sur ce dataset enrichi et sélectionner le meilleur.
- Déployer ce modèle sur une API pour le rendre accessible à tous.
- Mettre en place un système de monitoring pour suivre les performances du modèle en production et détecter les éventuelles dérives.


Ce rapport est divisé en plusieurs sections présentant de manière claire et structurée les différentes étapes du projet, depuis la collecte des données jusqu'au déploiement du modèle. Chaque section détaille les méthodes utilisées, les résultats obtenus et les leçons apprises tout au long du processus. La navigation entre sections est possible grâce à la table des matières présente dans la barre latérale gauche.
