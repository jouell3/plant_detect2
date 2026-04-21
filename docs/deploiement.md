---
layout: default
title: "Déploiement"
nav_order: 4
has_children: true
has_toc: false
---

# Deploiement du modèle de classification de plantes

## *Objectif du module*

Afin de pouvoir utiliser le modèle de classification de plantes que j'ai entrainé dans les étapes précédentes, j'ai déployé sur une API pour permettre à tous et chacun de l'utiliser pour faire la classification d'images de plantes. Cette étape de déploiement est cruciale pour rendre le modèle accessible et utilisable par un large public, et pour permettre à d'autres personnes de bénéficier des résultats de mon travail.

### **Architecture de l'application**

L'architecture de mon application de classification de plantes est composée de deux parties principales : le backend, qui héberge l'API de classification de plantes, et le frontend, qui fournit une interface utilisateur pour interagir avec l'API. Le backend est développé en utilisant FastAPI, un micro-framework web en Python, qui permet de créer une API RESTful pour recevoir des images de plantes, les traiter et retourner les prédictions des modèles. Le frontend est développé en utilisant Streamlit, une bibliothèque Python qui permet de créer des applications web interactives de manière simple et rapide, et qui offre une intégration facile avec les bibliothèques de data science et de machine learning en Python. Le frontend permet aux utilisateurs de télécharger des images de plantes, de faire des prédictions avec les modèles de classification de plantes, et d'afficher les résultats de manière claire et intuitive. L'architecture de l'application est conçue pour être simple et efficace, en permettant une communication fluide entre le frontend et le backend, et en offrant une expérience utilisateur agréable et intuitive pour faire la classification d'images de plantes.

Pour permettre un deploiement efficace de mon application, j'ai organisé mon projet de manière à isoler le code du backend et du frontend dans des répertoires dédiés. Le répertoire "backend" contient tout le code nécessaire pour héberger l'API de classification de plantes, y compris les scripts pour charger les modèles, traiter les images et retourner les prédictions. Le répertoire "frontend" contient tout le code nécessaire pour créer l'interface utilisateur avec Streamlit, y compris les scripts pour afficher les différentes pages de l'application et pour interagir avec l'API du backend. Cette organisation permet de mieux structurer le projet et de faciliter le développement et le déploiement de l'application, en permettant une séparation claire entre les différentes parties de l'application et en facilitant la maintenance et les mises à jour futures.

Voici la structure de mon projet pour le déploiement de l'application de classification de plantes :

```
├── backend
│   ├── app
│   │   ├── api
│   │   ├── core
│   │   ├── models_flowers_fruits
│   │   ├── new_model
│   │   ├── __pycache__
│   │   ├── src
│   │   └── templates
│   ├── __pycache__
│   ├── tests
│   │   └── __pycache__
├── data
│   ├── dataset
├── docker
├── docs
│   ├── collecte
│   ├── deploy
│   ├── figures
│   ├── frontend
│   ├── MLops
│   ├── superpowers
│   │   ├── plans
│   │   └── specs
│   └── training
├── frontend
│   ├── pages
│   │   └── __pycache__
│   └── __pycache__
├── models
│   └── wandb
│       ├── convnext_tiny_best
│       ├── efficientnet_b3_best
│       ├── efficientnet_b4_best
│       ├── label_encoder
│       ├── mobilenetv3_large_best
│       └── resnet50_best
├── notebooks
```

<br>

Les autres fichiers comme requirements.txt (tous les modules requis poru el bon fonctionnement de mon environment), Dockerfile (pour permettre de construire l'image Docker), pyproject.py (pour le module herb_detection, qui contient le code source pour les différentes fonctions de mon backend), poeptry.lock etc. sont à la racine du projet pour faciliter le développement et le déploiement de l'application de classification de plantes.


