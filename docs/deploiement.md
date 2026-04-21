---
layout: default
title: "Déploiement"
nav_order: 4
has_children: true
has_toc: false
---

# Deploiement du modèle de classification de plantes

## *Objectif du module*

Afin de rendre accessible le modèle de classification de plantes entraîné dans les étapes précédentes, j'ai déployé une API permettant à tout utilisateur de soumettre une image et d'obtenir une prédiction. Cette étape est cruciale pour concrétiser le travail réalisé et le mettre à disposition d'un large public.

### **Architecture de l'application**

```
┌──────────────────┐   image (HTTP POST)   ┌─────────────────────────────────┐
│    Frontend      │ ───────────────────►  │  Backend — Cloud Run            │
│   Streamlit      │ ◄───────────────────  │  FastAPI + 5 modèles timm       │
└──────────────────┘    JSON predictions   └──────────────┬──────────────────┘
                                                          │                   
                              téléchargement              │      logs métriques
                           artefacts (cache 3h)           │   (confiance, latence)
                                          ▼               ▼
                             ┌────────────────────────────────────┐
                             │     Weights & Biases               │
                             │  Registry (poids) + Dashboard      │
                             │  (monitoring production)           │
                             └────────────────────────────────────┘
```
<br>

> **API backend :** [https://plantdetectapi-2a5a6b4c0e-uc.a.run.app](https://plantdetectapi-2a5a6b4c0e-uc.a.run.app)
> **Application frontend :** [https://plantpredict.streamlit.app](https://plantpredict.streamlit.app)
> **Tableau de bord wandb :** [wandb.ai/certification](https://wandb.ai/home)

L'architecture de l'application est composée de deux parties principales : le **backend**, qui héberge l'API de classification, et le **frontend**, qui fournit une interface utilisateur pour interagir avec l'API.

Le backend est développé avec **FastAPI**, un micro-framework Python permettant de créer une API RESTful pour recevoir des images, les traiter et retourner les prédictions des modèles.

Le frontend est développé avec **Streamlit**, une bibliothèque Python permettant de créer des applications web interactives et bien intégrées avec les outils de data science. Il permet de téléverser des images, d'obtenir des prédictions et d'afficher les résultats de manière claire et intuitive.

Pour permettre un déploiement efficace, le projet est organisé de manière à isoler clairement le code du backend et du frontend dans des répertoires dédiés. Le répertoire `backend/` contient l'ensemble du code de l'API, y compris les scripts de chargement des modèles, de traitement des images et de retour des prédictions. Le répertoire `frontend/` contient les scripts Streamlit pour les différentes pages de l'application et la logique d'interaction avec l'API. Cette séparation facilite le développement, le déploiement et la maintenance indépendante des deux couches.

Voici la structure de mon projet pour le déploiement de l'application de classification de plantes :

```
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── main.py          # FastAPI — endpoints de prédiction
│   │   └── src/
│   │       └── herbs_detection/ # Package ML (timm_predictor, wandb_loader, model_registry)
│   └── tests/                   # Tests unitaires (pytest)
├── docker/
│   └── backend.Dockerfile
├── docs/                        # Rapport de certification (MkDocs)
├── frontend/
│   ├── main.py                  # Page d'accueil Streamlit
│   └── pages/                   # Pages de l'application
├── models/
│   └── wandb/                   # Cache local des artefacts wandb
├── notebooks/                   # Notebooks Colab (entraînement, évaluation, filtrage)
```

<br>

Les fichiers de configuration (`requirements.txt`, `Dockerfile`, `pyproject.toml`) sont placés à la racine ou dans les répertoires correspondants pour faciliter le développement et le déploiement de l'application.


