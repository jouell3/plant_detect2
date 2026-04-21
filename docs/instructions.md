---
layout: default
title: "Instructions"
nav_order: 7
---

# Instructions — Mise en route du projet

Cette page regroupe les instructions nécessaires pour cloner le dépôt, installer les dépendances et lancer l'application en local ou sur Google Cloud Run.

---

## Prérequis

| Outil | Version minimale | Usage |
|---|---|---|
| Python | 3.11+ | Backend et frontend |
| Docker Desktop | Récent | Build et run de l'image backend |
| Google Cloud SDK (`gcloud`) | Récent | Déploiement sur Cloud Run |
| Compte [Weights & Biases](https://wandb.ai) | — | Téléchargement des artefacts de modèles |

---

## 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/plant_detect2.git
cd plant_detect2
```

---

## 2. Installation des dépendances

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Principales dépendances installées :

| Package | Rôle |
|---|---|
| `fastapi` + `uvicorn` | Serveur API |
| `torch` + `torchvision` | Deep learning |
| `timm` | Modèles pré-entraînés (EfficientNet, ConvNeXt, etc.) |
| `wandb` | Téléchargement des artefacts de modèles |
| `pillow` | Traitement des images |
| `scikit-learn` | Label encoder |
| `loguru` | Logging |

> **Note :** `torch` est une dépendance lourde (~2 Go). L'installation peut prendre quelques minutes.

Pour les tests uniquement :

```bash
pip install -r requirements-dev.txt
```

### Frontend

```bash
cd frontend
pip install streamlit
```

---

## 3. Variables d'environnement

Ces variables sont nécessaires au démarrage du backend pour télécharger les artefacts depuis Weights & Biases :

```bash
export WANDB_API_KEY=<votre clé depuis wandb.ai/settings>
export WANDB_PROJECT=certification
export WANDB_ENTITY=<votre utilisateur wandb>
```

Pour le développement local avec Docker, l'authentification GCP est également nécessaire :

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project bootcamparomatic
```

---

## 4. Lancer le backend en local

### Sans Docker

```bash
PYTHONPATH=backend/app/src uvicorn backend.app.api.main:api \
  --reload --host 0.0.0.0 --port 8080
```

Ou via le Makefile :

```bash
make serve
```

Au démarrage, les 5 modèles sont chargés en parallèle dans des threads daemon. Les requêtes sont mises en attente jusqu'à ce que les poids soient disponibles.

### Avec Docker

```bash
make run
```

Cette commande construit l'image localement puis démarre le conteneur en transmettant les variables `WANDB_API_KEY`, `WANDB_PROJECT` et `WANDB_ENTITY` depuis l'environnement shell courant (assurez-vous qu'elles sont exportées au préalable).

---

## 5. Lancer le frontend

```bash
cd frontend
streamlit run main.py
```

Avant the lancer le frontend, il faut s'assurer que l'URL utiliser pour se connect aux API est bien configurée sur `http://localhost:8080` .

---

## 6. Lancer les tests

```bash
cd backend
PYTHONPATH=app/src pytest -v
```

Les tests stubbent entièrement `TimmPredictor` — aucun poids n'est chargé et aucun appel réseau n'est effectué.

---

## 7. Référence Makefile

| Commande | Description |
|---|---|
| `make serve` | Lance l'API localement avec le bon `PYTHONPATH` |
| `make build` | Construit l'image Docker localement |
| `make run` | Construit et démarre le conteneur avec les variables WandB transmises depuis l'environnement |
| `make build_local` | Construit l'image pour `linux/amd64` sans push (test de build GCP) |
| `make build_gcp` | Construit l'image pour `linux/amd64` et la pousse vers Artifact Registry |
| `make get_log_gcp` | Affiche les 50 derniers logs Cloud Run |

---

## 8. Déploiement sur Google Cloud Run

### Créer le repository Artifact Registry (une seule fois)

```bash
gcloud artifacts repositories create plant-detect \
  --repository-format=docker \
  --location=europe-west1 \
  --project=bootcamparomatic

gcloud auth configure-docker europe-west1-docker.pkg.dev
```

### Build et push

```bash
make build_gcp
```

### Premier déploiement

```bash
gcloud run deploy plant-detect-backend \
  --image=europe-west1-docker.pkg.dev/bootcamparomatic/plant-detect/plant-detect-backend \
  --platform=managed \
  --region=europe-west1 \
  --memory=4Gi \
  --cpu=2 \
  --timeout=300 \
  --set-env-vars WANDB_PROJECT=certification,WANDB_ENTITY=<votre-entity>,WANDB_CACHE_MAX_AGE_SECONDS=10800 \
  --set-secrets WANDB_API_KEY=wandb-api-key:latest \
  --allow-unauthenticated
```

> `--memory=4Gi` est requis pour charger les 5 modèles simultanément. `WANDB_API_KEY` doit être stocké dans Secret Manager avant le déploiement.

### Mises à jour

Après l'upload d'un nouvel artefact dans wandb, il n'est pas nécessaire de reconstruire l'image — le backend télécharge automatiquement la dernière version au démarrage (cache TTL de 3 h). Il suffit de relancer le service pour qu'il recharge les poids depuis wandb.

---

## 9. Fichiers clés du dépôt

| Fichier | Description |
|---|---|
| `backend/app/api/main.py` | Point d'entrée FastAPI — définit tous les endpoints, gère le chargement parallèle des modèles et les logs de métriques |
| `backend/app/src/herbs_detection/model_registry.py` | Source de vérité : liste des 5 `ModelConfig` (clé, nom timm, taille d'image, activé/désactivé) |
| `backend/app/src/herbs_detection/timm_predictor.py` | Classe générique de prédiction — fonctionne avec n'importe quel modèle timm via une interface unifiée |
| `backend/app/src/herbs_detection/wandb_loader.py` | Téléchargement et cache local (TTL configurable) des artefacts depuis le registre Weights & Biases |
| `backend/app/src/herbs_detection/metrics_store.py` | Store en mémoire des prédictions (classe, confiance, latence) — alimente les endpoints `/metrics` |
| `backend/app/src/herbs_detection/monitoring.py` | Envoi des métriques de production vers wandb en temps réel (table de prédictions, runs) |
| `backend/tests/conftest.py` | Stubs `TimmPredictor` pour les tests — aucun poids chargé, aucun appel réseau |
| `backend/tests/test_api.py` | Tests des 7 endpoints FastAPI (statuts, validation des paramètres, formats de réponse) |
| `backend/tests/test_wandb_loader.py` | Tests du cache TTL (miss / hit / expiration) |
| `backend/tests/test_timm_predictor.py` | Tests unitaires du prédicteur (top-3, batch, cas limites) |
| `backend/requirements.txt` | Dépendances du backend (torch, timm, fastapi, wandb…) |
| `docker/backend.Dockerfile` | Définition de l'image Docker du backend |
| `Makefile` | Commandes de build, run local et déploiement GCP (`serve`, `run`, `build_gcp`, `get_log_gcp`…) |
| `pyproject.toml` | Configuration du package `herbs_detection` (installable via pip depuis `backend/app/src`) |
| `frontend/main.py` | Page d'accueil Streamlit — point d'entrée de l'application |
| `frontend/i18n.py` | Gestion du changement de langue FR / EN |
| `frontend/pages/0_Prediction_aromate.py` | Page de prédiction simple (une image, top-3 par modèle) |
| `frontend/pages/1_Multiple_Predictions_Aromates.py` | Page de prédiction en batch (lots de 20 images) |
| `frontend/pages/2_Image_Labelling.py` | Outil de labellisation manuelle avec export CSV |
| `frontend/pages/3_Monitoring.py` | Tableau de bord de monitoring (confiance, latence, distribution des classes) |
| `notebooks/benchmark_models_colab.ipynb` | Benchmark des 5 architectures — KFold, heatmaps F1/précision, test de McNemar |
| `notebooks/convnext_tiny_full_pipeline_colab.ipynb` | Pipeline complet du modèle retenu sur split 70/15/15 |
| `notebooks/evaluation_models.ipynb` | Évaluation finale sur le test set — matrices de confusion, métriques par classe |
| `notebooks/confidence_exploration_colab.ipynb` | Analyse statistique des scores de confiance par classe et par modèle |
| `notebooks/02_clustering_xgboost.ipynb` | Pipeline de filtrage automatique — embeddings, KMeans, XGBoost |
| `notebooks/scrapping_flowers.ipynb` | Script de collecte via l'API iNaturalist avec filtrage automatique intégré |

