# Plant Detect

API de reconnaissance de plantes (herbes aromatiques, fleurs, arbres fruitiers) basée sur 5 modèles de deep learning entraînés via la bibliothèque [`timm`](https://github.com/huggingface/pytorch-image-models) et servie par FastAPI sur Google Cloud Run. Les poids des modèles sont versionnés dans le registre [Weights & Biases](https://wandb.ai) et téléchargés à la demande au démarrage du conteneur.

> **Évolution depuis le projet de groupe initial.**
> La version d'origine reconnaissait 23 herbes aromatiques à partir d'un unique modèle ResNet18 stocké sur GCS. Cette version, réalisée dans le cadre d'une certification en data science / IA, étend le projet à **58 classes** (23 aromates + 19 fleurs + 16 arbres fruitiers) sur un dataset de **57 000+ images** collectées via l'API iNaturalist, avec **5 architectures benchmarkées** pour justifier le modèle retenu et une **migration complète GCS → wandb** pour le registre de modèles.

---

## Architecture

```
Utilisateur
    │
    │ (image)
    ▼
Frontend Streamlit  ──►  Backend FastAPI (Cloud Run)
                               │
                               │ (download .pth + label encoder à froid,
                               │  cache TTL en local)
                               ▼
                      Weights & Biases Registry
                        (5 artefacts versionnés)
```

### Modèles déployés

| Clé API | Backbone | Résolution | Rôle |
|---|---|---|---|
| `convnext_tiny` | ConvNeXt-Tiny | 224 | Meilleur modèle du benchmark (95.4% val) |
| `efficientnet_b3` | EfficientNet-B3 | 300 | Baseline moderne |
| `efficientnet_b4` | EfficientNet-B4 | 380 | Variante plus profonde |
| `mobilenetv3_large` | MobileNetV3-Large | 224 | Option low-latency |
| `resnet50` | ResNet-50 | 224 | Baseline classique |

Tous les modèles partagent le même encodeur de classes (58 classes, ordre alphabétique), généré par `torchvision.datasets.ImageFolder` lors de l'entraînement et sauvegardé comme `label_encoder.pkl` dans l'artefact wandb (ou `classes.txt` en fallback).

---

## Prérequis

- Python 3.11+
- Docker Desktop
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`) pour le déploiement
- Un projet GCP : `bootcamparomatic`
- Un compte [Weights & Biases](https://wandb.ai) et une clé API

### Authentification GCP

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project bootcamparomatic
```

### Authentification wandb

```bash
export WANDB_API_KEY=<votre clé depuis wandb.ai/settings>
export WANDB_PROJECT=certification
export WANDB_ENTITY=<votre utilisateur wandb>
```

Ces variables doivent être disponibles au démarrage du backend pour permettre le téléchargement des artefacts.

---

## Structure

```
backend/
  app/
    api/
      main.py                   # FastAPI — 5 endpoints
    src/
      herbs_detection/
        model_registry.py       # Source de vérité : les 5 ModelConfig
        timm_predictor.py       # Classe générique pour n'importe quel modèle timm
        wandb_loader.py         # Download + cache TTL des artefacts wandb
        __init__.py
  tests/
    conftest.py                 # Stubs TimmPredictor (pas de poids ni de GPU en test)
    test_api.py
    test_wandb_loader.py
    test_timm_predictor.py
  requirements.txt              # timm, wandb, torch, fastapi, ...
  requirements-dev.txt          # pytest, httpx

frontend/
  main.py                       # Page d'accueil Streamlit (FR/EN)
  pages/
    0_Prediction_aromate.py     # Prédiction simple (une image)
    1_Multiple_Predictions_Aromates.py  # Prédiction par lot
    2_Image_Labelling.py        # Sélection manuelle pour datasets
    3_Monitoring.py             # Tableau de bord de monitoring (latence, confiance)

notebooks/
  benchmark_models_colab.ipynb           # Benchmark des 5 architectures + KFold + diagnostics
  convnext_tiny_full_pipeline_colab.ipynb  # Pipeline complet du modèle retenu (70/15/15)
  confidence_exploration_colab.ipynb     # Analyse statistique des scores de confiance
  efficient_B3_pytorch_colab_2.ipynb     # Expérimentations EfficientNet-B3
  02_clustering_xgboost.ipynb            # Pipeline de filtrage automatique (embeddings + XGBoost)
  evaluation_models.ipynb                # Évaluation comparative des modèles (test set final)

docker/
  backend.Dockerfile
```

---

## 1. Entraînement des modèles

Deux notebooks couvrent le cycle d'entraînement, conçus pour tourner sur Google Colab (GPU T4/L4) :

| Notebook | Usage |
|---|---|
| `notebooks/benchmark_models_colab.ipynb` | Entraîne et compare les 5 architectures, valide la stabilité par StratifiedKFold, produit les heatmaps F1 / précision et le test de McNemar pour la comparaison statistique |
| `notebooks/convnext_tiny_full_pipeline_colab.ipynb` | Pipeline complet du modèle gagnant sur split 70/15/15 (test set strictement réservé à l'évaluation finale) |
| `notebooks/evaluation_models.ipynb` | Évaluation finale des modèles sur le test set — métriques, matrices de confusion, analyse par classe |

Chaque exécution logue ses hyperparamètres, métriques et courbes dans wandb. Les meilleurs checkpoints sont uploadés comme artefacts nommés `<model_key>_best` (par exemple `convnext_tiny_best`, `efficientnet_b3_best`, etc.) — noms utilisés tels quels par `model_registry.py`.

> Ajouter `classes.txt` ou `label_encoder.pkl` à l'artefact est obligatoire : sans ce fichier le backend ne peut pas décoder les prédictions.

---

## 2. Développement local

### Lancer l'API sans Docker

```bash
PYTHONPATH=backend/app/src uvicorn backend.app.api.main:api \
  --reload --host 0.0.0.0 --port 8080
```

Au démarrage, l'API charge les 5 modèles en parallèle dans des threads daemon. Les requêtes `/predict` bloquent sur un `threading.Event` jusqu'à ce que les poids soient prêts.

### Lancer avec Docker

```bash
make run
```

Monte `~/.config/gcloud/application_default_credentials.json` dans le conteneur et transmet `WANDB_API_KEY` via l'environnement.

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Santé + liste des modèles activés |
| GET | `/models` | Détail de chaque modèle (key, timm_name, img_size) |
| GET | `/metrics` | Snapshot de santé en temps réel : KPIs, 20 dernières prédictions, distribution par classe, stats par modèle |
| GET | `/metrics/predictions` | Historique complet des prédictions depuis le démarrage de l'instance (liste plate pour export CSV) |
| POST | `/predict` | Prédiction sur une image, top-k par modèle (paramètre `models=` pour filtrer) |
| POST | `/predict-batch` | Prédiction par lot, top-1 par image par modèle |
| POST | `/explore` | Top-K par modèle avec rang — destiné à la comparaison côte-à-côte dans le frontend |

### Exemples

```bash
# Liste des modèles disponibles
curl http://localhost:8080/models

# Prédiction sur une image, tous les modèles, top-3
curl -X POST http://localhost:8080/predict \
  -F "file=@data/raw/all_images/dill_0.jpg"

# Deux modèles seulement, top-1
curl -X POST http://localhost:8080/predict \
  -F "file=@data/raw/all_images/dill_0.jpg" \
  -F "models=convnext_tiny,efficientnet_b3" \
  -F "top_k=1"

# Exploration top-5 sur tous les modèles
curl -X POST http://localhost:8080/explore \
  -F "file=@data/raw/all_images/dill_0.jpg"

# Batch — un seul modèle, deux images
curl -X POST http://localhost:8080/predict-batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "models=convnext_tiny"
```

### Frontend Streamlit

```bash
cd frontend
streamlit run main.py
```

---

## 3. Tests

```bash
cd backend
pip install -r requirements-dev.txt
PYTHONPATH=app/src pytest -v
```

| Fichier | Périmètre |
|---|---|
| `tests/test_api.py` | Les 5 endpoints FastAPI, validation des paramètres `models=` et `top_k` |
| `tests/test_timm_predictor.py` | Le prédicteur générique (cas top-3, batch, moins de 3 classes) |
| `tests/test_wandb_loader.py` | Le cache TTL (miss / hit / expiration) |

Les tests stubbent entièrement `TimmPredictor` via `conftest.py` — aucun poids n'est chargé et aucun appel réseau n'est fait.

---

## 4. Déploiement sur Google Cloud Run

### 4.1 Créer le repository Artifact Registry (une seule fois)

```bash
gcloud artifacts repositories create plant-detect \
  --repository-format=docker \
  --location=europe-west1 \
  --project=bootcamparomatic

gcloud auth configure-docker europe-west1-docker.pkg.dev
```

### 4.2 Build et push de l'image

```bash
make build_gcp
```

Build l'image pour `linux/amd64` et la push vers Artifact Registry.

### 4.3 Premier déploiement

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

> `--memory=4Gi` est nécessaire pour charger les 5 modèles en mémoire simultanément. Réduire la liste dans `model_registry.py` (via `enabled=False`) si la contrainte mémoire est trop forte.
>
> `WANDB_API_KEY` doit être stocké dans Secret Manager (`gcloud secrets create wandb-api-key --data-file=-` puis coller la clé, Ctrl+D).

### 4.4 Mises à jour

Après un nouveau benchmark et l'upload d'un nouvel artefact dans wandb, rebuilder l'image suffit — le backend pullera automatiquement la dernière version au prochain démarrage (cache TTL de 3 h par défaut).

```bash
make build_gcp
```

---

## 5. Logs

```bash
make get_log_gcp

# ou en direct :
gcloud run services logs tail plant-detect-backend \
  --region=europe-west1 \
  --project=bootcamparomatic
```

---

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `WANDB_API_KEY` | — | Clé API Weights & Biases (obligatoire en production) |
| `WANDB_PROJECT` | `certification` | Projet wandb contenant les artefacts |
| `WANDB_ENTITY` | — | Utilisateur ou équipe propriétaire du registre |
| `WANDB_CACHE_MAX_AGE_SECONDS` | `10800` (3 h) | TTL du cache local des artefacts |
| `MODEL_PATH` | `models/wandb` | Dossier de cache local |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | ADC GCP (dev local uniquement) |

---

## Auteurs

- Jimmy OUELLET (solo — version de certification)
- Version initiale (groupe) : Jimmy OUELLET, Jaimes DE SOUSA GOMES, Thomas HEBERT, Edouard STEINER
