---
layout: default
title: "Déploiement de l'API"
parent: "Déploiement"
nav_order: 1
---

## Déploiement de l'API de classification de plantes

### Plateforme

L'API est déployée sur **Google Cloud Run** (région `us-central1`, projet `bootcamparomatic`) via une image Docker poussée vers Google Artifact Registry. Cloud Run offre un scaling automatique, une facturation à l'usage et une URL publique HTTPS sans configuration de serveur.

> **URL publique :** [https://plantdetectapi-2a5a6b4c0e-uc.a.run.app](https://plantdetectapi-2a5a6b4c0e-uc.a.run.app)

---

### Processus de déploiement

1. Build de l'image Docker pour `linux/amd64` : `make build_gcp`
2. Push automatique vers Artifact Registry
3. Cloud Run récupère la nouvelle image au prochain démarrage
4. Au démarrage, les 5 modèles sont téléchargés depuis le registre Weights & Biases (cache TTL de 3 h)

---

### Référence API

#### `GET /`

Santé de l'API et liste des modèles actifs.

```bash
curl https://plantdetectapi-2a5a6b4c0e-uc.a.run.app/
```

```json
{
  "message": "Plant Detect API",
  "models": ["convnext_tiny", "efficientnet_b3", "efficientnet_b4", "mobilenetv3_large", "resnet50"]
}
```

---

#### `GET /models`

Détail de chaque modèle et liste complète des 58 classes reconnues.

```bash
curl https://plantdetectapi-2a5a6b4c0e-uc.a.run.app/models
```

```json
[
  [
    {"key": "convnext_tiny",      "timm_name": "convnext_tiny.fb_in22k_ft_in1k", "img_size": 224},
    {"key": "efficientnet_b3",    "timm_name": "efficientnet_b3.ra2_in1k",        "img_size": 300},
    {"key": "efficientnet_b4",    "timm_name": "efficientnet_b4.ra2_in1k",        "img_size": 380},
    {"key": "mobilenetv3_large",  "timm_name": "mobilenetv3_large_100",           "img_size": 224},
    {"key": "resnet50",           "timm_name": "resnet50.a1_in1k",                "img_size": 224}
  ],
  {
    "Aromatic herbs (23)": "Angelica, Basil, Borage, Chamomile, Chives, ...",
    "Flowers (19)":        "Daisy, Hellebore, Iris, Gerbera, Allium, ...",
    "Trees & berries (16)": "Blackberry, Blueberry, Cherry, Cranberry, ..."
  }
]
```

---

#### `POST /predict`

Prédiction top-K sur une image. Paramètres optionnels : `models` (défaut `all`) et `top_k` (défaut `3`, max `10`).

```bash
curl -X POST https://plantdetectapi-2a5a6b4c0e-uc.a.run.app/predict \
  -F "file=@lavande.jpg" \
  -F "models=convnext_tiny,efficientnet_b3" \
  -F "top_k=3"
```

```json
{
  "predictions": [
    {
      "model": "convnext_tiny",
      "top3": [
        {"class": "lavender",  "confidence": 0.944},
        {"class": "rosemary",  "confidence": 0.031},
        {"class": "hyssop",    "confidence": 0.012}
      ]
    },
    {
      "model": "efficientnet_b3",
      "top3": [
        {"class": "lavender",  "confidence": 0.901},
        {"class": "rosemary",  "confidence": 0.048},
        {"class": "mint",      "confidence": 0.021}
      ]
    }
  ]
}
```

---

#### `POST /predict-batch`

Prédiction top-1 par image pour un lot d'images. Paramètre optionnel : `models`.

```bash
curl -X POST https://plantdetectapi-2a5a6b4c0e-uc.a.run.app/predict-batch \
  -F "files=@plant1.jpg" \
  -F "files=@plant2.jpg" \
  -F "models=convnext_tiny"
```

```json
[
  {
    "filename": "plant1.jpg",
    "predictions": [
      {"model": "convnext_tiny", "class": "lavender",  "confidence": 0.944}
    ]
  },
  {
    "filename": "plant2.jpg",
    "predictions": [
      {"model": "convnext_tiny", "class": "sunflower", "confidence": 0.978}
    ]
  }
]
```

---

#### `POST /explore`

Top-K avec rang explicite par modèle — utilisé par la vue de comparaison côte-à-côte du frontend. Paramètres : `models`, `top_k` (défaut `5`).

```bash
curl -X POST https://plantdetectapi-2a5a6b4c0e-uc.a.run.app/explore \
  -F "file=@plant.jpg"
```

```json
{
  "filename": "plant.jpg",
  "predictions": [
    {
      "model": "convnext_tiny",
      "top_k": [
        {"rank": 1, "class": "lavender",  "confidence": 0.944},
        {"rank": 2, "class": "rosemary",  "confidence": 0.031},
        {"rank": 3, "class": "hyssop",    "confidence": 0.012},
        {"rank": 4, "class": "thyme",     "confidence": 0.008},
        {"rank": 5, "class": "mint",      "confidence": 0.003}
      ]
    }
  ]
}
```

---

#### `GET /metrics`

Snapshot de santé en temps réel : KPIs globaux, 20 dernières prédictions, distribution des classes prédites, stats moyennes par modèle. Rafraîchi toutes les 10 secondes par le frontend.

```bash
curl https://plantdetectapi-2a5a6b4c0e-uc.a.run.app/metrics
```

```json
{
  "kpis": {
    "total_requests":       42,
    "avg_latency_ms":      312.5,
    "avg_confidence":        0.871,
    "low_confidence_count":  2,
    "uptime_seconds":     3600
  },
  "recent_requests": [
    {
      "timestamp": "14:32:01",
      "convnext_tiny": {"class": "lavender", "confidence": 0.944, "latency_ms": 287.4}
    }
  ],
  "class_distribution": {"lavender": 8, "rosemary": 5, "sunflower": 3},
  "model_stats": {
    "convnext_tiny":     {"avg_latency_ms": 280.3, "avg_confidence": 0.895},
    "efficientnet_b3":   {"avg_latency_ms": 340.1, "avg_confidence": 0.851}
  }
}
```

---

#### `GET /metrics/predictions`

Historique complet des prédictions depuis le démarrage de l'instance (liste plate, exportable en CSV).

```bash
curl https://plantdetectapi-2a5a6b4c0e-uc.a.run.app/metrics/predictions
```

```json
{
  "predictions": [
    {
      "timestamp":   "14:32:01",
      "model":       "convnext_tiny",
      "class":       "lavender",
      "confidence":   0.944,
      "latency_ms":  287.4
    }
  ]
}
```
