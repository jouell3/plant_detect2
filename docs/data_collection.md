---
layout: default
---


# Première étape: Collecte de données



### 🎯 *Objectif du module*

<br>

**Un model de computer vision performant nécessite un dataset de qualité.** 

<br>
La collecte de données est une étape cruciale, mais elle peut être longue et fastidieuse, surtout lorsque les classes sont nombreuses et hétérogènes. Les images ont été obtenues à partir de l'API d'iNaturalist, qui est un site collaboratif qui permet au utilisateurs de contribuer et de partager des observations de différentes espèces d'être vivant (plantes, arbres, insects, animaux etc). Ces images sont libres de droit et la plus part ont été labellisées par des experts. Cependant, pas toutes ces images sont de qualité pour un entrainement d'un modèle de "computer vision". Il a donc fallu que je fasse un triage des images pour ne sélectionner que les meilleures. 

Pour m'aider dans cette tâche, j'ai dévelopé un outil de visualisation sur Streamlit qui m'a permit de sélectionner des premières images pour chacunes de ces nouvelles classes (la première passage a permit de sélection plus de 8000 images dans ces 33 catégories). J'ai ensuite utilisé ces images pour développer un pipeline de collecte automatisé (que je détaillerai plus tard) qui utilise l'API d'iNaturalist pour récupérer des images de plantes supplémentaire pour permettre le meilleur entrainement de modèles derrière. Pour ce faire, j'ai utilisé une approche de filtrage automatique basée sur les embeddings d'images en applicant des méthodes de classifications non suppervisées, ce qui a permis de détecter les images hors distribution sans introduire de biais de confirmation.

L’objectif de ce module de collect des données est d'avoir la plus haute qualité des images avant l’entraînement du modèle final, en utilisant cette approche :

- Téléchargement de 500 images par classe via l’API d’iNaturalist,
- Filtrage manuelle de ces images pour ne garder que les meilleures (environ 8000 images pour les 33 classes),
- Entrainement d’un pipeline de filtrage automatique basé sur les embeddings d’images, pour enrichir le dataset sans introduire de biais (image de qualité ou pas). 
- Pipeline de filtrage automatique des nouvelles images collectées via l'API d’iNaturalist, basé sur les embeddings d’images et un classifieur XGBoost.

Cette approche permet d’automatiser une partie du nettoyage du dataset **sans introduire le biais du self‑training**, contrairement à l’utilisation du modèle final comme filtre.

---

## Pipeline de filtrage automatique basé sur les embeddings d’images
<br>
Étant données que ce nouveau dataset de fleurs et d'arbres fruitiers contient peu d'images de chaque classe, j'ai décidé d'utiliser une méthode de clustering basée sur les embeddings d'images pour faire du filtrage automatique d'autres images. L'idée était d'utiliser les informatins contenu dans ces images manuellement triées pour faire du filtrage automatique des nouvelles images collectées via l'API d’iNaturalist. L'idée ici n'était pas d'utiliser le modèle final comme filtre pour ne pas biaiser le modèle final mais plutôt de créer un système de filtrage indépendant aux classes originales. Pour ce faire, j'ai utilisé les embeddings d'images extraits d'un modèle pré-entraîné (EfficientNet-B3) pour faire du clustering KMeans par classe, et ensuite utiliser ces clusters pour définir des seuils de filtrage automatique des nouvelles images. J'ai aussi utilisé un classifieur XGBoost multi-classe sur les embeddings pour faire du pré-filtrage des nouvelles images avant de faire le filtrage basé sur les distances aux centroïdes des clusters. Même si j'ai utilisé un classifieur XGBoost, l'idée n'était pas d'utiliser ce classifieur comme filtre principal mais plutôt comme un outil de pré-filtrage pour éviter de faire le filtrage basé sur les distances aux centroïdes des clusters pour des images qui sont clairement hors distribution (ex : une image d'un chat au lieu d'une image d'une plante).

<br>

### 🧩 Étape 1 — Extraction des embeddings (EfficientNet‑B3)
(Je vais revenir sur ce choix de modèle pré-entrainé dans la prochaine section - entrainement du modèle)

Chaque image est passée dans EfficientNet‑B3 (pré‑entraîné ImageNet).  
On récupère la sortie du dernier bloc convolutionnel, qui est un vecteur de dimension fixe (1536 features dans le cas de EfficientNet-B3).

Ce vecteur représente l’image dans un espace latent où des images similaires sont proches les unes des autres.

**Résultat :**

- un tableau `embeddings` de taille `(8001, 1536)`  
- un tableau `labels` de taille `(8001,)`

Ces embeddings servent de base à toutes les étapes suivantes.

---

### 🧩 Étape 2 — Clustering par classe (KMeans)

Pour chaque classe (ex : *cosmos*, *fig*, *zinnia*…), j’applique un clustering KMeans sur les embeddings correspondants.
Initiallement, je n'avais mis qu'un cluster par classe, mais cela ne capturait pas toute la diversité d'une classe (ex : différence entre fleurs et feuilles, angles de vue, conditions lumineuses).
J'ai donc décidé d'utiliser **4 clusters par classe**, ce qui permet de mieux modéliser la distribution interne de chaque classe (nombre un peu arbitraire ici).

Pourquoi par classe ?

- les images d’une même classe forment naturellement un cluster compact,
- cela permet de repérer les images atypiques (angles étranges, objets parasites, mauvaises plantes),
- cela permet de définir un **seuil automatique** pour filtrer les futures images.

Pour chaque classe, je calcule :

- le centroïde des clusters,
- la distance de chaque image de ces centroïdes,
- un seuil basé sur le percentile 99 * 1.2 des distances.

Ce seuil sert ensuite à détecter les outliers.

---

## 🧩 Étape 3 — Visualisation PCA

Pour valider visuellement la cohérence des clusters, j’ai appliqué une réduction de dimension (PCA) sur l’ensemble des embeddings.

Cela permet :

- de vérifier que les classes sont bien séparées,
- d’identifier les classes qui se chevauchent,
- de repérer les images anormales.

<br>

![PCA des embedings](/figures/PCA.png)

#### Figure 1 : Visualisation des embeddings avec PCA, colorés par classe. On observe une bonne séparation entre les classes, avec quelques chevauchements.
---

## 🧩 Étape 4 — XGBoost multi‑classe sur embeddings

Les embeddings sont ensuite utilisés comme features pour entraîner un modèle XGBoost multi‑classe.

Pourquoi XGBoost ?

- il gère très bien les données tabulaires,
- il est robuste aux classes déséquilibrées,
- il est rapide à entraîner,
- il permet de tester rapidement la qualité des embeddings,
- il fournit des probabilités de prédiction utiles pour le filtrage.

Ce modèle n’est pas destiné à remplacer le modèle final (EfficientNet fine‑tuned),  
mais il constitue un excellent outil d’analyse et de pré‑filtrage.

---

## 🧩 Étape 5 — Filtrage automatique des nouvelles images

Lorsqu’une nouvelle image est collectée :

1. on calcule son embedding via EfficientNet‑B3,
2. on demande à XGBoost de prédire une classe + une confiance,
3. on calcule la distance de l’embedding au centroïde de cette classe,
4. on compare cette distance au seuil défini lors du clustering.

Une image est acceptée si :

- la confiance XGBoost est suffisante,  
- **et** la distance au centroïde est inférieure au seuil.

Ce mécanisme permet :

- d’accepter automatiquement les images “typiques”,
- de rejeter les images douteuses,
- d’éviter d’introduire du bruit dans le dataset.

---

## 🧩 Avantages de cette approche

### ✔ Pas de biais de confirmation  
Contrairement au self‑training, le modèle final n’est jamais utilisé pour filtrer les données.

# Results

Après avoir appliqué ce pipeline de filtrage automatique, et avoir combiné toutes les images du dataset "aromates" avec les nouvelles images de fleurs et d'arbres fruitiers, j'ai obtenu un dataset final de 58 000 images réparties sur 58 classes différentes (23 classes d'aromates, 20 classes de fleurs et 15 classes d'arbres fruitiers). Ce dataset est de haute qualité, avec des images variées et représentatives de chaque classe, ce qui a permis d'entraîner un modèle de classification performant par la suite.

Voici la distribution finale des classes dans le dataset après le filtrage automatique :

![Distribution des classes dans le dataset final](/figures/distributions.png)



---
<br><br><br>

<div align="center">

| ⬅ [Previous] | [⬆ Main page](index.md) | [Dataset Collection ➡](dataset_collection.md) |
|-------------------------------|---------------------|-------------------------|

</div>
