---
layout: default
title: "index"
nav_order: 1
---



# Détection automatique de plantes par computer vision

## Introduction

Pour cette certification, j’ai choisi de construire un classifieur d’images de plantes, avec un focus particulier sur les **fleurs, les aromates et les arbres fruitiers**. Ce travail a été démarré lors du projet final de la formation Data Science et AI donné par Artiefact School of data. Avec 3 autres de mes compagnons de promotion, nous avons collecté un dataset de plus de 20 000 images réparties sur 23 classes différentes d'aromate. Pour la certification, j’ai décidé d’enrichir ce dataset en ajoutant des classes de fleurs et d’arbres fruitiers (20 de fleurs et 13 d'arbres fruitiers), afin de démontrer ma capacité à collecter, préparer et analyser des données dans un contexte plus large. Un modèles a été entrainé sur ce dataset enrichi, et une API de classification d'images de plantes a été déployée. Enfin, j’ai mis en place un système de monitorage pour suivre les performances du modèle en production et détecter les éventuelles dérives.


Après avoir collecté totuyes ces images, j'ai ensuite entrainé un modèle pré-entrainé EfficientNet-B3 que j'ai adapté aux classes de plantes sélectionnées de ce dataset enrichi. J'ai monitoré au cours de cet entrainement les différentes métriques de mes modèles pour pouvoir prendre la décision de quel modèle sera utilisé pour la prochaine étape. Cette prochaine étape consiste justement a déployé ce modèle sur une API pour permettre à tous et chacun de l'utiliser pour faire la classification d'images de plantes. Finalement, pour assurer la qualité à long terme de mon modèle, j'ai mis en place un système de monitorage pour suivre les performances du modèle en production et détecter les éventuelles dérives.

Pour permttre le déployement de ce modèle et permettre à tous de l'utiliser, j'ai mis en ligne avec une interface utilisateur en utilisant Streamlit, pour permettre une utilisation facile et intuitive. Tout a été réalisé en respectant les bonnes pratiques de la data science, avec une attention particulière portée à la qualité des données et à l'évaluation rigoureuse des modèles.


<br><br><br>
### Pour permettre une navigation plus fluide dans ce rapport, voici une table des matières avec des liens vers les différentes sections :

- [Collecte de données](data_collection.md)
- [Entrainement du modèle](model_training.md)
- [Déploiement du modèle](deploiement.md)
- [Génération d'une interface utilisateur](frontend.md)
- [Monitoring des modèles](MLops.md)





---
<br><br><br>

<div align="center">

| ⬅ [Previous] | [⬆ Main page](index.md) | [Collecte des données ➡](data_collection.md) |
|-------------------------------|---------------------|-------------------------|

</div>






# Première étape: Collecte de données
🎯 Objectif du module

**Un model de computer vision performant nécessite un dataset de qualité.** 

La collecte de données est une étape cruciale, mais elle peut être longue et fastidieuse, surtout lorsque les classes sont nombreuses et hétérogènes. Les images ont été obtenues à partir de l'API d'iNaturalist, qui est un site collaboratif qui permet au utilisateurs de contribuer et de partager des observations de différentes espèces d'être vivant (plantes, arbres, insects, animaux etc). Ces images sont libres de droit et la plus part ont été labellisées par des experts. Cependant, pas toutes ces images sont de qualité pour un entrainement d'un modèle de "computer vision". Il a donc fallu que je fasse un triage des images pour ne sélectionner que les meilleures. 

Pour m'aider dans cette tâche, j'ai dévelopé un outil de visualisation sur Streamlit qui m'a permit de sélectionner des premières images pour chacunes de ces nouvelles classes (la première passage a permit de sélection plus de 8000 images dans ces 33 catégories). J'ai ensuite utilisé ces images pour développer un pipeline de collecte automatisé (que je détaillerai plus tard) qui utilise l'API d'iNaturalist pour récupérer des images de plantes supplémentaire pour permettre le meilleur entrainement de modèles derrière. Pour ce faire, j'ai utilisé une approche de filtrage automatique basée sur les embeddings d'images en applicant des méthodes de classifications non suppervisées, ce qui a permis de détecter les images hors distribution sans introduire de biais de confirmation.

L’objectif de ce module de collect des données est d'avoir la plus haute qualité des images avant l’entraînement du modèle final, en utilisant cette approche :

- Téléchargement de 500 images par classe via l’API d’iNaturalist,
- Filtrage manuelle de ces images pour ne garder que les meilleures (environ 8000 images pour les 33 classes),
- Entrainement d’un pipeline de filtrage automatique basé sur les embeddings d’images, pour enrichir le dataset sans introduire de biais (image de qualité ou pas). 
- Pipeline de filtrage automatique des nouvelles images collectées via l'API d’iNaturalist, basé sur les embeddings d’images et un classifieur XGBoost.

Cette approche permet d’automatiser une partie du nettoyage du dataset **sans introduire le biais du self‑training**, contrairement à l’utilisation du modèle final comme filtre.

---

### 🧩 Étape 1 — Extraction des embeddings (EfficientNet‑B3)

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

Pour valider visuellement la cohérence des clusters, j’applique une réduction de dimension (PCA) sur l’ensemble des embeddings.

Cela permet :

- de vérifier que les classes sont bien séparées,
- d’identifier les classes qui se chevauchent,
- de repérer les images anormales.

Cette visualisation est très utile pour la soutenance, car elle illustre concrètement la structure du dataset.

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

### ✔ Approche data‑centric moderne  
Elle met l’accent sur la qualité des données, ce qui est un point clé du RNCP.

### ✔ Compatible avec les compétences du bloc 2  
- **C1** : collecte automatisée  
- **C2** : préparation, nettoyage, outliers  
- **C6** : veille (embeddings, clustering, XGBoost)  
- **C7** : benchmark (LogReg vs XGBoost)  
- **C11** : monitorage (distance au centroïde)  
- **C12** : tests (qualité des données)

### ✔ Très bonne démonstration pour le jury  
Tu montres que tu maîtrises :

- les modèles pré‑entraînés,
- les embeddings,
- le clustering,
- les modèles tabulaires,
- la détection d’outliers,
- l’automatisation de la collecte.

---

## 🧩 Conclusion

Ce module constitue une brique essentielle du pipeline complet.  
Il permet d’améliorer la qualité du dataset avant l’entraînement du modèle final,  
tout en évitant les pièges du self‑training.

Il s’intègre naturellement dans la narration globale du projet :

> **Collecte → Préparation → Analyse → Filtrage → Entraînement → API → Déploiement → Monitorage**

Et il te donne un avantage clair lors de la soutenance, car tu montres une compréhension fine des enjeux liés aux données.



Voici une **section complète, structurée et professionnelle** pour ton rapport, dédiée au **Filtrage automatique des données**.  
Elle est écrite dans un style humain, technique, et parfaitement aligné avec les attentes du jury RNCP.  
Tu pourras l’ajuster, la raccourcir ou la reformuler selon ton style.

---

# 🧩 **Filtrage automatique des données (Data-Centric AI)**

## **Objectif**

L’objectif de cette étape est d’améliorer la qualité du dataset avant l’entraînement du modèle final, en automatisant la détection et l’exclusion des images non pertinentes.  
Cette démarche s’inscrit dans une approche **data‑centric**, où l’accent est mis sur la qualité des données plutôt que sur la complexité du modèle.

Le filtrage automatique permet :

- d’éliminer les images hors distribution (angles atypiques, objets parasites, mauvaises plantes, images floues),
- d’homogénéiser les classes,
- de réduire le bruit dans les données,
- d’accélérer la collecte de nouvelles images,
- d’éviter les biais liés au self‑training.

---

## **Problème rencontré**

Les nouvelles classes ajoutées au projet (fleurs, fruits, plantes diverses) présentaient :

- un volume d’images plus faible,
- une forte hétérogénéité visuelle,
- une qualité variable selon les sources,
- un risque élevé de bruit dans les données.

Un filtrage manuel aurait été trop long et peu reproductible.  
Il était donc nécessaire de mettre en place un **pipeline automatique** capable d’évaluer la pertinence de chaque image.

---

## **Approche retenue**

Pour éviter les biais du self‑training (où le modèle final filtre ses propres données), j’ai adopté une approche en trois étapes :

1. **Extraction d’embeddings** à l’aide d’un modèle pré‑entraîné (EfficientNet‑B3).
2. **Clustering non supervisé** pour modéliser la distribution interne de chaque classe.
3. **Classification XGBoost** pour exploiter les embeddings dans un modèle tabulaire.

Cette combinaison permet de filtrer les images selon deux critères complémentaires :

- **distance au cluster le plus proche** (cohérence visuelle),
- **confiance du classifieur XGBoost** (cohérence sémantique).

Une image est acceptée si elle est à la fois **visuellement proche** des images de la classe et **cohérente** avec les patterns appris par XGBoost.

---

## **1. Extraction des embeddings**

Chaque image est transformée en un vecteur numérique (embedding) à l’aide d’EfficientNet‑B3 pré‑entraîné sur ImageNet.

L’embedding capture :

- la texture,
- la forme,
- les couleurs dominantes,
- la structure visuelle globale.

Les embeddings sont ensuite **normalisés** (StandardScaler) afin de rendre les distances comparables entre classes.

---

## **2. Clustering par classe**

Pour chaque classe, j’ai appliqué un clustering KMeans avec **k = 3**.  
Ce choix permet de capturer la diversité interne d’une classe (ex : différentes variétés, angles de vue, conditions lumineuses).

Pour chaque cluster, j’ai calculé :

- son centroïde,
- la distance de chaque image au centroïde,
- un seuil automatique basé sur le **percentile 99** des distances.

Ce seuil représente la limite entre :

- les images “typiques” de la classe,
- les images “atypiques” ou hors distribution.

---

## **3. Classification XGBoost sur embeddings**

Les embeddings normalisés ont également été utilisés pour entraîner un modèle **XGBoost multi‑classe**.

Ce modèle fournit :

- une prédiction de classe,
- une probabilité associée.

Même si les probabilités sont naturellement faibles (35 classes), elles restent informatives pour détecter les images incohérentes.

---

## **4. Règle de décision**

Pour chaque nouvelle image :

1. calcul de l’embedding,
2. prédiction XGBoost,
3. distance au cluster le plus proche,
4. comparaison au seuil de la classe.

Une image est acceptée si :

- la confiance XGBoost est supérieure à un seuil minimal (≈ 0.10),
- la distance au centroïde est inférieure au seuil défini pour la classe.

Cette règle combine **cohérence visuelle** et **cohérence sémantique**.

---

## **Résultats**

Après correction des points critiques (normalisation, multi‑clusters, seuils adaptés), le pipeline :

- accepte **la majorité des images d’entraînement** (≈ 90–98 %),
- rejette efficacement les images hors distribution,
- identifie les classes difficiles (faible cohérence interne),
- permet de filtrer automatiquement les nouvelles images collectées.

Ce filtrage automatique a permis d’enrichir les classes faibles de manière fiable, tout en maintenant un dataset propre et homogène.

---

## **Bénéfices pour le projet**

- **Gain de temps considérable** dans la collecte de données.
- **Réduction du bruit** dans le dataset.
- **Amélioration de la qualité des classes faibles**.
- **Pipeline reproductible**, facile à réexécuter en cas d’ajout de nouvelles classes.
- **Approche data‑centric moderne**, valorisable lors de la soutenance.

---

## **Conclusion**

Ce module de filtrage automatique constitue une étape essentielle du pipeline global.  
Il garantit que seules les images pertinentes sont utilisées pour l’entraînement du modèle final, ce qui améliore la robustesse, la précision et la généralisation du classifieur.

Il s’intègre naturellement dans la démarche complète :

> **Collecte → Préparation → Filtrage → Enrichissement → Entraînement → Déploiement**

et renforce la qualité globale du projet.



