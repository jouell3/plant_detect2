---
layout: default
title: "index"
nav_order: 1
---



# Détection automatique de plantes par computer vision

## Introduction et Expression du besoin

### Context du projet

Pour cette certification, j’ai choisi de construire un classifieur d’images de plantes, avec un focus particulier sur les **fleurs, les aromates et les arbres fruitiers**. Ce travail a été démarré lors du projet final de la formation Data Science et AI donné par Artiefact School of data. Avec 3 autres de mes compagnons de promotion, nous avons collecté un dataset de plus de 20 000 images réparties sur 23 classes différentes d'aromate. 


### Expression du besoin

Bien que cette première application fonctionnait plutôt bien, elle était plutôt limitée : seulement 23 espères de plantes étaient reconnues pour ce premier modèles et la plupart des herbes aromatiques bien connues des gens. J’ai donc voulu aller plus loin pour la certification en enrichissant ce dataset avec des classes de fleurs et d'arbres fruitiers, afin de permetttre au modèle de déterminer l'espère de plus de plantes. Également, lors de la collecte des images pour ce projet de groupe, j'ai utilisé uen approche qui est un peu discutable ; un premier modèle avait été entrainené avec les premières images triées manuellement et utilisé ensuite pour faire la selection des autres images du dataset. Cette approach a probablement introduit un biais puisque les images sélectionnées pour l'entraînement du modèle sont ensuite utilisées pour faire la sélection des autres images du dataset, ce qui peut conduire à une sur-représentation de certains types d'images et à une sous-représentation d'autres types d'images. 

Les objectifs de ce projet de certification sont donc les suivants :

- Enrichir le dataset de plantes avec des classes de fleurs et d'arbres fruitiers, pour permettre au modèle de déterminer l'espère de plus de plantes.
- Mettre en place un processus de filtrage automatique des données pour éviter les biais liés au self-training, et pour garantir la qualité et la pertinence des images collectées pour l'entraînement du modèle final.
- Entraîner quelques modèles de classification de plantes sur ce dataset enrichi, et sélectionner le meilleur modèle
- Déployer sur une API ce modèle pour permettre à tous et chacun de l'utiliser pour faire la classification d'images de plantes.
- Mettre en place un système de monitorage pour suivre les performances du modèle en production et détecter les éventuelles dérives.  



afin de démontrer ma capacité à collecter, préparer et analyser des données dans un contexte plus large. J'ai collecté plus de 10 000 images supplémentaires pour ces nouvelles classes, en utilisant différentes sources telles que Google Images, Flickr et des bases de données spécialisées. J'ai ensuite appliqué un processus de filtrage automatique pour garantir la qualité et la pertinence des images collectées, en utilisant une combinaison d'extraction d'embeddings avec un modèle pré-entraîné (EfficientNet-B3), de clustering non supervisé pour modéliser la distribution interne de chaque classe, et de classification XGBoost pour exploiter les embeddings dans un modèle tabulaire. Ce processus m'a permis d'améliorer la qualité du dataset avant l'entraînement du modèle final, en automatisant la détection et l'exclusion des images non pertinentes.

Pour la certification, j’ai décidé d’enrichir ce dataset en ajoutant des classes de fleurs et d’arbres fruitiers (20 de fleurs et 13 d'arbres fruitiers), afin de démontrer ma capacité à collecter, préparer et analyser des données dans un contexte plus large. Un modèles a été entrainé sur ce dataset enrichi, et une API de classification d'images de plantes a été déployée. Enfin, j’ai mis en place un système de monitorage pour suivre les performances du modèle en production et détecter les éventuelles dérives.


Après avoir collecté toutes ces images, j'ai ensuite entrainé un modèle pré-entrainé EfficientNet-B3 que j'ai adapté aux classes de plantes sélectionnées de ce dataset enrichi. J'ai monitoré au cours de cet entrainement les différentes métriques de mes modèles pour pouvoir prendre la décision de quel modèle sera utilisé pour la prochaine étape. Cette prochaine étape consiste justement a déployé ce modèle sur une API pour permettre à tous et chacun de l'utiliser pour faire la classification d'images de plantes. Finalement, pour assurer la qualité à long terme de mon modèle, j'ai mis en place un système de monitorage pour suivre les performances du modèle en production et détecter les éventuelles dérives.

Pour permttre le déployement de ce modèle et permettre à tous de l'utiliser, j'ai mis en ligne avec une interface utilisateur en utilisant Streamlit, pour permettre une utilisation facile et intuitive. Tout a été réalisé en respectant les bonnes pratiques de la data science, avec une attention particulière portée à la qualité des données et à l'évaluation rigoureuse des modèles.


<br><br><br>
### Pour permettre une navigation plus fluide dans ce rapport, voici une table des matières avec des liens vers les différentes sections :








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



