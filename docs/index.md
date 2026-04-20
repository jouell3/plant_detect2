---
layout: default
title: "index"
nav_order: 1
---



# Détection automatique de plantes par computer vision

## Introduction et Expression du besoin

### Context du projet

Pour cette certification, j’ai choisi de construire un classifieur d’images de plantes, avec un focus particulier sur les **fleurs, les aromates et les arbres fruitiers**. Ce travail a été initié lors du projet final de la formation Data Science et AI donné par Artiefact School of data. Avec 3 autres de mes colègues de promotion, nous avons collecté un dataset de plus de 27 000 images réparties sur 23 classes différentes d'aromate que nous avions utilisées porur entrainer nos 4 modèles différents de reconnaissance d'aromates. 


### Expression du besoin

Bien que cette première application fonctionnait plutôt bien, elle était limitée : seulement 23 espères de plantes étaient reconnues pour ce premier modèles et la plupart des herbes aromatiques reconnues  pour nos mod;les étaient bien connues des gens. J’ai donc voulu aller plus loin pour la certification en enrichissant ce dataset avec des classes de fleurs et d'arbres fruitiers, afin de permetttre au modèle de déterminer l'espère de plus de plantes. Cet outil de classification d'images de plantes pourrait être très utile pour les jardiniers amateurs, les botanistes, les agriculteurs, ou même les simples curieux qui veulent identifier une plante qu'ils ont trouvée dans la nature. 

J'ai aussi rencontré un soucis lors de la collecte des images pour ce projet de groupe. J'ai utilisé une approche qui n'était probablement pas idéale ; un premier modèle avait été entrainené avec les premières images triées manuellement (environ 8000 images), et utilisé ensuite pour faire la selection des autres images du dataset (en selectionnant seulement les images qui correspondait à la classe prédite). Cette approach a probablement introduit un biais puisque les images sélectionnées pour l'entraînement du modèle sont ensuite utilisées pour faire la sélection des autres images du dataset, ce qui peut conduire à une sur-représentation de certains types d'images et à une sous-représentation d'autres types d'images. 

Les objectifs de ce projet de certification sont donc les suivants :

- Enrichir le dataset de plantes avec des classes de fleurs et d'arbres fruitiers, pour permettre au modèle de déterminer l'espère de plus de plantes.
- Mettre en place un processus de filtrage automatique des données pour éviter les biais liés au self-training, et pour garantir la qualité et la pertinence des images collectées pour l'entraînement du modèle final.
- Entraîner quelques modèles de classification de plantes sur ce dataset enrichi, et sélectionner le meilleur modèle
- Déployer sur une API ce modèle pour permettre à tous et chacun de l'utiliser pour faire la classification d'images de plantes.
- Mettre en place un système de monitorage pour suivre les performances du modèle en production et détecter les éventuelles dérives.  


Ce rapport a été divisé en plusieurs sections pour présenter de manière claire et structurée les différentes étapes du projet, depuis la collecte des données jusqu'au déploiement du modèle. Chaque section détaille les méthodes utilisées, les résultats obtenus et les leçons apprises tout au long du processus. Il est possible de naviger facilement entre les sections grâce à la table des matières présente sur la bordure gauche de ce rapport.
