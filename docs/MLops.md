---
layout: default
title: "Monitoring / MLOps"
nav_order: 5
has_children: true
has_toc: false
---


# Monitoring des modèles de classification de plantes durant l'entrainement et après la mise en production

<br>

## *Objectif du module*

<br>
Pour cette partie crutial dans le cycle de vie d'un projet de data science, j'ai mis en place un système de monitorage pour suivre les performances des différents modèles lors de la sélection du meilleur modèle mais également, j'ai mis en place des outils qui permettront de suivre le modèle choisi un fois déployé en production, pour détecter d'éventuelles dérives (et intervenir pour les corriger le plus rapidement possible). Le monitorage est essentiel pour assurer la qualité à long terme du modèle, en permettant de détecter rapidement les problèmes et de prendre des mesures correctives si nécessaire.


### **Introduction**

<br>

Le monitorage des modèles, de l'entrainement initiale de différents modèles jusqu'au déploiement en production, est une étape cruciale pour assurer la qualité et la performance du modèle à long terme. Comme le projet demandais beaucoup de calculs (donc l'utilisation d'un GPU assez puissant), j'ai utilisé Google Colab pour faire les entrainements et les tests requis pour la selection du meilleur modèle. Comme j'ai utilisé Google Colab en ligne, il me fallait trouver une solution "Cloud" pour suivre les différentes métriques de mes modèles. Plusieurs options sont disponible pour faire du monitorage de modèles en ligne, comme Weights & Biases, MLflow, Neptune.ai, Hugginface etc. J'ai choisi d'utiliser Weights & Biases pour suivre les différentes métriques de mes modèles lors de la sélection du meilleur modèle, car c'est une plateforme très populaire et facile à utiliser pour le suivi des expériences de machine learning (et surtout, ils offrent un mois gratuit lors de l'inscription, ce qui m'a permit de l'utiliser lors de la préparation de cette certification). Weights & Biases m'a permis de suivre les différentes métriques d'entrainement (précision, rappel, F1-score) en temps réel lors des entrainements mais également après, lors de la mise en production.

<br><br>

Comme pour MlFlow (qui a été l'option discuté durant la formation), Weights & Biases offre aussi des outils pour le monitorage durant la selection mais également après, en production, une fois que le ou les modèles sont déployés. Weights & Biases offre des outils pour suivre les métriques de performance du modèle en temps réel, ainsi que des alertes pour détecter rapidement les problèmes et intervenir pour les corriger le plus rapidement possible. Le monitorage est essentiel pour assurer la qualité à long terme du modèle, en permettant de détecter rapidement les problèmes et de prendre des mesures correctives si nécessaire. 

<br>

Pour plus de détails sur les différentes étapes, voici le menu de navigation (aussi présent dans la barre latérale gauche) pour cette section de monitorage des modèles:

- [Étape 1 — Monitorage durant la sélection du modèle](MLops/etape-1-selection.md)
- [Étape 2 — Tests de performance des différents modèles](MLops/etape-2-performances.md)
- [Étape 3 — Monitorage en production](MLops/etape-3-production.md)  