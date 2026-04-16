---
layout: default
title: "Entrainement du modèle"
nav_order: 3
has_children: true
has_toc: false
---



# Entrainement du modèle de classification de plantes


### 🎯 *Objectif du module*

Après avoir collecté toutes les images de plantes aromatiques, fleurs et arbres fruitiers, j'ai entrainé un modèle de type EfficientNet-B3 que j'ai adapté aux classes de plantes sélectionnées. J'ai monitoré au cours de cet entrainement les différentes métriques de mes modèles pour pouvoir prendre la décision de quel modèle sera utilisé pour la prochaine étape. Cette prochaine étape consiste justement a déployé ce modèle sur une API pour permettre à tous et chacun de l'utiliser pour faire la classification d'images de plantes.


## Introduction

L’entrainement du modèle de classification de plantes est une étape cruciale dans le projet, car c’est à ce moment que le modèle apprend à reconnaître les différentes classes de plantes à partir des images collectées. La première étape a été de comparer différents modèles de classification. J’ai utilisé 5 modèles pré-entrainé :

- EfficientNet-B3
- EfficientNet-B4
- ResNet-50
- ConvNeXt-Tiny
- MobileNetV3-Large

C'est 5 modèles sont des modèles de convolution efficace et performant qui sont courrament utiliser pour la classification d’images. Cependant, ils ont tous leurs forces et faiblesse et il est très difficile voir impossible à prédire lequel performera le mieux. Même s'ils ont été pré-entrainés, il fallait que je les adapte aux classes de plantes sélectionnées pour déterminer lequel performera le mieux. 

Comme ces modèles ont déjà été entrainé sur ImageNet, j'ai utilisé la technique de fine-tuning pour adapter ces modèles à mon dataset de plantes. J'ai ajouté une couche de classification adaptée au nombre de classes de mon dataset, et j'ai entrainé ces modèles sur le dataset enrichi de plantes aromatiques, fleurs et arbres fruitiers, en utilisant des techniques d’augmentation de données pour améliorer la généralisation du modèle. J'ai monitoré les différentes métriques d’entrainement, telles que la précision, le rappel et le F1-score, pour évaluer les performances du modèle et prendre la décision de quel modèle sera utilisé pour la prochaine étape de déploiement sur une API. L’objectif de cette étape est d’obtenir un modèle performant et robuste pour la classification d’images de plantes, qui pourra être utilisé par tous et chacun pour identifier les différentes classes de plantes à partir de leurs images.

## Architecture des modèles

Tous ces modèles sont basés sur des architectures de convolution efficaces et performantes, qui ont été pré-entrainées sur ImageNet. Ils consistent d'une "tête" de convolution qui extrait les caractéristiques visuelles des images, suivie d'une "queue" de classification qui prédit la classe de l'image à partir de ces caractéristiques (qui se termine par une couche "softmax" qui a autant de sorties que de classes).

# Comparatif des architectures utilisées

| Modèle              | Backbone (Queue)                       | Tête (Head)                           | Input recommandé | Embedding | Paramètres | Points clés                     |
|---------------------|-----------------------------------------|----------------------------------------|------------------|-----------|------------|----------------------------------|
| **EfficientNet‑B3** | MBConv + SE + scaling                   | GAP + Dropout + Linear(1536 → C)       | 300×300          | 1536      | ~12M       | Excellent ratio perf/poids       |
| **EfficientNet‑B4** | MBConv + SE + scaling ↑                 | GAP + Dropout + Linear(1792 → C)       | 380×380          | 1792      | ~19M       | Plus précis mais plus lourd      |
| **ResNet‑50**       | Bottleneck blocks + skip connections    | GAP + Linear(2048 → C)                 | 224×224          | 2048      | ~25.6M     | Architecture robuste et stable   |
| **ConvNeXt‑Tiny**   | Conv large kernel (7×7) + LayerNorm     | GAP + LayerNorm + Linear(768 → C)      | 224×224          | 768       | ~28M       | Moderne, très performant         |
| **MobileNetV3‑Large** | Depthwise + SE + h‑swish              | GAP + Linear(960 → 1280 → C)           | 224×224          | 1280      | ~5.4M      | Ultra‑léger, optimisé mobile     |


## EfficientNet‑B3 / B4
- **Backbone** : MBConv + Squeeze‑and‑Excitation  
- **Forces** : efficacité, précision, faible coût  
- **Entrée** : B3 → 300×300, B4 → 380×380  
- **Embedding** : 1536 (B3), 1792 (B4)  
- **Usage idéal** : classification multi‑classe haute précision  

## ResNet‑50
- **Backbone** : blocs bottleneck + résidual connections  
- **Forces** : stabilité, généralisation  
- **Entrée** : 224×224  
- **Embedding** : 2048  
- **Usage idéal** : baseline robuste, facile à fine‑tuner  

## ConvNeXt‑Tiny
- **Backbone** : convolution large kernel (7×7), LayerNorm  
- **Forces** : performance proche des Transformers  
- **Entrée** : 224×224  
- **Embedding** : 768  
- **Usage idéal** : textures complexes, datasets variés  

## MobileNetV3‑Large
- **Backbone** : depthwise separable + SE + h‑swish  
- **Forces** : ultra‑léger, très rapide  
- **Entrée** : 224×224  
- **Embedding** : 1280  
- **Usage idéal** : mobile / edge computing  


 J'ai adapté ces modèles en ajoutant une couche de classification adaptée au nombre de classes de mon dataset, et j'ai entrainé ces modèles sur le dataset enrichi de plantes aromatiques, fleurs et arbres fruitiers, en utilisant des techniques d’augmentation de données pour améliorer la généralisation du modèle.

## 🧩 Étape 1 — Extraction d’embeddings avec EfficientNet‑B3

J’ai utilisé EfficientNet‑B3 pré‑entraîné sur ImageNet pour extraire des embeddings de dimension 1536 à partir des images du dataset. Ces embeddings capturent les caractéristiques visuelles des plantes, et servent de base pour les étapes suivantes.

# Résults


![precision](figures/precision_bycategory.png)
#### Figure 1 : Precision des modèles par classe et parcatégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories.

<br><br><br>

![F1_score](figures/F1score_bycategory.png)
#### Figure 1 : F1-score des modèles par classe et par catégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories.



