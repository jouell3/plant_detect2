---
layout: default
title: "Entrainement du modèle"
nav_order: 3
has_children: true
has_toc: false
---



# Entraînement des modèles de classification de plantes

<br>

### *Objectif du module*

Après avoir collecté toutes les images de plantes aromatiques, fleurs et arbres fruitiers, j'ai entraîné des modèles de type **EfficientNet-B3/B4, ResNet-50, MobileNetV3-Large et ConvNeXt-Tiny** que j'ai adapté aux classes de plantes sélectionnées. J'ai monitoré au cours de cet entrainement les différentes métriques de mes modèles pour pouvoir prendre la décision de quel modèle sera utilisé pour la prochaine étape. Cette prochaine étape consiste justement a déployé ce modèle sur une API pour permettre à tous et chacun de l'utiliser pour faire la classification d'images de plantes.

<br>

### **Introduction**

L’entrainement du modèle de classification de plantes est une étape cruciale dans le projet, car c’est à ce moment que le modèle apprend à reconnaître les différentes classes de plantes à partir des images collectées. La première étape a été de comparer différents modèles de classification. J’ai utilisé 5 modèles pré-entrainé :

- EfficientNet-B3
- EfficientNet-B4
- ResNet-50
- ConvNeXt-Tiny
- MobileNetV3-Large

<br>

Ces 5 modèles sont des architectures convolutionnelles efficaces et performantes, couramment utilisées pour la classification d’images. Ils ont cependant tous leurs forces et faiblesses, et il est difficile, voire impossible, de prédire a priori lequel performera le mieux sur cette tâche de classification de plantes. Même s’ils ont été pré-entraînés, il était nécessaire de les adapter aux classes de plantes sélectionnées pour déterminer lequel offre les meilleures performances. 

<br>

Comme ces modèles ont déjà été entraînés sur ImageNet, j'ai utilisé la technique du transfer learning pour les adapter à mon dataset de plantes. Ces architectures se composent de deux parties : le **backbone** (extracteur de caractéristiques, dont les poids sont pré-entraînés sur plus de 1 000 classes ImageNet et permettent de générer des embeddings visuels) et la **tête de classification** (couche linéaire finale, adaptée au nombre de classes cibles et terminée par une activation softmax). Le processus d'adaptation consiste à remplacer la tête de classification d'origine par une nouvelle couche correspondant aux 58 classes du dataset, puis à affiner les poids du modèle via le fine-tuning. 
<br>

Tout au long du processus d'entrainement, j'ai monitoré les différentes métriques d’entrainement, telles que la précision, le rappel et le F1-score, pour évaluer les performances des modèles et prendre la décision de quel modèle sera utilisé pour la prochaine étape de déploiement sur une API. L’objectif de cette étape est d’obtenir un modèle performant et robuste pour la classification d’images de plantes, qui pourra être mis à disposition de tout utilisateur souhaitant identifier une espèce à partir d’une image.

### **Architecture des modèles**

Tous ces modèles sont basés sur des architectures de convolution efficaces et performantes, qui ont été pré-entrainées sur ImageNet. 

## Comparatif des architectures utilisées

| Modèle              | Backbone (Queue)                       | Tête (Head)                           | Input recommandé | Embedding | Paramètres | Points clés                     |
|---------------------|-----------------------------------------|----------------------------------------|------------------|-----------|------------|----------------------------------|
| **EfficientNet‑B3** | MBConv + SE + scaling                   | GAP + Dropout + Linear(1536 → C)       | 300×300          | 1536      | ~12M       | Excellent ratio perf/poids       |
| **EfficientNet‑B4** | MBConv + SE + scaling ↑                 | GAP + Dropout + Linear(1792 → C)       | 380×380          | 1792      | ~19M       | Plus précis mais plus lourd      |
| **ResNet‑50**       | Bottleneck blocks + skip connections    | GAP + Linear(2048 → C)                 | 224×224          | 2048      | ~25.6M     | Architecture robuste et stable   |
| **ConvNeXt‑Tiny**   | Conv large kernel (7×7) + LayerNorm     | GAP + LayerNorm + Linear(768 → C)      | 224×224          | 768       | ~28M       | Moderne, très performant         |
| **MobileNetV3‑Large** | Depthwise + SE + h‑swish              | GAP + Linear(960 → 1280 → C)           | 224×224          | 1280      | ~5.4M      | Ultra‑léger, optimisé mobile     |


<br><br><br>
---

Pour plus de détail, vous pouvez consulter les différentes étapes de cette section d'entrainement du modèle de classification de plantes, qui sont détaillées dans les sous-sections suivantes (aussi présentes dans la barre latérale gauche): 

- [Étape 1 — Sélection du modèle](training/step2_model_selection.md)
- [Étape 2 — Évaluation du modèle de classification de plantes](training/step3_evaluation.md)
- [Étape 3 — Entrainement du modèle final](training/step4_final_training.md)



