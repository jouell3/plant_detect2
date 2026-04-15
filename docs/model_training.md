---
layout: default
---



# Entrainement du modèle de classification de plantes

### 🎯 *Objectif du module*

Après avoir collecté toutes les images de plantes aromatiques, fleurs et arbres fruitiers, j'ai entrainé un modèle de type EfficientNet-B3 que j'ai adapté aux classes de plantes sélectionnées. J'ai monitoré au cours de cet entrainement les différentes métriques de mes modèles pour pouvoir prendre la décision de quel modèle sera utilisé pour la prochaine étape. Cette prochaine étape consiste justement a déployé ce modèle sur une API pour permettre à tous et chacun de l'utiliser pour faire la classification d'images de plantes.



## 🧩 Étape 1 — Extraction d’embeddings avec EfficientNet‑B3

J’ai utilisé EfficientNet‑B3 pré‑entraîné sur ImageNet pour extraire des embeddings de dimension 1536 à partir des images du dataset. Ces embeddings capturent les caractéristiques visuelles des plantes, et servent de base pour les étapes suivantes.

# Résults


![precision](../data/figures/precision_bycategory.png)
#### Figure 1 : Precision des modèles par classe et parcatégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories.

<br><br><br>

![F1_score](../data/figures/F1score_bycategory.png)
#### Figure 1 : F1-score des modèles par classe et par catégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories.



---
<br><br><br>



<div align="center">

| ⬅ [Collect de données](data_collection.md) | [⬆ Main page](index.md) | [Déploiement du modèle ➡](deploiement.md) |
|-------------------------------|---------------------|-------------------------|

</div>