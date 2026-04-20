---
layout: default
title: "Étape 2 — Sélection du modèle"
parent: "Entrainement du modèle"
nav_order: 2
---

# Objectif de cette étape

Cette étape de sélection du modèle est cruciale pour déterminer quel modèle de classification de plantes sera utilisé pour la prochaine étape de déploiement sur une API. Après avoir entrainé différents modèles de classification de plantes à partir des images collectées, il est important d'évaluer leurs performances et de sélectionner le modèle qui offre les meilleures performances globales pour la classification d’images de plantes. Pour ce faire, j'ai monitoré différentes métriques d’entrainement, telles que la précision, le rappel et le F1-score, pour évaluer les performances de mes modèles et prendre la décision de quel modèle sera utilisé pour la prochaine étape de déploiement sur une API. L’objectif de cette étape est d’obtenir un modèle performant et robuste pour la classification d’images de plantes, qui pourra être utilisé par tous et chacun pour identifier les différentes classes de plantes à partir de leurs images.

## Précision er F1 score des modèles de classification de plantes

Pour permettre de bien comparer les performances de mes modèles de classification de plantes, j'ai monitoré différentes métriques d’entrainement, telles que la précision et le F1-score. Cette étape ce fait à partir du jeu de données qui a été séparé pour la validation (20% des données initiales). La **précision** est une métrique qui mesure la proportion de prédictions correctes parmi les prédictions positives, tandis que le **F1-score** est une métrique qui combine la précision et le rappel pour fournir une mesure globale de la performance du modèle. En évaluant ces métriques pour chaque modèle, j'ai pu déterminer lequel offre les meilleures performances globales pour la classification d’images de plantes.

Voici pour tous les modèles et toutes les catégories, la précision et le F1-score obtenus sur le jeu de validation:


![alt text](../figures/precision_heatmap_allmodels.png)
#### Figure 1 : Précision des modèles de classification de plantes par classe et par catégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories. Il est aussi possible de voir que certaines classes sont plus difficiles à classifier que d'autres, comme par exemple la classe "avocat et kiwi" qui ont une précision plus faible que les autres classes.

<br><br>

![alt text](../figures/f1_heatmap_allmodels.png)

#### Figure 2 : F1-score des modèles de classification de plantes par classe et par catégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories. 

## Conclusion

Il est intéressant de voir que pour certaines classes et que pour certains modèles, la précision et le F1-score sont plus faibles que pour d'autres classes et d'autres modèles. Cela peut être dû à plusieurs facteurs, tels que la qualité des images, la complexité de la classe ou encore l'imbalance des classes dans le dataset. Cependant, dans l'ensemble, les modèles ont obtenu des performances élevées pour la classification d’images de plantes, ce qui est un bon indicateur de leur capacité à généraliser à de nouvelles données. En comparant les différentes métriques d’entrainement pour chaque modèle, j'ai pu déterminer lequel offre les meilleures performances globales pour la classification d’images de plantes.

Comme le modèle ConvNeXt-Tiny a obtenu les meilleures performances globales en termes de précision et de F1-score, j'ai décidé de pousser un peu plus les tests sur ce modèle. Des tests classiques pour évaluer ce modèles ont été fait et seront plus détaillés dans la section suivante, qui est dédiée à l'évaluation du modèle de classification de plantes.



![precision](../figures/precision_bycategory.png)
#### Figure 1 : Precision des modèles par classe et parcatégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories.

<br><br><br>

![F1_score](../figures/F1score_bycategory.png)
#### Figure 1 : F1-score des modèles par classe et par catégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories.