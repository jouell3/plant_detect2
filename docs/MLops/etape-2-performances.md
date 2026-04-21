---
layout: default
title: "Étape 2 — Tests de performance"
parent: "Monitoring / MLOps"
nav_order: 2
---


## Objectif de cette étape

Il a été présenté dans la section précédente que le modèle ConvNeXt-Tiny a obtenu les meilleures performances globales en termes de précision et de F1-score. Dans cette section, je vais pousser un peu plus les tests sur ce modèle pour évaluer ses performances de manière plus détaillée. Des tests classiques pour évaluer ce modèle ont été faits et seront plus détaillés dans cette section, qui est dédiée à l'évaluation du modèle de classification de plantes.

### **Analyse en détail des performances du modèle de classification de plantes choisi**

La première étape que j'ai voulu voir est les détails des scores de précision et de F1-score pour chaque classe et catégories. Ceci va me permettre de voir si le modèle avait des performances plus faibles pour certaines classes, ce qui pourrait indiquer des problèmes spécifiques à ces classes. Ces tests ont été effectués à partir du jeu de données de validation, qui a été séparé du jeu de données d'entrainement (80% des données initiales). Voici pour ce modèle les détails des scores de précision et de F1-score pour chaque classe : 

<br>

![precision](../figures/precision_bycategory.png)
##### Figure 1 : Précision des modèles par classe et par catégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories.

<br>

![F1_score](../figures/F1score_bycategory.png)
##### Figure 2 : F1-score des modèles par classe et par catégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories.

Globalement, on observe que les performances du modèle ConvNeXt-Tiny sont élevées pour la plupart des classes (>80% de précision pour toutes les classes). Il est intéressant de noter que certaines classes ont des performances légèrement plus faibles que d'autres, ce qui pourrait indiquer des problèmes spécifiques à ces classes. Cette baisse de performance pour certaines classes pourrait être due à plusieurs facteurs, tels que la qualité des images, la complexité de la classe (le fait que cette classe soit plus similaire à une autre), ou encore l'imbalance des classes dans le dataset (par example, il y avait beaucoup moins d'image de kiwi, ce qui pourrait expliquer le fait que les performances soient plus faibles). 

Il pourrait être intéressant de faire une analyse plus approfondie pour comprendre les raisons de ces variations de performance entre les classes, et éventuellement trouver des moyens d'améliorer les performances pour ces classes spécifiques. Cependant, dans l'ensemble, le modèle ConvNeXt-Tiny a obtenu des performances élevées pour la classification d’images de plantes, ce qui est un bon indicateur de sa capacité à généraliser à de nouvelles données.

### **Tests de performance du modèle sélectionné**

Pour s'assurer qu'il n'y a pas eu de surapprentissage durant la phase initiale d'entrainement, il est possible de faire une analyse de type K-fold stratified cross-validation. Cette technique d’évaluation de la performance d’un modèle de machine learning consiste à diviser le dataset en K sous-ensembles (ou "folds") de manière stratifiée, c’est-à-dire en respectant la distribution des classes dans chaque fold. Le modèle est ensuite entrainé K fois, chaque fois en utilisant K-1 folds pour l’entrainement et le fold restant pour le test. Les performances du modèle (accuracy) est ensuite déterminée pour chaque fold, la différence entre ces K folds indique si le modèle est robuste.


![Kfold statifié](../figures/Kfold_strat_all.png)
##### Figure 3 : Résultats de la K-fold stratified cross-validation.

Il est possible de voir que les 5 K-fold d'entrainement du modèle ConvNeXt-Tiny ont obtenu des performances élevées, avec une précision moyenne de 94.6% et une faible variance entre les folds (écart-type de 0.004). Cela indique que le modèle est robuste et qu'il n'y a pas eu de surapprentissage durant la phase initiale d'entrainement. En utilisant K-fold stratified cross-validation, j'ai pu évaluer les performances du modèle sélectionné de manière plus fiable, en prenant en compte la variabilité des données et en fournissant une évaluation plus robuste de la performance du modèle sur des données non vues. 

### **Division du dataset en train/validation/test**

Une approche complémentaire pour évaluer les performances du modèle de classification de plantes sélectionné est de faire une division du dataset en train/validation/test plus classique, avec un split de 70% pour l'entrainement, 15% pour la validation et 15% pour le test. Cette approche permet d'obtenir une estimation plus fiable de la performance du modèle sur des données non vues, en réduisant le risque de surapprentissage (overfitting) et en fournissant une évaluation plus robuste de la performance du modèle sur des données jamais vues durant l'entrainement. En utilisant cette approche, j'ai pu vérifier que les résultats obtenus avec la K-fold stratified cross-validation sont similaires à ceux obtenus avec un split plus classique du dataset, ce qui confirme la robustesse des résultats et la performance du modèle sélectionné pour la classification d’images de plantes.


![Kfold stratifié split 70-15-15](../figures/Kfold_strat_all_split70_15_15.png)
##### Figure 4 : Résultats de la K-fold stratified cross-validation avec un split plus classique du dataset (70% train, 15% validation, 15% test).

### **Run Summary (detailed)**

| Metric                     | Score     | Interpretation |
|----------------------------|-----------|----------------|
| **Test Accuracy**          | **0.94627** | Très haute précision globale |
| **F1‑Score Macro**         | 0.94478   | Bon équilibre entre classes |
| **Precision Macro**        | 0.94521   | Peu de faux positifs |
| **Recall Macro**           | 0.94480   | Peu de faux négatifs |
| **Val–Test Gap**           | 0.00945   | Excellente généralisation |



### **Résultats par classe — ConvNeXt-Tiny (test set)**

Points notables :
- **Classes parfaites :** `borage` et `lemongrass` atteignent 1.00 en précision, rappel et F1
- **Classes les plus difficiles :** `chrysanthemum` (F1 0.79, recall 0.73), `hydrangea` (0.83) et `allium` (0.84) — toutes des fleurs visuellement similaires entre elles
- **Classes sous-représentées :** `kiwi` (88 images) et `lovage` (85 images) obtiennent malgré tout des F1 corrects (0.86 et 0.98), ce qui valide la robustesse du filtrage automatique

| Classe | Catégorie | Précision | Rappel | F1 | Support |
|---|---|---:|---:|---:|---:|
| chrysanthemum | fleurs | 0.85 | 0.73 | **0.79** | 200 |
| hydrangea | fleurs | 0.87 | 0.79 | **0.83** | 200 |
| allium | fleurs | 0.88 | 0.81 | **0.84** | 200 |
| blackberry | fruits | 0.90 | 0.81 | **0.85** | 200 |
| pear | fruits | 0.82 | 0.90 | **0.85** | 200 |
| apple | fruits | 0.87 | 0.85 | **0.86** | 200 |
| kiwi | fruits | 0.84 | 0.89 | **0.86** | 88 |
| lily | fleurs | 0.85 | 0.92 | **0.88** | 200 |
| cherry | fruits | 0.84 | 0.91 | **0.88** | 191 |
| iris | fleurs | 0.93 | 0.88 | **0.90** | 200 |
| fig | fruits | 0.88 | 0.93 | **0.90** | 200 |
| mango | fruits | 0.88 | 0.92 | **0.90** | 200 |
| ranunculus | fleurs | 0.87 | 0.83 | **0.85** | 200 |
| wisteria | fleurs | 0.95 | 0.89 | **0.92** | 200 |
| freesia | fleurs | 0.95 | 0.89 | **0.92** | 200 |
| blueberry | fruits | 0.91 | 0.94 | **0.92** | 200 |
| cranberry | fruits | 0.91 | 0.95 | **0.93** | 200 |
| coriander | aromates | 0.91 | 0.96 | **0.93** | 181 |
| melon | fruits | 0.93 | 0.94 | **0.93** | 200 |
| oregano | aromates | 0.94 | 0.93 | **0.94** | 213 |
| parsley | aromates | 0.94 | 0.94 | **0.94** | 219 |
| dill | aromates | 0.95 | 0.93 | **0.94** | 214 |
| hellebore | fleurs | 0.96 | 0.93 | **0.94** | 200 |
| lemonverbena | aromates | 0.96 | 0.94 | **0.95** | 189 |
| mugwort | aromates | 0.93 | 0.98 | **0.95** | 220 |
| mint | aromates | 0.95 | 0.95 | **0.95** | 239 |
| gypsophila | fleurs | 0.97 | 0.94 | **0.95** | 200 |
| daisy | fleurs | 0.96 | 0.94 | **0.95** | 200 |
| basil | aromates | 0.97 | 0.96 | **0.96** | 224 |
| chamomile | aromates | 0.93 | 0.99 | **0.96** | 226 |
| chives | aromates | 0.95 | 0.97 | **0.96** | 316 |
| fennel | aromates | 0.95 | 0.96 | **0.96** | 232 |
| cosmos | fleurs | 0.98 | 0.94 | **0.96** | 200 |
| foxglove | fleurs | 0.98 | 0.94 | **0.96** | 200 |
| gerbera | fleurs | 0.95 | 0.98 | **0.96** | 200 |
| tarragon | aromates | 0.96 | 0.97 | **0.96** | 203 |
| angelica | aromates | 0.95 | 0.99 | **0.97** | 193 |
| poppy | fleurs | 0.97 | 0.96 | **0.97** | 200 |
| sage | aromates | 0.96 | 0.98 | **0.97** | 213 |
| sunflower | fleurs | 0.98 | 0.94 | **0.96** | 200 |
| zinnia | fleurs | 0.98 | 0.96 | **0.97** | 200 |
| grape | fruits | 0.88 | 0.90 | **0.89** | 200 |
| peach | fruits | 0.88 | 0.88 | **0.88** | 200 |
| raspberry | fruits | 0.87 | 0.92 | **0.90** | 200 |
| strawberry | fruits | 0.94 | 0.99 | **0.97** | 200 |
| avocado | fruits | 0.88 | 0.84 | **0.86** | 200 |
| lemon | fruits | 0.86 | 0.88 | **0.87** | 200 |
| hyssop | aromates | 0.97 | 0.99 | **0.98** | 227 |
| lavender | aromates | 0.98 | 0.98 | **0.98** | 259 |
| lovage | aromates | 0.98 | 0.99 | **0.98** | 85 |
| savory | aromates | 0.96 | 1.00 | **0.98** | 122 |
| thyme | aromates | 0.98 | 0.98 | **0.98** | 283 |
| lisianthus | fleurs | 0.98 | 0.98 | **0.98** | 200 |
| bird_of_paradise | fleurs | 0.98 | 0.99 | **0.98** | 151 |
| rosemary | aromates | 0.99 | 0.99 | **0.99** | 246 |
| wintergreen | aromates | 0.97 | 1.00 | **0.99** | 220 |
| lemongrass | aromates | 1.00 | 1.00 | **1.00** | 208 |
| borage | aromates | 1.00 | 1.00 | **1.00** | 219 |

---

### **Confiance moyenne par classe et par modèle**

Le tableau ci-dessous montre la confiance moyenne du modèle sur les prédictions correctes pour chaque classe. ConvNeXt-Tiny est systématiquement le plus confiant. Les classes présentant la confiance la plus faible (`chives`, `thyme`, `blackberry`) correspondent aux mêmes classes difficiles identifiées dans la matrice de confusion.

| Classe | ConvNeXt-Tiny | EfficientNet-B3 | EfficientNet-B4 | MobileNetV3 | ResNet-50 |
|---|---:|---:|---:|---:|---:|
| chives | 0.803 | 0.762 | 0.778 | 0.695 | 0.725 |
| thyme | 0.828 | 0.774 | 0.780 | 0.730 | 0.673 |
| blackberry | 0.842 | 0.789 | 0.757 | 0.657 | 0.562 |
| apple | 0.844 | 0.806 | 0.771 | 0.696 | 0.543 |
| lavender | 0.844 | 0.802 | 0.781 | 0.729 | 0.734 |
| mint | 0.845 | 0.791 | 0.776 | 0.684 | 0.662 |
| poppy | 0.847 | 0.830 | 0.818 | 0.781 | 0.732 |
| dill | 0.848 | 0.784 | 0.811 | 0.772 | 0.751 |
| chamomile | 0.848 | 0.816 | 0.823 | 0.746 | 0.750 |
| chrysanthemum | 0.849 | 0.835 | 0.743 | 0.624 | 0.527 |
| peach | 0.852 | 0.840 | 0.787 | 0.692 | 0.581 |
| lemon | 0.852 | 0.761 | 0.801 | 0.701 | 0.620 |
| fennel | 0.853 | 0.820 | 0.789 | 0.769 | 0.739 |
| gypsophila | 0.853 | 0.806 | 0.805 | 0.761 | 0.686 |
| rosemary | 0.853 | 0.823 | 0.820 | 0.762 | 0.744 |
| grape | 0.854 | 0.803 | 0.782 | 0.709 | 0.620 |
| borage | 0.854 | 0.810 | 0.819 | 0.772 | 0.763 |
| lemongrass | 0.855 | 0.821 | 0.811 | 0.797 | 0.783 |
| oregano | 0.855 | 0.793 | 0.791 | 0.745 | 0.695 |
| mango | 0.855 | 0.815 | 0.759 | 0.717 | 0.640 |
| lemonverbena | 0.856 | 0.842 | 0.824 | 0.768 | 0.688 |
| avocado | 0.857 | 0.800 | 0.800 | 0.680 | 0.551 |
| parsley | 0.857 | 0.803 | 0.792 | 0.728 | 0.675 |
| iris | 0.857 | 0.797 | 0.815 | 0.720 | 0.679 |
| melon | 0.858 | 0.803 | 0.778 | 0.760 | 0.701 |
| lily | 0.858 | 0.841 | 0.803 | 0.756 | 0.648 |
| basil | 0.858 | 0.819 | 0.787 | 0.764 | 0.709 |
| mugwort | 0.860 | 0.797 | 0.785 | 0.758 | 0.670 |
| foxglove | 0.860 | 0.812 | 0.766 | 0.764 | 0.730 |
| wisteria | 0.861 | 0.800 | 0.781 | 0.726 | 0.648 |
| pear | 0.862 | 0.809 | 0.762 | 0.743 | 0.636 |
| ranunculus | 0.862 | 0.843 | 0.785 | 0.674 | 0.594 |
| lisianthus | 0.864 | 0.834 | 0.837 | 0.751 | 0.764 |
| zinnia | 0.866 | 0.846 | 0.829 | 0.783 | 0.755 |
| cranberry | 0.866 | 0.815 | 0.812 | 0.784 | 0.709 |
| hyssop | 0.866 | 0.807 | 0.802 | 0.765 | 0.733 |
| angelica | 0.867 | 0.840 | 0.827 | 0.786 | 0.739 |
| allium | 0.867 | 0.820 | 0.783 | 0.681 | 0.634 |
| tarragon | 0.867 | 0.810 | 0.785 | 0.767 | 0.698 |
| fig | 0.868 | 0.781 | 0.798 | 0.740 | 0.659 |
| strawberry | 0.870 | 0.822 | 0.820 | 0.800 | 0.774 |
| sage | 0.870 | 0.781 | 0.844 | 0.783 | 0.721 |
| raspberry | 0.870 | 0.829 | 0.774 | 0.699 | 0.609 |
| wintergreen | 0.871 | 0.817 | 0.812 | 0.776 | 0.777 |
| coriander | 0.872 | 0.802 | 0.823 | 0.781 | 0.756 |
| hydrangea | 0.872 | 0.805 | 0.792 | 0.684 | 0.540 |
| gerbera | 0.873 | 0.829 | 0.819 | 0.782 | 0.776 |
| hellebore | 0.873 | 0.805 | 0.816 | 0.761 | 0.699 |
| daisy | 0.875 | 0.829 | 0.828 | 0.780 | 0.761 |
| cosmos | 0.875 | 0.834 | 0.824 | 0.787 | 0.767 |
| cherry | 0.880 | 0.834 | 0.798 | 0.700 | 0.635 |
| freesia | 0.882 | 0.827 | 0.801 | 0.719 | 0.681 |
| blueberry | 0.883 | 0.817 | 0.814 | 0.748 | 0.697 |
| sunflower | 0.886 | 0.831 | 0.829 | 0.788 | 0.746 |
| bird_of_paradise | 0.908 | 0.875 | 0.861 | 0.820 | 0.817 |
| kiwi | 0.912 | 0.922 | 0.913 | 0.897 | 0.805 |
| savory | 0.928 | 0.920 | 0.919 | 0.905 | 0.871 |
| lovage | 0.950 | 0.914 | 0.926 | 0.919 | 0.862 |

---

### **Conclusion**

Tous ces tests de performance ont permis de confirmer que le modèle ConvNeXt-Tiny sélectionné pour la classification d’images de plantes est robuste et performant, avec une précision élevée et une bonne capacité à généraliser à de nouvelles données. En utilisant différentes techniques d’évaluation, telles que la K-fold stratified cross-validation et un split plus classique du dataset, j'ai pu obtenir une évaluation plus fiable de la performance du modèle sur des données non vues, ce qui est crucial pour garantir que le modèle sera efficace une fois déployé sur une API et mis à disposition de tout utilisateur souhaitant identifier une espèce de plante.