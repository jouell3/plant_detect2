---
layout: default
title: "Étape 2 — Sélection du modèle"
parent: "Entrainement du modèle"
nav_order: 2
---

## Objectif de cette étape

L’objectif de cette étape est de sélectionner le modèle offrant les meilleures performances généralisées, sur la base de la précision, du rappel et du F1-score mesurés sur le jeu de validation (20 % des données initiales).

### **Précision et F1-score des modèles de classification de plantes**

Pour permettre de bien comparer les performances de mes modèles de classification de plantes, j'ai monitoré différentes métriques d’entrainement, telles que la précision et le F1-score. Cette évaluation est réalisée sur le jeu de validation (20 % des données initiales). La **précision** mesure la proportion de prédictions correctes parmi les prédictions positives, tandis que le **F1-score** combine précision et rappel pour fournir une mesure globale des performances. L’évaluation de ces deux métriques pour chaque modèle permet de déterminer lequel offre les meilleures performances globales.

Voici pour tous les modèles et toutes les catégories, la précision et le F1-score obtenus sur le jeu de validation:


![alt text](../figures/precision_heatmap_allmodels.png)
##### Figure 1 : Précision des modèles de classification de plantes par classe et par catégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories. Il est aussi possible de voir que certaines classes sont plus difficiles à classifier que d'autres, comme par exemple la classe "avocat et kiwi" qui ont une précision plus faible que les autres classes.

<br><br>

![alt text](../figures/f1_heatmap_allmodels.png)
##### Figure 2 : F1-score des modèles de classification de plantes par classe et par catégorie. On observe que les modèles ont des performances globalement élevées, avec quelques variations entre les classes et catégories. 

### **Test de McNemar — validation statistique du classement**

Les heatmaps montrent que ConvNeXt-Tiny (95.4 %) devance EfficientNet-B3 (92.9 %) d’environ 2.5 points. Mais cette différence est-elle réellement significative, ou pourrait-elle être un artefact du split de validation ?

Le **test de McNemar** répond à cette question. C’est un test statistique adapté aux comparaisons de classifieurs évalués sur les **mêmes échantillons** (données appariées). Il analyse le tableau de contingence des désaccords entre deux modèles — uniquement les cas où l’un se trompe et pas l’autre — et teste l’hypothèse nulle H₀ : *les deux modèles font le même nombre d’erreurs*.

| Comparaison | b (A✓ B✗) | c (A✗ B✓) | χ² | p-value | Conclusion |
|---|---:|---:|---:|---:|---|
| ConvNeXt-Tiny vs EfficientNet-B3 | 468 | 171 | 137.11 | < 0.0001 | **ConvNeXt-Tiny significativement meilleur** |
| EfficientNet-B3 vs EfficientNet-B4 | 335 | 324 | 0.15 | 0.697 | Différence non significative |

La première comparaison est sans ambiguïté : ConvNeXt-Tiny corrige 468 erreurs d’EfficientNet-B3 pour seulement 171 dans l’autre sens — un écart très improbable sous H₀ (p < 0.0001). En revanche, EfficientNet-B3 et B4 sont statistiquement interchangeables malgré leurs scores légèrement différents.

### **Conclusion**

Ces résultats confirment le choix de ConvNeXt-Tiny comme modèle de production. L’avance sur EfficientNet-B3 n’est pas un artefact — elle est robuste et statistiquement validée. La section suivante détaille les tests de performance approfondis réalisés sur ce modèle retenu.



