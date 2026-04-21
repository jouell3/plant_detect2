---
layout: default
title: "Étape 3 — Monitorage en production"
parent: "Monitoring / MLOps"
nav_order: 3
---


## Objectif de cette étape

Une fois le modèle sélectionné et évalué, il est important de continuer à suivre ses performances une fois déployé en production. Le monitoring en production permet de s'assurer que le modèle continue à généraliser correctement sur de nouvelles données et de détecter rapidement toute dérive de performance. Dans cette section, je présente les techniques de monitoring mises en place pour suivre les performances du modèle en production.

### **Monitoring des performances du modèle en production**

De la même façon que les métriques sont suivies durant l'entraînement, il est possible de monitorer les performances du modèle une fois déployé. J'ai mis en place plusieurs mécanismes de suivi, principalement axés sur la latence des prédictions et la confiance des sorties du modèle pour chaque classe. Ces indicateurs permettent de détecter rapidement tout changement de comportement en production — par exemple, une baisse de confiance sur certaines classes pourrait indiquer un problème de distribution des données entrantes.

À plusieurs points de l'API, des logs sont émis pour chaque prédiction : classe prédite, score de confiance et temps de réponse. Ces données sont transmises en temps réel à Weights & Biases, ce qui permet un suivi centralisé des performances en production.

<br><br><br>

![example de monitoring de la confiance](../figures/monitoring.png)
##### Figure 1 : Exemple de monitoring en production — suivi de la confiance des prédictions par modèle.

![example de monitoring de la latence](../figures/monitoring_latency.png)
##### Figure 2 : Exemple de monitoring de la latence des prédictions en production.

### **Conclusion**

La durée de la certification n'a pas permis d'effectuer un monitoring en production sur une longue période, mais les mécanismes mis en place permettent un suivi en temps réel des performances et une détection rapide des dérives éventuelles. Ces outils constituent une base solide pour assurer la qualité du modèle sur le long terme, une fois l'application utilisée par un public plus large.
