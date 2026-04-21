---
layout: default
title: "Déploiement de l'interface utilisateur"
parent: "Déploiement"
nav_order: 2
---


# Génération d'une interface utilisateur pour l'utilisation des modèles de classification de plantes


## Objectif du module

### **Introduction**
<br>
L'objectif de ce module est de déployer les modèles de classification de plantes entraînés dans les étapes précédentes, pour permettre une utilisation facile et intuitive par des personnes sans connaissance technique. Pour ce faire, j'ai choisi d'utiliser Streamlit, une bibliothèque Python qui permet de créer des applications web interactives de manière simple et rapide, avec une intégration native des outils de data science. L'interface utilisateur permet de téléverser des images de plantes, d'obtenir des prédictions et d'afficher les résultats de manière claire et intuitive.

<br>

### **Processus de création de l'interface utilisateur**
<br>

La première étape a été de configurer le répertoire pour permettre le développement du frontend de façon efficace. Tout le code frontend est isolé dans un répertoire dédié `frontend/`, ce qui facilite le développement et le déploiement indépendant de cette couche.

<br>

L'application Streamlit permet d'avoir plusieurs pages indépendantes avec un système de navigation intégré. Il faut pour cela créer un répertoire `pages/` contenant les scripts de chaque page (le nom du fichier est utilisé par Streamlit comme titre de la page), un fichier `main.py` servant de point d'entrée, et un fichier `requirements.txt` pour les dépendances au déploiement sur Streamlit Cloud.

L'application possède 4 pages :

- La page d'accueil, qui détaille les fonctionnalités de l'application et explique comment l'utiliser.
- La page de prédiction simple, à partir d'une image téléversée ou prise avec la caméra.
- La page de prédiction en batch, pour traiter plusieurs images simultanément.
- La page de monitoring en temps réel des performances du modèle en production.

<br>

### **Détails des pages de l'application Streamlit**
<br>

La page de prédiction simple permet à l'utilisateur de téléverser une image ou de la prendre avec la caméra de son téléphone. Une fois l'image soumise, les résultats des 5 modèles sont affichés côte à côte. Même si ConvNeXt-Tiny est le modèle le plus performant, les autres modèles sont inclus à titre comparatif pour les utilisateurs souhaitant explorer les différences de performance entre architectures. Lorsque la plante est identifiée, une fiche descriptive s'affiche avec des conseils de culture et, si la plante est comestible, des suggestions de recettes.

<br>

Il est également possible de faire des prédictions en batch à partir de plusieurs images téléversées. Les images sont traitées par lots de 20 par le backend, et la page se rafraîchit à chaque fois que les résultats du lot suivant sont disponibles.

<br>

La page de monitoring permet de suivre en temps réel les prédictions faites par le modèle, la confiance moyenne par classe et la latence des requêtes — données récupérées depuis le backend via l'endpoint `/metrics` et rafraîchies toutes les 10 secondes.

<br>

Voici un exemple de chacune des pages de l'application :

<br>

![Frontend main page](../figures/frontend_main.png)
##### Figure 1 : Page d'accueil de l'application Streamlit.


<br><br>
![Page single prediction](../figures/page1.png)


##### Figure 2 : Page de prédiction simple. L'utilisateur peut téléverser une image ou utiliser la caméra. Les résultats des 5 modèles s'affichent avec une fiche descriptive de la plante identifiée, des conseils de culture et des suggestions de recettes si la plante est comestible.

<br><br>

![Prediction en batch](../figures/page2.png)

##### Figure 3 : Page de prédiction en batch. Les images sont traitées par lots de 20 ; la page se rafraîchit à chaque lot.

<br><br>

![Page monitoring](../figures/page_monitoring.png)

##### Figure 4 : Page de monitoring en temps réel — prédictions récentes, confiance par classe et latence des requêtes.


### **Vote majoritaire pondéré**

<br>

Pour les pages de prédiction simple et batch, les résultats des 5 modèles sont agrégés via un **vote majoritaire pondéré** (*weighted majority vote*). Plutôt que de compter le nombre de modèles qui s'accordent sur la même prédiction, chaque modèle contribue proportionnellement à sa précision de validation :

```python
WEIGHTS = {
    "convnext_tiny":     0.954,
    "efficientnet_b3":   0.929,
    "efficientnet_b4":   0.928,
    "mobilenetv3_large": 0.879,
    "resnet50":          0.862,
}

def weighted_vote(predictions):
    # Étape 1 — sélection de l'espèce gagnante
    scores = defaultdict(float)
    for item in predictions:
        w = WEIGHTS[item["model"]]
        scores[item["top1"]["class"]] += w * item["top1"]["confidence"]
    winner = max(scores, key=scores.get)

    # Étape 2 — confiance affichée : moyenne pondérée des modèles concordants
    weight_sum = weighted_conf = 0.0
    for item in predictions:
        if item["top1"]["class"] == winner:
            w = WEIGHTS[item["model"]]
            weighted_conf += w * item["top1"]["confidence"]
            weight_sum += w
    return winner, weighted_conf / weight_sum
```

Seule la prédiction top-1 de chaque modèle est prise en compte, ce qui est cohérent avec l'endpoint `/predict-batch` qui ne retourne qu'un top-1 par modèle. L'algorithme fonctionne en deux étapes : d'abord la sélection de l'espèce gagnante par accumulation des scores pondérés, puis le calcul d'un score de confiance affiché. Ce score est la **moyenne pondérée des confiances des modèles concordants uniquement** — il reste donc ancré dans les sorties réelles des modèles et ne peut jamais dépasser la confiance individuelle la plus élevée parmi ceux qui s'accordent. ConvNeXt-Tiny (95.4 %) pèse davantage que ResNet-50 (86.2 %) dans les deux étapes.

#### Exemple concret

| Modèle | Poids | Top-1 | Confiance |
|---|---|---|---|
| ConvNeXt-Tiny | 0.954 | **Fraise** | 69.5 % |
| EfficientNet-B3 | 0.929 | **Fraise** | 59.5 % |
| EfficientNet-B4 | 0.928 | Lisianthus | 17.2 % |
| MobileNetV3 | 0.879 | **Fraise** | 24.0 % |
| ResNet-50 | 0.862 | Zinnia | 16.5 % |

**Étape 1 — sélection du gagnant :**

| Espèce | Score accumulé |
|---|---|
| Fraise | 0.954×0.695 + 0.929×0.595 + 0.879×0.240 = **1.43** |
| Lisianthus | 0.928×0.172 = 0.16 |
| Zinnia | 0.862×0.165 = 0.14 |

→ **Fraise** remporte le vote.

**Étape 2 — confiance affichée (moyenne pondérée des modèles concordants) :**

```
(0.954×0.695 + 0.929×0.595 + 0.879×0.240) / (0.954 + 0.929 + 0.879) = 1.43 / 2.76 ≈ 51.7 %
```

Le score affiché (51.7 %) reste ancré dans les confiances réelles des modèles qui s'accordent, sans jamais dépasser la confiance individuelle la plus élevée (69.5 % pour ConvNeXt-Tiny).

<br>

### Conclusion

<br>

L'interface Streamlit rend les modèles de classification accessibles sans aucune connaissance technique. Elle offre trois modes d'utilisation (prédiction simple, batch, monitoring) et s'intègre directement avec le backend FastAPI déployé sur Cloud Run.


<br><br><br>

---
Application accessible à l'adresse suivante :

## [Plant detect application](https://plantpredict.streamlit.app/).
