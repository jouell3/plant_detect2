# 📘 **Documentation : Pipeline Embeddings + Clustering + XGBoost**

## 🎯 Objectif du module

L’objectif de ce module est d’améliorer la qualité du dataset avant l’entraînement du modèle final, en utilisant une approche **data‑centric** basée sur :

- l’extraction d’embeddings d’images via un modèle pré‑entraîné (EfficientNet‑B3),
- le clustering des embeddings pour repérer les images atypiques ou hors distribution,
- un classifieur XGBoost multi‑classe pour exploiter ces embeddings dans un modèle tabulaire,
- un mécanisme de filtrage automatique des nouvelles images collectées sur Internet.

Cette approche permet d’automatiser une partie du nettoyage du dataset **sans introduire le biais du self‑training**, contrairement à l’utilisation du modèle final comme filtre.

---

## 🧠 Contexte et justification

Lors de la construction du premier dataset d’aromates, une méthode de filtrage automatique avait été utilisée :  
le modèle entraîné servait à valider ou rejeter les nouvelles images collectées.

Bien que pratique, cette méthode présente un défaut majeur :  
elle renforce les biais du modèle et réduit la diversité du dataset.  
Les formateurs ont d’ailleurs souligné ce point.

Pour éviter ce problème, j’ai adopté une approche différente :

> **Utiliser un modèle pré‑entraîné uniquement pour extraire des représentations (embeddings), puis appliquer des méthodes non supervisées pour analyser la structure du dataset.**

Cette approche est plus robuste, plus neutre, et mieux alignée avec les bonnes pratiques modernes en data‑centric AI.

---

## 🧩 Étape 1 — Extraction des embeddings (EfficientNet‑B3)

Chaque image est passée dans EfficientNet‑B3 (pré‑entraîné ImageNet).  
On récupère la sortie du dernier bloc convolutionnel, puis on applique un pooling global pour obtenir un vecteur de dimension fixe.

Ce vecteur représente l’image dans un espace latent où des images similaires sont proches les unes des autres.

**Résultat :**

- un tableau `embeddings` de taille `(N, D)`  
- un tableau `labels` de taille `(N,)`

Ces embeddings servent de base à toutes les étapes suivantes.

---

## 🧩 Étape 2 — Clustering par classe (KMeans)

Pour chaque classe (ex : *cosmos*, *fig*, *zinnia*…), j’applique un clustering KMeans sur les embeddings correspondants.

Pourquoi par classe ?

- les images d’une même classe forment naturellement un cluster compact,
- cela permet de repérer les images atypiques (angles étranges, objets parasites, mauvaises plantes),
- cela permet de définir un **seuil automatique** pour filtrer les futures images.

Pour chaque classe, je calcule :

- le centroïde du cluster,
- la distance de chaque image au centroïde,
- un seuil basé sur le percentile 95 des distances.

Ce seuil sert ensuite à détecter les outliers.

---

## 🧩 Étape 3 — Visualisation PCA

Pour valider visuellement la cohérence des clusters, j’applique une réduction de dimension (PCA) sur l’ensemble des embeddings.

Cela permet :

- de vérifier que les classes sont bien séparées,
- d’identifier les classes qui se chevauchent,
- de repérer les images anormales.

Cette visualisation est très utile pour la soutenance, car elle illustre concrètement la structure du dataset.

---

## 🧩 Étape 4 — XGBoost multi‑classe sur embeddings

Les embeddings sont ensuite utilisés comme features pour entraîner un modèle XGBoost multi‑classe.

Pourquoi XGBoost ?

- il gère très bien les données tabulaires,
- il est robuste aux classes déséquilibrées,
- il est rapide à entraîner,
- il permet de tester rapidement la qualité des embeddings,
- il fournit des probabilités de prédiction utiles pour le filtrage.

Ce modèle n’est pas destiné à remplacer le modèle final (EfficientNet fine‑tuned),  
mais il constitue un excellent outil d’analyse et de pré‑filtrage.

---

## 🧩 Étape 5 — Filtrage automatique des nouvelles images

Lorsqu’une nouvelle image est collectée :

1. on calcule son embedding via EfficientNet‑B3,
2. on demande à XGBoost de prédire une classe + une confiance,
3. on calcule la distance de l’embedding au centroïde de cette classe,
4. on compare cette distance au seuil défini lors du clustering.

Une image est acceptée si :

- la confiance XGBoost est suffisante,  
- **et** la distance au centroïde est inférieure au seuil.

Ce mécanisme permet :

- d’accepter automatiquement les images “typiques”,
- de rejeter les images douteuses,
- d’éviter d’introduire du bruit dans le dataset.

---

## 🧩 Avantages de cette approche

### ✔ Pas de biais de confirmation  
Contrairement au self‑training, le modèle final n’est jamais utilisé pour filtrer les données.

### ✔ Approche data‑centric moderne  
Elle met l’accent sur la qualité des données, ce qui est un point clé du RNCP.

### ✔ Compatible avec les compétences du bloc 2  
- **C1** : collecte automatisée  
- **C2** : préparation, nettoyage, outliers  
- **C6** : veille (embeddings, clustering, XGBoost)  
- **C7** : benchmark (LogReg vs XGBoost)  
- **C11** : monitorage (distance au centroïde)  
- **C12** : tests (qualité des données)

### ✔ Très bonne démonstration pour le jury  
Tu montres que tu maîtrises :

- les modèles pré‑entraînés,
- les embeddings,
- le clustering,
- les modèles tabulaires,
- la détection d’outliers,
- l’automatisation de la collecte.

---

## 🧩 Conclusion

Ce module constitue une brique essentielle du pipeline complet.  
Il permet d’améliorer la qualité du dataset avant l’entraînement du modèle final,  
tout en évitant les pièges du self‑training.

Il s’intègre naturellement dans la narration globale du projet :

> **Collecte → Préparation → Analyse → Filtrage → Entraînement → API → Déploiement → Monitorage**

Et il te donne un avantage clair lors de la soutenance, car tu montres une compréhension fine des enjeux liés aux données.

---

Si tu veux, je peux maintenant :

- t’aider à rédiger la **section “Collecte & Préparation”** de ton rapport,  
- t’aider à intégrer cette partie dans ton **GitHub Pages**,  
- t’aider à préparer les **slides** correspondantes pour la soutenance,  
- ou t’aider à connecter ce pipeline à ton **entraînement final EfficientNet‑B3**.

Tu me dis ce que tu veux faire ensuite.
