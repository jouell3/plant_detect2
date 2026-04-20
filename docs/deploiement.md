---
layout: default
title: "Déploiement"
nav_order: 4
has_children: true
has_toc: false
---

# Deploiement du modèle de classification de plantes

### 🎯 *Objectif du module*

Afin de pouvoir utiliser le modèle de classification de plantes que j'ai entrainé dans les étapes précédentes, j'ai déployé sur une API pour permettre à tous et chacun de l'utiliser pour faire la classification d'images de plantes. Cette étape de déploiement est cruciale pour rendre le modèle accessible et utilisable par un large public, et pour permettre à d'autres personnes de bénéficier des résultats de mon travail.

Pour le deploiement de mon modèle, j'ai utilisé la plateforme de cloud de Google (Google Cloud Service, GCS) qui offre une infrastructure robuste et scalable pour héberger des applications et des services. Mon choix s'est arrêté sur GCS principalement parce que c'Est la plateforme que nous avons utilisé durant la formation mais il en existe lusieurs autres, comme Amazon Web Services (AWS) ou Microsoft Azure, qui sont également très populaires. GCS est particulièrement apprécié pour son intégration avec d'autres services de Google, et son tierce gratuit qui permet de tester et de déployer des applications sans frais initiaux. De plus, GCS est reconnu pour sa fiabilité, sa sécurité et sa facilité d'utilisation, ainsi que pour les nombreux services qu'elle propose pour le déploiement de modèles de machine learning.


### 🚀 *Processus de déploiement*

Avant de pouvoir déployer mon modèle, j'ai dû créer une architecture qui permet de faire une image Docker de mon application. J'ai utilisé FastAPI, un micro-framework web en Python, pour créer une API RESTful qui reçoit des images de plantes, les traite et retourne les prédictions du modèle. J'ai ensuite créé un fichier Dockerfile qui définit l'environnement nécessaire pour exécuter mon application, y compris l'installation de Python, des bibliothèques nécessaires et la copie de mon code source. Pour aussi éviter tout soucis avec des connections internets de mauvaises qualités et pour anticiper la fin de mon abbonnement gratuit à la plateforme de Weights & Biases, j'ai inclut les modèles de classification de plantes dans l'image Docker pour qu'il soit disponible localement lors de l'exécution de l'application.

Avant de construire l'image Docker et de la pousser vers le registre de conteneurs, j'ai testé mon application localement pour m'assurer qu'elle fonctionne correctement. J'ai lancé localement mon backend et j'ai vérifié que toutes les routes fonctionnaient comme prévu en utilisant l'interface de FastAPI pour faire les tests des différents endpoints. Ensuite, j'ai lancé mon frontend localement aussi pour vérifier que l'interface utilisateur était fonctionnelle et que les interactions avec le backend se déroulaient sans problème. J'ai également vérifié que les prédictions du modèle étaient correctes en testant avec différentes images de plantes.

Une fois tous ces tests locaux réussis, j'ai pu construire l'image Docker de mon application. J'ai tout d'abord fait des tests locaux de mon image Docker pour m'assurer que tout fonctionnait correctement dans l'environnement conteneurisé avant de la pousser vers le registre de conteneurs de Google. J'ai utilisé la commande `docker build` pour construire l'image Docker et j'ai vérifié que l'application fonctionnait correctement en exécutant un conteneur localement à partir de cette image. Une fois que j'étais satisfait du fonctionnement de l'image Docker, j'ai utilisé la commande `docker push` pour pousser cette image vers le registre de conteneurs de Google (Google Artifact Registry, GAR) afin de pouvoir la déployer sur GCS. 

Finalement, une fois que l'image était sur Artifact Registry, j'ai utilisé Google Cloud Run, un service de compute sans serveur qui permet de déployer des applications conteneurisées, pour déployer mon application sur GCS. J'ai configuré Cloud Run pour utiliser l'image Docker que j'avais poussée vers Artifact Registry, et j'ai défini les paramètres de déploiement tels que la région, les ressources allouées et les autorisations d'accès. Une fois le déploiement terminé, mon application était accessible via une URL publique fournie par Cloud Run, ce qui permet à n'importe qui de faire des requêtes à l'API pour obtenir des prédictions de classification de plantes. L'idée était ici de faire en sorte que le frontend puisse communiquer avec le backend de manière fluide et transparente, en utilisant l'URL publique fournie par Cloud Run pour faire les requêtes à l'API et obtenir les résultats de classification de plantes.

## *Dépoiement de l'application sur Streamlit Cloud*

Une fois le backend deploié sur GCS, j'ai décidé de déployer le frontend de mon application sur Streamlit Cloud, une plateforme de déploiement spécialement conçue pour les applications Streamlit. J'ai créé un compte sur Streamlit Cloud et j'ai connecté mon dépôt GitHub contenant le code de mon frontend. J'ai configuré les paramètres de déploiement pour spécifier la branche à utiliser et les dépendances nécessaires pour exécuter l'application. Une fois le déploiement terminé, mon frontend était accessible via une URL publique fournie par Streamlit Cloud, ce qui permet à n'importe qui d'accéder à l'interface utilisateur de mon application et d'interagir avec le backend déployé sur GCS. Le seul changement a faire dans le code de mon frontend pour le déploiement sur Streamlit Cloud était de mettre à jour l'URL du backend pour qu'elle pointe vers l'URL publique fournie par Cloud Run, afin que le frontend puisse communiquer correctement avec le backend déployé sur GCS.

Plus de détails sur la création de l'interface utilisateur de mon application de classification de plantes dans la section suivante : [Génération d'une interface utilisateur pour l'utilisation des modèles de classification de plantes](./frontend.md).

## 🔍 *Tests et validation*

Il était maintenant le temps de faire quelques tests complets pour s'assurer que tout fonctionnait correctement après le déploiement. J'ai commencé par faire des tests manuels en utilisant l'interface utilisateur de mon application pour vérifier que les différentes fonctionnalités étaient opérationnelles et que les prédictions du modèle étaient correctes. J'ai testé avec différentes images de plantes pour m'assurer que le modèle de classification fonctionnait bien et que les résultats étaient précis.

Ensuite, j'ai lancé des tests plus approfondis en utilisant des outils comme Postman pour simuler des requêtes HTTP et vérifier les réponses de l'API. J'ai également vérifié que le modèle de classification de plantes était correctement chargé et que les prédictions étaient précises.

Enfin, j'ai surveillé les performances de l'application en utilisant les outils de monitoring fournis par GCS pour m'assurer que l'application fonctionnait de manière fluide et sans erreurs. J'ai vérifié les logs pour identifier et résoudre tout problème potentiel, et j'ai fait des ajustements si nécessaire pour améliorer les performances de l'application. Après tous ces tests et validations, j'étais confiant que mon application de classification de plantes était prête à être utilisée par un large public, et que les utilisateurs pourraient bénéficier des résultats de mon travail de manière simple et efficace.

