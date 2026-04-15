import streamlit as st

from i18n import get_language, render_language_selector

st.set_page_config(page_title="Plant Detect", layout="centered")

with st.sidebar:
    render_language_selector()

lang = get_language()

if lang == "fr":
    st.title("🌿 Plant Detect")
    st.markdown("### Identification d'herbes aromatiques par IA")

    st.divider()

    st.markdown("""
Bienvenue sur **Plant Detect**, une application de reconnaissance d'herbes aromatiques, fleurs et arbres fruitiers/baies, alimentee par des modeles d'IA
entraines sur des images réelles.

---

### 📌 Que fait cette application ?

Cette application permet d'identifier une herbe aromatique, fleur ou arbre fruitier/baie a partir d'une simple photo prise avec la camera du telephone
ou télechargée depuis un dossier.
\n Elle donne une prédiction en temps réel avec un score de confiance, et affiche le top 3 des espèces les plus probables a partir de différent modeles.


---

Voici une petite description des differentes pages de l'application :

### 🔍 Prédiction d'herbes aromatiques, fleur ou arbre fruitier/baie

Accédé a cet onglet pour :
- **Télécharger une images** de plantes aromatiques depuis un dossier present sur votre ordinateur
- Alternativement, il est possible de **Prendre une photo** directement depuis la camera du téléphone
\n Ceci permettra d'obtenir une **identification automatique** de l'espece avec un score de confiance.
\n Il est également possible de visualiser le **top 3** des espèces les plus probables predit par le modèle.
\n Si votre image correspond à une espèce reconnue d'herbe aromatique, vous aurez *en bonus* une suggestion de plats ou boissons possible a faire avec cette herbe aromatique identifiee, ainsi que des conseils de culture et d'entretien.
\n Un prompt de generation de recette a partir de l'herbe aromatique identifiée sera également disponible.

""")
    st.caption("""Les differents modèles de reconnaissance d'herbes aromatiques ont été entrainés sur un dataset de plus de 58 000+ images réelles d'herbes aromatiques, fleurs et arbres fruitiers/baies courantes prises dans des conditions variees (lumiere, angles, arriere-plans.
               Les images ont été obtenues via une API du site iNaturalist, qui regroupe des photos de plantes du monde entier avec des metadonnees de localisation et d'espece.  """)

    st.markdown("""
##### Voici la liste complete des especes reconnues par le modele :

**Angelique, Basilic, Bourrache, Camomille, Ciboulette, Coriandre, Aneth, Fenouil, Hysope, Lavande, Citronnelle, Verveine citronnee, Liveche, Menthe, Armoise, Origan, Persil, Romarin, Sauge, Sarriette, Estragon, Thym, Gaultherie**
Fleurs:
**Marguerite, hellébore, iris, gerbera, allium, tournesol, chrysanthème, freesia, lisianthus, renoncule, glycine, digitale, gypsophile, cosmos, pavot, hortensia, zinnia, lys, oiseau de paradis**
**
Arbres fruitiers et baies:
**Mûre, myrtille, cerise, canneberge, figue, raisin, kiwi, citron, melon, pêche, poire, framboise, fraise**
---

### 📊 Prediction par lot (Multiples prédictions)

Accede a cet onglet pour :
- **Telecharger plusieurs images** a la fois
- Obtenir le top 3 des predictions pour chaque image ou faire un demande en bloc pour obtenir la premiere prediction de chaque image pour chaque modele.
- **Visualiser les predictions** obtenues a partir de differents modeles :
    - a RestNet50 (ResNet18)
    - EfficientNet B4/B5 
    - ConvNeXt-Tiny
    - MobileNetV3-Large

### 🏷️ Selection d'Images (Image labelling)

Accede a cet onglet pour :
- **Parcourir toutes les images** d'un dossier contenant des photos.
- **Selectionner** les differentes images (bonne qualite) qui serviront a l'entrainement des differents modeles de reconnaissance d'herbes aromatiques.


Le nom des images selectionnees seront sauvegardes dans un fichier CSV reutilisable pour entrainer ou re-entrainer les modeles.
""")
    st.divider()
    st.caption("Les API qui permettent d'obtenir les predictions sont hebergees sur **Google Cloud Run** et les predictions sont renvoyees en temps reel.")

    st.divider()
    st.caption("""• Modele : ResNet50, EfficientNet B4/B5, ConvNeXt-Tiny et MobileNetV3-Large
               • Deploiement : Google Cloud Run
               • Stockage : Weigth and Bias cloud service
               • Dataset : 58000+ images d'herbes aromatiques, fleurs et arbres fruitiers/baies (iNaturalist)
               • Auteurs : Jimmy OUELLET
               • Code source : [GitHub](https://github.com/jimmyouellet/plant-detect2)
               """)
else:
    st.title("🌿 Plant Detect")
    st.markdown("### AI-Powered Aromatic Herb Identification")

    st.divider()

    st.markdown("""
Welcome to **Plant Detect**, an aromatic herb recognition application
powered by deep learning models (ResNet18, EfficientNet B3, and TensorFlow) trained on real-world images.

---

### 📌 What does this app do?

This app identifies an aromatic herb from a simple photo captured with a phone camera
or uploaded from a local folder.
\n It returns a real-time prediction with a confidence score, and displays the top 3 most likely species across multiple models.

---

Here is a quick overview of the app pages:

### 🔍 Aromatic Herb Prediction

Use this page to:
- **Upload a plant image** from a folder on your computer
- Or **take a photo** directly with your phone camera
\n This gives you an **automatic species identification** with a confidence score.
\n You can also view the **top 3** most likely species predicted by the model.
\n You also get recipe and drink suggestions for the detected herb, plus cultivation and care guidance.
\n A recipe prompt based on the detected herb is available as well.

""")
    st.caption("""The herb recognition models were trained on a dataset of 24,000+ real images of common aromatic herbs captured in varied conditions (lighting, angles, backgrounds).
               Images were collected through the iNaturalist API, which aggregates plant photos from around the world with species and location metadata.""")

    st.markdown("""
##### Full list of species recognized by the model:

Herb list:
**Angelica, Basil, Borage, Chamomile, Chives, Coriander, Dill, Fennel, Hyssop, Lavender, Lemongrass, Lemon Verbena, Lovage, Mint, Mugwort, Oregano, Parsley, Rosemary, Sage, Savory, Tarragon, Thyme, Wintergreen**

Flowers:
**Daisy, Hellebore, Iris, Gerbera, Allium, Sunflower, Chrysanthemum, Freesia, Lisianthus, Ranunculus, Wisteria, Foxglove, Gypsophila, Cosmos, Poppy, Hydrangea, Zinnia, Lily, Bird of Paradise**

**
Trees and berries:
**Blackberry, Blueberry, Cherry, Cranberry, Fig, Grape, Kiwi, Lemon, Melon, Peach, Pear, Raspberry, Strawberry**

---

### 📊 Batch Prediction (Multiple Predictions)

Use this page to:
- **Upload multiple images** at once
- Get top-3 predictions for each image, or submit a bulk request to get top-1 predictions per model.
- **Compare predictions** from multiple models:
    - a RestNet50 (ResNet18)
    - EfficientNet B4/B5 
    - ConvNeXt-Tiny
    - MobileNetV3-Large

### 🏷️ Image Selection (Image Labeling)

Use this page to:
- **Browse all images** from a folder.
- **Select** high-quality images that will be used for model training.

Selected image names can be exported to a reusable CSV file for future training or retraining.
""")
    st.divider()
    st.caption("Prediction APIs are hosted on **Google Cloud Run** and return results in real time.")

    st.divider()
    st.caption("""• Model: ResNet50, EfficientNet B4/B5, ConvNeXt-Tiny and MobileNetV3-Large  
               • Deployment: Google Cloud Run  
               • Storage: Weight and Bias cloud service  
               • Dataset: 58,000+ real aromatic herb, flowers and trees/berries images (iNaturalist)  
               • Authors: Jimmy OUELLET  
               • Source code: [GitHub](https://github.com/jimmyouellet/plant-detect2)  
               """)
