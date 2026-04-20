import streamlit as st

from i18n import get_language, render_language_selector
from styles import inject_global_css

st.set_page_config(page_title="Plant Detect", layout="centered")
inject_global_css()

with st.sidebar:
    render_language_selector()

lang = get_language()

_FEATURE_CARDS = {
    "fr": [
        ("🔍", "Prédiction", "Identifiez une plante à partir d'une photo ou d'une image importée. Obtenez le top 3 des espèces probables avec score de confiance, fiche botanique et suggestions de recettes."),
        ("📊", "Prédiction par lot", "Importez jusqu'à 20 images à la fois et comparez les prédictions de 5 modèles en parallèle. Export CSV inclus."),
        ("🏷️", "Étiquetage d'images", "Parcourez et sélectionnez manuellement des images pour constituer votre jeu de données d'entraînement. Export CSV des sélections."),
        ("📡", "Monitoring", "Suivez en temps réel la santé de l'API : latence, confiance, distribution des classes, taux d'accord entre modèles."),
    ],
    "en": [
        ("🔍", "Prediction", "Identify a plant from a photo or uploaded image. Get top-3 species with confidence scores, a botanical profile, and recipe suggestions."),
        ("📊", "Batch Prediction", "Upload up to 20 images at once and compare predictions from 5 models in parallel. CSV export included."),
        ("🏷️", "Image Labelling", "Browse and manually select images to build your training dataset. Export selections as CSV."),
        ("📡", "Monitoring", "Track API health in real time: latency, confidence, class distribution, and inter-model agreement rate."),
    ],
}

_SPECIES = {
    "fr": {
        "🌿 Herbes aromatiques (23)": "Angélique, Basilic, Bourrache, Camomille, Ciboulette, Coriandre, Aneth, Fenouil, Hysope, Lavande, Citronnelle, Verveine citronnée, Livèche, Menthe, Armoise, Origan, Persil, Romarin, Sauge, Sarriette, Estragon, Thym, Gaulthérie",
        "🌸 Fleurs (19)": "Marguerite, Hellébore, Iris, Gerbera, Allium, Tournesol, Chrysanthème, Freesia, Lisianthus, Renoncule, Glycine, Digitale, Gypsophile, Cosmos, Pavot, Hortensia, Zinnia, Lys, Oiseau de paradis",
        "🍒 Arbres & baies (16)": "Mûre, Myrtille, Cerise, Canneberge, Figue, Raisin, Kiwi, Citron, Melon, Pêche, Poire, Framboise, Fraise, Pomme, Prune, Abricot",
    },
    "en": {
        "🌿 Aromatic herbs (23)": "Angelica, Basil, Borage, Chamomile, Chives, Coriander, Dill, Fennel, Hyssop, Lavender, Lemongrass, Lemon Verbena, Lovage, Mint, Mugwort, Oregano, Parsley, Rosemary, Sage, Savory, Tarragon, Thyme, Wintergreen",
        "🌸 Flowers (19)": "Daisy, Hellebore, Iris, Gerbera, Allium, Sunflower, Chrysanthemum, Freesia, Lisianthus, Ranunculus, Wisteria, Foxglove, Gypsophila, Cosmos, Poppy, Hydrangea, Zinnia, Lily, Bird of Paradise",
        "🍒 Trees & berries (16)": "Blackberry, Blueberry, Cherry, Cranberry, Fig, Grape, Kiwi, Lemon, Melon, Peach, Pear, Raspberry, Strawberry, Apple, Plum, Apricot",
    },
}

is_fr = lang == "fr"

st.title("🌿 Plant Detect")
st.markdown(
    "### Identification de plantes par intelligence artificielle" if is_fr
    else "### AI-powered plant identification"
)
st.markdown(
    "Chargez une photo — obtenez l'espèce, le score de confiance, la fiche botanique et des suggestions de recettes."
    if is_fr
    else "Upload a photo — get the species, confidence score, botanical profile, and recipe suggestions."
)

st.divider()

# ── Feature cards ──────────────────────────────────────────────────────────
cards = _FEATURE_CARDS["fr" if is_fr else "en"]
card_html = "".join(
    f"<div style='flex:1;background:white;border:1px solid #dde5dd;border-radius:14px;"
    f"padding:1.2rem 1rem;box-shadow:0 2px 8px rgba(26,46,35,0.06)'>"
    f"<div style='font-size:1.6rem;margin-bottom:8px'>{icon}</div>"
    f"<div style='font-family:Playfair Display,Georgia,serif;font-weight:700;"
    f"color:#1a2e23;font-size:1rem;margin-bottom:6px'>{title}</div>"
    f"<div style='font-size:0.82rem;color:#5a7a62;line-height:1.55'>{desc}</div>"
    f"</div>"
    for icon, title, desc in cards
)
st.markdown(
    f"<div style='display:flex;gap:1rem;align-items:stretch'>{card_html}</div>",
    unsafe_allow_html=True,
)

st.divider()

# ── Species list ───────────────────────────────────────────────────────────
st.markdown("##### " + ("Espèces reconnues (58 catégories)" if is_fr else "Recognised species (58 categories)"))
for group, names in _SPECIES["fr" if is_fr else "en"].items():
    with st.expander(group, expanded=False):
        st.markdown(
            f"<p style='font-size:0.95rem;color:#1a2e23;line-height:1.7;margin:0'>{names}</p>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Tech stack footer ──────────────────────────────────────────────────────
stack = (
    "Modèles · ResNet-50 · EfficientNet B4/B5 · ConvNeXt-Tiny · MobileNetV3-Large"
    if is_fr else
    "Models · ResNet-50 · EfficientNet B4/B5 · ConvNeXt-Tiny · MobileNetV3-Large"
)
st.caption(
    f"{stack}  \n"
    + ("Déploiement · Google Cloud Run · Artifacts · Weights & Biases · Dataset · 58 000+ images iNaturalist  \n"
       if is_fr else
       "Deployment · Google Cloud Run · Artifacts · Weights & Biases · Dataset · 58,000+ iNaturalist images  \n")
    + ("Auteur · Jimmy OUELLET · [Code source](https://github.com/jimmyouellet/plant-detect2)"
       if is_fr else
       "Author · Jimmy OUELLET · [Source code](https://github.com/jimmyouellet/plant-detect2)")
)
