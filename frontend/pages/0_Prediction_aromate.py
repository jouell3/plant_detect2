import io
import json
import os
import random
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from loguru import logger
from PIL import Image

from i18n import get_language, render_language_selector
from styles import COLORS, confidence_color, confidence_badge, styled_info_card, page_header, inject_global_css
from utils import post_with_retries, validate_image_file

# Local API URL - change to your deployed API endpoint if needed
#API_URL = "http://localhost:8080"

# Deployed API URL (replace with your actual endpoint)
API_URL = "https://plant-predictor-966041648100.europe-west1.run.app"

MAX_HISTORY_ITEMS = 20
RETRY_DELAYS_SECONDS = (0.8, 1.6)


_FICHES_PATH = Path(__file__).parent.parent / "fiches.json"
FICHES: dict = json.loads(_FICHES_PATH.read_text(encoding="utf-8")) if _FICHES_PATH.exists() else {}
_SUGGESTIONS_PATH = Path(__file__).parent.parent / "suggestions.json"
SUGGESTIONS: dict = json.loads(_SUGGESTIONS_PATH.read_text(encoding="utf-8")) if _SUGGESTIONS_PATH.exists() else {}

st.set_page_config(page_title="Plant Predictor", layout="wide")
inject_global_css()

# ---------------------------------------------------------------------------
# Session state — prediction history
# ---------------------------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []  # list of {name, species, confidence, thumb_bytes, timestamp}
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "suggestions_pool" not in st.session_state:
    st.session_state.suggestions_pool = []
if "suggestions_visible_count" not in st.session_state:
    st.session_state.suggestions_visible_count = 0
if "suggestions_species_key" not in st.session_state:
    st.session_state.suggestions_species_key = ""
if "last_uploaded_id" not in st.session_state:
    st.session_state.last_uploaded_id = None


def _normalize_species_key(value: str) -> str:
    return (value or "").strip().lower().replace("-", " ")


def _display_species_name(species: str) -> str:
    fiche = FICHES.get(_normalize_species_key(species), {})
    if get_language() == "en":
        return fiche.get("nom_en", fiche.get("nom_fr", species))
    return fiche.get("nom_fr", species)


def _fiche_value(fiche: dict, key: str, language: str):
    if language == "en":
        return fiche.get(f"{key}_en", fiche.get(key))
    return fiche.get(key)


def _suggestion_value(suggestion: dict, key: str, language: str) -> str:
    if language == "en":
        return suggestion.get(f"{key}_en", suggestion.get(key, ""))
    return suggestion.get(key, "")


def _generate_recipe_prompt(dish_name: str, herb_name: str) -> str:
    """Generate a detailed recipe prompt for an AI chat."""
    if get_language() == "en":
        return (
            f"Give me a complete and detailed recipe for {dish_name} using fresh {herb_name.lower()}. "
            "Include ingredients, step-by-step instructions, and cooking tips."
        )
    return (
        f"Donne-moi une recette complete et detaillee pour {dish_name} en utilisant du {herb_name.lower()} frais. "
        "Inclus les ingredients, les instructions etape par etape, et les conseils de cuisson."
    )


def _get_suggestions_for_species(species_key: str) -> list[dict]:
    entry = SUGGESTIONS.get(species_key, [])
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        suggestions = entry.get("suggestions", [])
        return suggestions if isinstance(suggestions, list) else []
    return []

# ---------------------------------------------------------------------------
# Sidebar — prediction history
# ---------------------------------------------------------------------------
with st.sidebar:
    render_language_selector()
    lang = get_language()
    st.markdown("### History" if lang == "en" else "### Historique")
    if not st.session_state.prediction_history:
        st.caption("No predictions yet." if lang == "en" else "Aucune prediction pour le moment.")
    else:
        show_history = st.checkbox("Show history" if lang == "en" else "Afficher l'historique", value=True)
        if st.button("Clear" if lang == "en" else "Effacer", width='stretch'):
            st.session_state.prediction_history = []
            st.rerun()
        if show_history:
            st.caption(
                f"{len(st.session_state.prediction_history)} / {MAX_HISTORY_ITEMS} items"
                if lang == "en"
                else f"{len(st.session_state.prediction_history)} / {MAX_HISTORY_ITEMS} elements"
            )
            for entry in reversed(st.session_state.prediction_history):
                conf = entry["confidence"]
                color = confidence_color(conf)
                st.image(entry["thumb_bytes"], width=60)
                st.markdown(
                    f"**{entry['species']}**  \n"
                    f"<span style='color:{color}'>{conf:.0%}</span> · "
                    f"<span style='font-size:0.75rem; color:#616161'>{entry['timestamp']}</span>",
                    unsafe_allow_html=True,
                )
                st.divider()

lang = get_language()
st.title("🌿 Aromatic Herb Predictor" if lang == "en" else "🌿 Predicteur d'aromate")
st.markdown(
    "Choose an image from your files or take a photo directly with your camera. "
    "The aromatic herb recognition model will return a real-time prediction with a confidence score. "
    "Feel free to test several images."
    if lang == "en"
    else "Vous pouvez soit choisir une image dans vos dossiers, soit prendre une photo directement avec votre camera. "
    "Le modele de reconnaissance d'herbes aromatiques vous donnera une prediction en temps reel avec un score de confiance. "
    "N'hesitez pas a tester plusieurs images pour voir les resultats !"
)
st.markdown(
    "To predict multiple images at once, go to the 'Batch Predict' page."
    if lang == "en"
    else "Pour predire plusieurs images a la fois, rendez-vous dans l'onglet 'Batch Predict'."
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
tab_upload, tab_camera = st.tabs([
    "📁 Upload image" if lang == "en" else "📁 Upload image",
    "📷 Camera" if lang == "en" else "📷 Camera",
])

uploaded_file = None

with tab_upload:
    f = st.file_uploader(
        "Choose an image (jpg/jpeg/png)" if lang == "en" else "Choisissez une image (jpg/jpeg/png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if f:
        uploaded_file = f

with tab_camera:
    photo = st.camera_input("Take a photo" if lang == "en" else "Prenez une photo", label_visibility="collapsed")
    if photo:
        uploaded_file = photo

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if not uploaded_file:
    st.stop()

# Clear previous prediction when a new file is selected
_file_id = (uploaded_file.name, uploaded_file.size)
if _file_id != st.session_state.last_uploaded_id:
    st.session_state.last_prediction = None
    st.session_state.last_uploaded_id = _file_id

# Validate uploaded image before proceeding
is_valid, error_msg = validate_image_file(uploaded_file)
if not is_valid:
    st.error(f"❌ {error_msg}")
    st.stop()

st.divider()

st.markdown(
    "##### When you are ready, click the button below to launch prediction. "
    "Depending on server load and image complexity, this can take 30 to 60 seconds."
    if lang == "en"
    else "##### Quand vous serez pret, vous pouvez cliquer sur le bouton ci-dessous pour lancer la prediction. "
    "En fonction de la charge du serveur et de la complexite de l'image, cela peut prendre entre 30 secondes et 1 minute. "
    "Merci pour votre patience !"
)


if st.button("🔍 Identify", type="primary", width='content'):
    with st.spinner("Analyzing (~30-60s)..." if lang == "en" else "Analyse en cours (~30-60s)..."):
        try:
            logger.info("predict_herb | file={}", uploaded_file.name)
            file_bytes = uploaded_file.getvalue()
            response = post_with_retries(
                url=f"{API_URL}/predict",
                files={"file": (uploaded_file.name, file_bytes, uploaded_file.type or "image/jpeg")},
                timeout=60,
                retry_delays_seconds=RETRY_DELAYS_SECONDS,
                log_message="predict_herb failed",
            )
        except requests.exceptions.ConnectionError:
            st.error("Unable to reach the API. Please verify the service is online." if lang == "en" else "Impossible de joindre l'API. Verifiez que le service est en ligne.")
            logger.error("API connection error | url={}", API_URL)
            st.stop()
        except requests.exceptions.Timeout:
            st.error("The service is taking too long to respond. Please retry in a few seconds." if lang == "en" else "Le service met trop de temps a repondre. Reessayez dans quelques secondes.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error((f"API error: {e}") if lang == "en" else (f"Erreur API: {e}"))
            logger.error("API HTTP error | {}", e)
            st.stop()

    data = response.json() 
    models_used = list({item["model"] for item in data["predictions"]})
    first_model = models_used[0]
    top_species = []
    mean_confidence = []
    good_models = []
    for i, key in enumerate(models_used):
        top_species.append(data["predictions"][i]["top3"][0]["class"])
    top_species = max(set(top_species), key=top_species.count)  # Most common top species among models
    other_species = [s for s in top_species if s != top_species]
    for i, key in enumerate(models_used):
        if data["predictions"][i]["top3"][0]["class"] == top_species:
            mean_confidence.append(data["predictions"][i]["top3"][0]["confidence"])
            good_models.append(key)
    mean_confidence = sum(mean_confidence) / len(mean_confidence) if mean_confidence else 0.0

    st.session_state.last_prediction = {
        "data": data,
        "models_used": models_used,
        "top_species": top_species,
        "mean_confidence": mean_confidence,
        "good_models": good_models,
        "uploaded_name": uploaded_file.name,
        "uploaded_bytes": file_bytes,
    }

    species_key = _normalize_species_key(top_species)
    all_suggestions = _get_suggestions_for_species(species_key)
    shuffled_suggestions = list(all_suggestions)
    random.shuffle(shuffled_suggestions)
    st.session_state.suggestions_pool = shuffled_suggestions
    st.session_state.suggestions_visible_count = min(6, len(shuffled_suggestions), 12)
    st.session_state.suggestions_species_key = species_key

    # ── Record to history ─────────────────────────────────────────────────
    st.session_state.prediction_history.append({
        "name": uploaded_file.name,
        "species": top_species,
        "confidence": mean_confidence,
        "thumb_bytes": uploaded_file.getvalue(),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    if len(st.session_state.prediction_history) > MAX_HISTORY_ITEMS:
        st.session_state.prediction_history = st.session_state.prediction_history[-MAX_HISTORY_ITEMS:]

prediction = st.session_state.last_prediction
if prediction:
    data = prediction["data"]
    models_used = prediction["models_used"]
    top_species = prediction["top_species"]
    mean_confidence = prediction["mean_confidence"]
    good_models = prediction["good_models"]
    herb_found = [data["predictions"][i]["top3"][0]["class"] for i in range(len(models_used))]
    unique_preds = len(set(herb_found))
    fiche = FICHES.get(_normalize_species_key(top_species))

    # ── Hero result card ──────────────────────────────────────────────────
    color = confidence_color(mean_confidence)
    if unique_preds == 1:
        agreement_badge = f"<span style='background:#e8f5e9;color:#2e7d32;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600'>✅ {'All models agree' if lang == 'en' else 'Tous les modèles concordent'}</span>"
    elif unique_preds == 2:
        agreement_badge = f"<span style='background:#fff3e0;color:#e65100;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600'>⚠️ {'Most models agree' if lang == 'en' else 'La majorité concorde'}</span>"
    else:
        agreement_badge = f"<span style='background:#fce4ec;color:#c62828;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600'>❌ {'Models disagree' if lang == 'en' else 'Les modèles divergent'}</span>"

    display_name = _display_species_name(top_species)
    hero_html = f"""
    <div style='background:white;border:1px solid #dde5dd;border-radius:16px;padding:1.4rem 1.8rem;
                margin:1rem 0 1.5rem;box-shadow:0 2px 12px rgba(26,46,35,0.08);
                display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap'>
        <div style='flex:1;min-width:200px'>
            <div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:1.4px;
                        color:#7a9e87;font-weight:600;margin-bottom:6px'>
                {"Espèce identifiée" if lang == "fr" else "Identified species"}
            </div>
            <div style='font-size:2rem;font-weight:700;color:#1a2e23;
                        font-family:Playfair Display,Georgia,serif;line-height:1.2;margin-bottom:8px'>
                {display_name}
            </div>
            {agreement_badge}
        </div>
        <div style='text-align:center;padding:0.8rem 1.4rem;background:#f8f6f1;
                    border-radius:12px;border:1px solid #dde5dd'>
            <div style='font-size:2.4rem;font-weight:700;color:{color};line-height:1'>{mean_confidence:.0%}</div>
            <div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.8px;
                        color:#7a9e87;margin-top:4px'>{"confiance" if lang == "fr" else "confidence"}</div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    if unique_preds > 2:
        st.warning(
            f"The {len(models_used)} models disagree on the prediction. Try another image or improve lighting and angle."
            if lang == "en"
            else f"Les {len(models_used)} modèles ne concordent pas. Essayez avec une autre image ou de meilleures conditions d'éclairage."
        )
    elif mean_confidence < 0.50:
        st.info(
            "Low confidence — try a sharper photo with better lighting and a tighter crop on the plant."
            if lang == "en"
            else "Confiance faible — essayez une photo plus nette avec une meilleure lumière et un cadrage serré sur la plante."
        )

    # ── Image + botanical info ─────────────────────────────────────────────
    col_img, col_info = st.columns([1, 3], gap="large")

    with col_img:
        img = Image.open(io.BytesIO(prediction["uploaded_bytes"]))
        st.image(img, width='stretch', caption=prediction["uploaded_name"])

    with col_info:
        if unique_preds <= 2 and fiche:
            nom_fr_md = f"[{fiche['nom_fr']}]({fiche['wikipedia_fr']})" if fiche.get("wikipedia_fr") else fiche["nom_fr"]
            nom_en_md = f"[{fiche['nom_en']}]({fiche['wikipedia_en']})" if fiche.get("wikipedia_en") else fiche["nom_en"]
            herb_display_name = fiche["nom_en"] if lang == "en" else fiche["nom_fr"]

            tab_about, tab_recipes = st.tabs([
                "🌿 About" if lang == "en" else "🌿 À propos",
                "🍽️ Recipes & Dishes" if lang == "en" else "🍽️ Recettes & Plats",
            ])

            with tab_about:
                st.markdown(
                    f"#### {nom_en_md} (*{nom_fr_md}*)" if lang == "en"
                    else f"#### {nom_fr_md} (*{nom_en_md}*)"
                )
                st.markdown(_fiche_value(fiche, "description", lang))
                info_dict = {
                    ("🌸 Aroma" if lang == "en" else "🌸 Arôme"): _fiche_value(fiche, "arome", lang),
                    ("🌱 Cultivation" if lang == "en" else "🌱 Culture"): _fiche_value(fiche, "culture", lang),
                    ("⚠️ Toxicity" if lang == "en" else "⚠️ Toxicité"): _fiche_value(fiche, "toxicite", lang),
                    ("🍽️ Uses" if lang == "en" else "🍽️ Usages"): ", ".join(_fiche_value(fiche, "usages", lang) or []),
                    ("🤝 Pairings" if lang == "en" else "🤝 Compatible"): ", ".join(_fiche_value(fiche, "compatibilites", lang) or []),
                }
                styled_info_card("Properties" if lang == "en" else "Propriétés", info_dict)

            with tab_recipes:
                species_key = _normalize_species_key(top_species)
                if st.session_state.suggestions_species_key == species_key and st.session_state.suggestions_pool:
                    max_display = min(12, len(st.session_state.suggestions_pool))
                    visible_count = min(st.session_state.suggestions_visible_count, max_display)
                    suggestions = st.session_state.suggestions_pool[:visible_count]
                    st.markdown(
                        f"Dish ideas with fresh **{herb_display_name}**. Click an entry for an AI recipe prompt."
                        if lang == "en"
                        else f"Idées de plats avec du **{herb_display_name}** frais. Cliquez pour obtenir un prompt de recette IA."
                    )
                    for idx in range(0, len(suggestions), 3):
                        recipe_cols = st.columns(3)
                        for col_idx, suggestion in enumerate(suggestions[idx:idx + 3]):
                            with recipe_cols[col_idx]:
                                dish_name = _suggestion_value(suggestion, "plat", lang)
                                dish_desc = _suggestion_value(suggestion, "description", lang)
                                st.markdown(
                                    f"<div style='background:#f8f6f1;border-left:3px solid #2e7d32;"
                                    f"padding:12px 14px;border-radius:6px;margin-bottom:4px'>"
                                    f"<div style='font-weight:700;color:#1a2e23;font-size:0.9rem;"
                                    f"margin-bottom:4px'>{dish_name}</div>"
                                    f"<div style='font-size:0.78rem;color:#5a7a62;line-height:1.5'>{dish_desc}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                                prompt = _generate_recipe_prompt(dish_name, herb_display_name)
                                with st.expander(
                                    f"📋 Recipe prompt" if lang == "en" else "📋 Prompt recette",
                                    expanded=False,
                                ):
                                    st.code(prompt, language="text")
                                    st.caption(
                                        "Paste into ChatGPT, Claude, etc." if lang == "en"
                                        else "Collez dans ChatGPT, Claude, etc."
                                    )
                    if st.button(
                        "Show 3 more" if lang == "en" else "3 de plus",
                        disabled=visible_count >= max_display,
                    ):
                        st.session_state.suggestions_visible_count = min(visible_count + 3, max_display)
                        st.rerun()
                else:
                    st.info(
                        "No recipe suggestions available for this species."
                        if lang == "en"
                        else "Aucune suggestion de recette disponible pour cette espèce."
                    )

        elif unique_preds <= 2:
            st.markdown(f"### {top_species}")
            st.info(
                "No botanical profile available for this species."
                if lang == "en"
                else "Aucune fiche botanique disponible pour cette espèce."
            )
            wiki = f"https://fr.wikipedia.org/wiki/Special:Search?search={top_species.replace(' ', '+')}"
            st.markdown(f"[Search Wikipedia]({wiki})" if lang == "en" else f"[Chercher sur Wikipédia]({wiki})")

    # ── Per-model details ─────────────────────────────────────────────────
    st.divider()
    with st.expander(
        "🔬 Prediction details (all models)" if lang == "en" else "🔬 Détails par modèle",
        expanded=False,
    ):
        models_list = list(models_used)
        for row_start in range(0, len(models_list), 2):
            grid_cols = st.columns(2)
            for j, key in enumerate(models_list[row_start:row_start + 2]):
                idx = row_start + j
                with grid_cols[j]:
                    pred_item = data["predictions"][idx]
                    species = pred_item["top3"][0]["class"]
                    confidence = pred_item["top3"][0]["confidence"]
                    color = confidence_color(confidence)
                    st.markdown(
                        f"<div style='background:white;border:1px solid #dde5dd;border-radius:10px;"
                        f"padding:14px 16px;margin-bottom:8px'>"
                        f"<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;"
                        f"color:#7a9e87;font-weight:600;margin-bottom:4px'>{key.upper()}</div>"
                        f"<div style='font-size:1.25rem;font-weight:700;color:#1a2e23;margin-bottom:2px'>"
                        f"{_display_species_name(species)}</div>"
                        f"<div style='color:{color};font-size:1rem;font-weight:600'>{confidence:.0%}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    for rank, pred in enumerate(pred_item["top3"], 1):
                        bar_pct = int(pred["confidence"] * 100)
                        st.markdown(f"**{rank}.** {_display_species_name(pred['class'])} — {pred['confidence']:.0%}")
                        st.progress(bar_pct)