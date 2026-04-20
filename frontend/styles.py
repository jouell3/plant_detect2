"""Centralized styling and UX components for Plant Detect app."""

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Color tokens — semantic confidence visualization
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "success": "#2e7d32",        # ≥90% confidence, "good" labels
    "warning": "#f57c00",        # 50-90% confidence, caution states
    "error": "#c62828",          # <50% confidence, critical states
    "text_primary": "#212121",   # Main text
    "text_muted": "#616161",     # Secondary text (timestamps, captions)
    "text_disabled": "#bdbdbd",  # Disabled button text
    "border": "#e0e0e0",         # Card/divider borders
    "background_light": "#f5f5f5",  # Light backgrounds
}

# ─────────────────────────────────────────────────────────────────────────────
# Typography — standardized Streamlit + inline HTML styles
# ─────────────────────────────────────────────────────────────────────────────
TYPOGRAPHY = {
    "caption_muted": f"font-size: 0.75rem; color: {COLORS['text_muted']}; margin-top: 0.25rem;",
    "label_small": f"font-size: 0.875rem; color: {COLORS['text_primary']}; font-weight: 500;",
    "hint_small": f"font-size: 0.8rem; color: {COLORS['text_muted']};",
}


def confidence_color(confidence: float) -> str:
    """Return color hex for a confidence value (0-1)."""
    if confidence >= 0.90:
        return COLORS["success"]
    elif confidence >= 0.50:
        return COLORS["warning"]
    else:
        return COLORS["error"]


def confidence_badge(species: str, confidence: float, model_name: str | None = None) -> None:
    """Render a styled confidence card with semantic coloring.
    
    Args:
        species: Plant species name to display
        confidence: Confidence value (0-1)
        model_name: Optional model identifier (e.g., "pytorch", "sklearn")
    """
    color = confidence_color(confidence)
    muted_color = COLORS["text_muted"]
    subtitle_html = f"<div style='font-size: 0.75rem; color: {muted_color}; margin-top: 4px;'>{model_name}</div>" if model_name else ""
    
    html = f"""
    <div style='
        background: {color}11; 
        border-left: 4px solid {color};
        padding: 12px; 
        border-radius: 4px; 
        margin: 8px 0;
    '>
        <div style='font-weight: 700; color: {COLORS["text_primary"]};'>
            {species}
        </div>
        <div style='font-size: 1.1rem; color: {color}; font-weight: 600;'>
            {confidence:.0%}
        </div>
        {subtitle_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def styled_info_card(title: str, content_dict: dict[str, str], subtitle: str | None = None) -> None:
    """Render a visually distinct info/prediction card.
    
    Args:
        title: Card title
        content_dict: Dictionary of key-value pairs to display (key: label, value: content)
        subtitle: Optional subtitle or metadata
    """
    content_html = "".join(
        f"<div style='margin-bottom: 8px;'><strong>{k}:</strong> {v}</div>"
        for k, v in content_dict.items()
    )
    muted_color = COLORS["text_muted"]
    subtitle_html = f"<div style='font-size: 0.85rem; color: {muted_color}; margin-bottom: 8px;'>{subtitle}</div>" if subtitle else ""
    
    # Keep all style attributes on single lines for proper Streamlit HTML rendering
    html = f"<div style='background: {COLORS['background_light']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 16px; margin: 12px 0;'><div style='font-weight: 700; font-size: 1.85rem; margin-bottom: 8px;'>{title}</div>{subtitle_html}<div style='font-size: 1.2rem; line-height: 1.6; color: {COLORS['text_primary']};'>{content_html}</div></div>"
    st.markdown(html, unsafe_allow_html=True)


def page_header(title: str, description: str = "", icon: str = "") -> None:
    """Render a consistent page header with title and optional description."""
    full_title = f"{icon} {title}" if icon else title
    st.title(full_title)
    if description:
        st.markdown(description)


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — botanical theme
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ── Foundations ── */
html, body { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
[data-testid="stAppViewContainer"] > .main { background-color: #f8f6f1; }
[data-testid="stHeader"] { background: linear-gradient(90deg, #1a2e23 0%, #2d5a3d 100%); }
.block-container { padding-top: 1.5rem !important; }

/* ── Headings ── */
h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #1a2e23 !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}
h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #2d5a3d !important;
}
h4, h5, h6 {
    font-family: 'DM Sans', sans-serif !important;
    color: #1a2e23 !important;
    font-weight: 600 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #1a2e23 0%, #243b2e 100%) !important;
}
[data-testid="stSidebar"] * { color: #b8d0bb !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e0ede0 !important;
    font-family: 'Playfair Display', Georgia, serif !important;
}
[data-testid="stSidebar"] .stCaption * { color: #6a9472 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #2d4a38 !important;
    border-color: #3d5a48 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #c8dcc9 !important; }
[data-testid="stSidebar"] hr { border-top-color: #2d4a38 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #b8d0bb !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #2d4a38 !important;
    border: 1px solid #3d5a48 !important;
    color: #c8dcc9 !important;
}

/* ── Primary / secondary buttons ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    letter-spacing: 0.2px !important;
    transition: all 0.18s ease !important;
    border: 1px solid #c8dcc9 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2e7d32 0%, #388e3c 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(46,125,50,0.28) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(46,125,50,0.40) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #2e7d32 !important;
    color: #2e7d32 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    border-bottom: 2px solid #dde5dd !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #5a7a62 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 18px !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #1a2e23 !important;
    background: #edf3ee !important;
    border-bottom: 2px solid #2e7d32 !important;
    font-weight: 600 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid #dde5dd !important;
    border-radius: 12px !important;
    padding: 1.1rem 1rem !important;
    box-shadow: 0 2px 8px rgba(26,46,35,0.07) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #1a2e23 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #6a9472 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #dde5dd !important;
    border-radius: 10px !important;
    background: white !important;
    box-shadow: 0 1px 4px rgba(26,46,35,0.04) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #2d5a3d !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #b0ccb4 !important;
    border-radius: 12px !important;
    background: #f0f7f1 !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2e7d32 !important;
    background: #e8f5e9 !important;
}

/* ── Progress bars ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2e7d32, #66bb6a) !important;
    border-radius: 999px !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid #dde5dd !important;
    margin: 1.25rem 0 !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    border: 1px solid #2e7d32 !important;
    color: #2e7d32 !important;
    background: white !important;
}
[data-testid="stDownloadButton"] > button:hover { background: #e8f5e9 !important; }

/* ── Captions ── */
[data-testid="stCaptionContainer"] p, .stCaption p {
    color: #7a9e87 !important;
    font-size: 0.82rem !important;
}

/* ── Select boxes ── */
[data-baseweb="select"] > div { border-radius: 8px !important; }

/* ── Code blocks ── */
[data-testid="stCode"] { border-radius: 8px !important; }

/* ── Image captions ── */
[data-testid="stImageCaption"] { color: #7a9e87 !important; font-size: 0.8rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #f0ede8; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #b0ccb4; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2e7d32; }
</style>
"""


def inject_global_css() -> None:
    """Inject project-wide CSS overrides for typography, colour, and component polish."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
