"""
Big Cats — Streamlit Wildlife Species Identifier
=================================================

A pixel-faithful port of the React "Big Cats" UI to Streamlit.

Run locally:
    pip install streamlit pillow
    streamlit run app.py

Integrate your model:
    Replace `predict(image)` with a call to your classifier.
    It must return a list of dicts matching the Prediction shape below.
"""

from __future__ import annotations

import base64
import json
import io
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Big Cats — Identify wildlife species instantly",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme state (dark mode toggle)
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "results" not in st.session_state:
    st.session_state.results = None
if "preview_bytes" not in st.session_state:
    st.session_state.preview_bytes = None

# ---------------------------------------------------------------------------
# Design tokens — mirror the React app's src/styles.css
# ---------------------------------------------------------------------------
LIGHT_TOKENS = {
    "background": "#f5f3ee",
    "foreground": "#1a2e22",
    "card": "#f9f7f3",
    "muted": "#e8e4db",
    "muted_foreground": "#5e6e5a",
    "border": "#d4cfbe",
    "primary": "#1a4a32",
    "primary_foreground": "#f5f3ee",
    "secondary": "#e4e0d4",
    "accent": "#c89b3c",
    "fern": "#2d8a5e",
    "clay": "#b07840",
}
DARK_TOKENS = {
    "background": "#1a2820",
    "foreground": "#eceae5",
    "card": "#22332a",
    "muted": "#2e3e34",
    "muted_foreground": "#98a898",
    "border": "rgba(255,255,255,0.12)",
    "primary": "#5cb88a",
    "primary_foreground": "#1a2820",
    "secondary": "#2a3a30",
    "accent": "#c89b3c",
    "fern": "#5cb88a",
    "clay": "#c89b3c",
}

# IUCN status palette
STATUS_META = {
    "LC": ("Least Concern",          "#2d8a5e"),
    "NT": ("Near Threatened",        "#b8a832"),
    "VU": ("Vulnerable",             "#d4a020"),
    "EN": ("Endangered",             "#d06030"),
    "CR": ("Critically Endangered",  "#c0362c"),
    "EW": ("Extinct in the Wild",    "#606060"),
    "EX": ("Extinct",                "#404040"),
}

T = DARK_TOKENS if st.session_state.theme == "dark" else LIGHT_TOKENS

# ---------------------------------------------------------------------------
# Global CSS — fonts, layout reset, component styles
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
      /* Reset Streamlit chrome */
      #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; height: 0; }}
      .stApp {{
        background: {T['background']};
        color: {T['foreground']};
        font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
      }}
      .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
      }}
      /* Typography */
      h1, h2, h3, .font-display {{
        font-family: 'Fraunces', ui-serif, Georgia, serif;
        letter-spacing: -0.02em;
        color: {T['foreground']};
      }}
      p, span, div, label {{ color: {T['foreground']}; }}
      .muted {{ color: {T['muted_foreground']}; }}
      .mono {{ font-family: ui-monospace, 'SF Mono', Menlo, monospace; }}

      /* Layout shell */
      .shell {{ max-width: 1120px; margin: 0 auto; padding: 0 20px; }}
      @media (min-width: 768px) {{ .shell {{ padding: 0 32px; }} }}

      /* Header */
      .bc-header {{
        position: sticky; top: 0; z-index: 30;
        background: {T['background']};
        backdrop-filter: blur(18px);
        border-bottom: 1px solid {T['border']};
      }}
      .bc-header-inner {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 8px 0;
      }}
      .bc-logo {{ display: flex; align-items: center; gap: 8px; }}
      .bc-logo svg {{ color: {T['primary']}; width: 24px; height: 24px; }}
      .header-logo-img {{ height: 48px; width: auto; object-fit: contain; border-radius: 50%; }}
      .footer-logo-img {{ height: 40px; width: auto; object-fit: contain; border-radius: 50%; }}
      .bc-logo .name {{ font-family: 'Fraunces'; font-weight: 600; font-size: 16px; line-height: 1; }}
      .bc-logo .sub {{ font-size: 9px; letter-spacing: 0.18em; text-transform: uppercase;
                       color: {T['muted_foreground']}; margin-top: 2px; }}

      /* Hero */
      .bc-hero {{
        background: {T['secondary']};
        border-bottom: 1px solid {T['border']};
      }}
      .bc-hero-grid {{
        display: grid; gap: 40px; align-items: center;
        padding: 56px 0;
      }}
      @media (min-width: 768px) {{
        .bc-hero-grid {{ grid-template-columns: 1.1fr 1fr; padding: 80px 0; }}
      }}
      .bc-eyebrow {{
        display: inline-flex; align-items: center; gap: 8px;
        background: {T['background']};
        border: 1px solid {T['border']};
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 11px; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.18em;
        color: {T['muted_foreground']};
      }}
      .bc-eyebrow .dot {{ width: 6px; height: 6px; border-radius: 999px; background: {T['accent']}; }}
      .bc-hero h1 {{
        font-size: clamp(36px, 5vw, 60px);
        font-weight: 600;
        line-height: 1.05;
        margin: 20px 0 0;
      }}
      .bc-hero h1 em {{ font-style: italic; color: {T['fern']}; font-weight: 500; }}
      .bc-hero p.lead {{
        margin: 20px 0 0; max-width: 560px;
        color: {T['muted_foreground']};
        font-size: 17px; line-height: 1.65;
      }}
      .bc-cta-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 32px; }}
      .bc-btn-primary {{
        display: inline-flex; align-items: center; gap: 8px;
        background: {T['primary']}; color: {T['primary_foreground']};
        padding: 12px 20px; border-radius: 999px;
        font-size: 14px; font-weight: 500;
        text-decoration: none;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.08);
        transition: opacity .2s;
      }}
      .bc-btn-primary:hover {{ opacity: .9; }}
      .bc-meta {{ display: inline-flex; align-items: center; gap: 8px; font-size: 12px; color: {T['muted_foreground']}; }}

      .bc-hero-card {{
        position: relative; border-radius: 24px; overflow: hidden;
        border: 1px solid {T['border']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.08), 0 24px 60px -12px rgba(0,0,0,0.22);
      }}
      .bc-hero-card img {{ width: 100%; aspect-ratio: 4/3; object-fit: cover; display: block; }}
      .bc-hero-tag {{
        position: absolute; left: 16px; right: 16px; bottom: 16px;
        display: flex; align-items: center; justify-content: space-between;
        background: {T['background']};
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 12px 16px;
        font-size: 12px;
      }}
      .bc-hero-tag .sci {{ font-family: ui-monospace; color: {T['muted_foreground']}; }}
      .bc-hero-tag .name {{ font-family: 'Fraunces'; font-weight: 500; font-size: 16px; margin-top: 2px; }}

      /* Section header */
      .bc-section {{ padding: 64px 0; }}
      @media (min-width: 768px) {{ .bc-section {{ padding: 96px 0; }} }}
      .bc-step {{
        font-size: 11px; text-transform: uppercase; letter-spacing: 0.2em;
        color: {T['muted_foreground']};
      }}
      .bc-h2 {{
        font-size: clamp(28px, 4vw, 38px);
        font-weight: 600; margin: 12px 0 0;
        letter-spacing: -0.02em;
      }}

      /* Uploader */
      .bc-uploader {{
        max-width: 720px; margin: 40px auto 0;
        border-radius: 32px;
        background: {T['card']};
        border: 2px dashed {T['border']};
        padding: 20px;
      }}
      .bc-uploader-inner {{
        aspect-ratio: 16/9;
        background: {T['secondary']};
        border-radius: 24px;
        display: flex; align-items: center; justify-content: center;
        overflow: hidden;
        position: relative;
      }}
      .bc-uploader-empty {{ text-align: center; padding: 24px; }}
      .bc-upload-icon {{
        width: 56px; height: 56px; border-radius: 999px;
        background: rgba(26,74,50,0.12);
        color: {T['primary']};
        display: inline-flex; align-items: center; justify-content: center;
        margin-bottom: 12px;
      }}

      /* Override Streamlit file uploader to look like the design */
      [data-testid="stFileUploader"] {{ width: 100%; }}
      [data-testid="stFileUploader"] section {{
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
      }}
      [data-testid="stFileUploader"] section > div {{
        background: {T['card']};
        border: 2px dashed {T['border']} !important;
        border-radius: 32px !important;
        padding: 32px 20px !important;
        transition: border-color .2s, background .2s;
      }}
      [data-testid="stFileUploader"] section > div:hover {{
        border-color: {T['primary']} !important;
      }}
      [data-testid="stFileUploader"] small,
      [data-testid="stFileUploader"] span {{ color: {T['muted_foreground']} !important; }}
      [data-testid="stFileUploaderDropzone"] button,
      [data-testid="baseButton-secondary"] {{
        background: {T['primary']} !important;
        color: {T['primary_foreground']} !important;
        border: 0 !important;
        border-radius: 999px !important;
        padding: 8px 18px !important;
        font-weight: 500 !important;
      }}
      .stButton > button {{
        background: {T['card']} !important;
        color: {T['foreground']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 999px !important;
        padding: 8px 18px !important;
        font-weight: 500 !important;
        transition: background .2s;
      }}
      .stButton > button:hover {{
        background: {T['secondary']} !important;
        border-color: {T['border']} !important;
      }}
      .stButton.primary > button {{
        background: {T['primary']} !important;
        color: {T['primary_foreground']} !important;
        border-color: transparent !important;
      }}
      
      /* Theme toggle button */
      [key="theme_toggle"] button {{
        background: transparent !important;
        border: 1px solid {T['border']} !important;
        border-radius: 6px !important;
        padding: 4px 6px !important;
        font-size: 14px !important;
        min-height: 28px !important;
        height: 28px !important;
      }}
      [key="theme_toggle"] button:hover {{
        background: {T['secondary']} !important;
      }}

      /* Result card */
      .bc-card {{
        background: {T['card']};
        border: 1px solid {T['border']};
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.08);
        transition: transform .25s, box-shadow .25s;
        animation: bcRise .55s cubic-bezier(.2,.7,.2,1) both;
        height: 100%;
        display: flex; flex-direction: column;
      }}
      .bc-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.08), 0 24px 60px -12px rgba(0,0,0,0.22);
      }}
      .bc-card.top {{
        border-width: 2px;
        border-color: {T['primary']};
      }}
      @keyframes bcRise {{
        from {{ opacity: 0; transform: translateY(16px); }}
        to {{ opacity: 1; transform: translateY(0); }}
      }}

      .bc-card-image {{
        position: relative; aspect-ratio: 5/3; overflow: hidden;
      }}
      .bc-card-image img {{
        width: 100%; height: 100%; object-fit: cover;
        transition: transform .7s;
      }}
      .bc-card:hover .bc-card-image img {{ transform: scale(1.05); }}
      .bc-top-ribbon {{
        position: absolute; top: 14px; right: 14px; z-index: 2;
        background: {T['accent']};
        color: #1a2e22;
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em;
        padding: 4px 10px; border-radius: 999px;
      }}
      .bc-card-body {{ padding: 22px; display: flex; flex-direction: column; gap: 14px; flex: 1; }}
      .bc-card .rank {{
        font-family: ui-monospace; font-size: 10px; letter-spacing: 0.12em;
        text-transform: uppercase; color: {T['muted_foreground']};
      }}
      .bc-card h3 {{
        font-family: 'Fraunces'; font-size: 24px; font-weight: 600; line-height: 1.15;
        margin: 4px 0 0;
      }}
      .bc-card .sci {{
        font-family: 'Fraunces'; font-style: italic; font-size: 14px;
        color: {T['muted_foreground']}; margin-top: 2px;
      }}

      /* Status badge */
      .bc-status {{
        display: inline-flex; align-items: center; gap: 6px;
        background: {T['background']};
        backdrop-filter: blur(8px);
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 10px; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.08em;
      }}
      .bc-status .dot {{ width: 6px; height: 6px; border-radius: 999px; }}
      .bc-status .lbl {{ color: {T['muted_foreground']}; text-transform: none; letter-spacing: normal; }}
      .bc-status-pos {{
        position: absolute; left: 12px; top: 12px; z-index: 2;
        border: 1px solid rgba(128,128,128,0.25);
      }}

      /* India indicator */
      .bc-india {{
        display: inline-flex; align-items: center; gap: 4px;
        background: rgba(200,155,60,0.15);
        border: 1px solid rgba(200,155,60,0.30);
        padding: 2px 8px; border-radius: 999px;
        font-size: 10px; font-weight: 500;
      }}

      /* Confidence bar */
      .bc-conf-row {{
        display: flex; justify-content: space-between;
        font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
        color: {T['muted_foreground']};
      }}
      .bc-conf-row .v {{ font-family: ui-monospace; color: {T['foreground']}; }}
      .bc-conf-track {{
        height: 6px; width: 100%; border-radius: 999px;
        background: {T['secondary']}; overflow: hidden; margin-top: 6px;
      }}
      .bc-conf-fill {{
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, {T['fern']}, {T['primary']});
        transition: width .8s ease;
      }}

      .bc-desc {{ font-size: 14px; line-height: 1.6; color: {T['muted_foreground']}; }}

      .bc-fact {{
        background: rgba(200,155,60,0.12);
        border: 1px solid rgba(200,155,60,0.25);
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: auto;
      }}
      .bc-fact .label {{
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      .bc-fact p {{ margin: 6px 0 0; font-size: 14px; line-height: 1.55; }}

      /* Loading skeleton */
      @keyframes bcPulse {{ 0%,100% {{ opacity: .6 }} 50% {{ opacity: 1 }} }}
      .bc-skel {{
        background: {T['secondary']};
        border-radius: 8px;
        animation: bcPulse 1.4s ease-in-out infinite;
      }}

      /* Footer */
      .bc-footer {{
        background: {T['secondary']};
        border-top: 1px solid {T['border']};
        padding: 32px 0 16px;
        margin-top: 48px;
      }}
      .bc-footer-inner {{
        display: flex; flex-direction: column; gap: 16px;
        justify-content: space-between; align-items: flex-start;
      }}
      @media (min-width: 768px) {{
        .bc-footer-inner {{ flex-direction: row; align-items: center; }}
      }}
      .bc-footer .links {{ display: flex; gap: 20px; font-size: 12px; color: {T['muted_foreground']}; }}
      .bc-copy {{
        border-top: 1px solid {T['border']};
        text-align: center; padding-top: 12px; margin-top: 24px;
        font-size: 11px; color: {T['muted_foreground']};
      }}

      /* Hide the streamlit "deploy" toolbar / running indicator */
      [data-testid="stToolbar"] {{ display: none; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    rank: int
    common_name: str
    scientific_name: str
    confidence: float
    status: str
    description: str
    fun_fact: str
    native_to_india: bool = False
    image_data_uri: Optional[str] = None  # base64 data URI for inline display


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def img_to_data_uri(img: Image.Image, fmt: str = "JPEG", quality: int = 88) -> str:
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def file_to_data_uri(path: Path) -> Optional[str]:
    """Load a packaged sample image as a data URI. Returns None if file missing or on error."""
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        # Detect format from file extension
        fmt = path.suffix.lstrip('.').upper()
        if fmt not in ['JPEG', 'JPG', 'PNG', 'WEBP', 'GIF']:
            fmt = "JPEG"
        if fmt == "JPG":
            fmt = "JPEG"
        return img_to_data_uri(img, fmt=fmt, quality=88)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Species metadata loader
# ---------------------------------------------------------------------------
@st.cache_data
def load_species_db() -> dict:
    db_path = Path(__file__).parent / "species_data.json"
    if db_path.exists():
        with open(db_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Real model predictor
# ---------------------------------------------------------------------------
_WEIGHTS_PATH = Path(__file__).parent / "big_cats_efficientnet.pth"


@st.cache_resource(show_spinner=False)
def load_classifier(weights_mtime: float = 0.0):
    """Load and cache the classifier, keyed on the weights file mtime.

    Using the mtime as a cache key means Streamlit invalidates the cache
    automatically whenever the .pth file is added or replaced, preventing
    a stale DemoClassifier from persisting across restarts.
    """
    from model import get_classifier
    return get_classifier()


def predict(image: Image.Image, uploaded_uri: str) -> List[Prediction]:
    """
    Run inference and return top-3 Prediction objects.

    Uses the fine-tuned EfficientNet-B0 if `big_cats_efficientnet.pth` is present,
    otherwise falls back to CLIP ViT-B/32 zero-shot classification.
    """
    species_db = load_species_db()
    weights_mtime = _WEIGHTS_PATH.stat().st_mtime if _WEIGHTS_PATH.exists() else 0.0
    classifier = load_classifier(weights_mtime)

    # Determine which model is active and show appropriate banner
    from model import EfficientNetClassifier, CLIPClassifier, _load_error
    if isinstance(classifier, EfficientNetClassifier):
        pass  # fine-tuned model — no banner needed
    elif isinstance(classifier, CLIPClassifier):
        st.info("🔍 **CLIP zero-shot** — no fine-tuned weights found. Using zero-shot classification.", icon=None)
    else:
        if _load_error and _WEIGHTS_PATH.exists():
            st.error(f"⚠️ EfficientNet weights found but failed to load: {_load_error}")
        else:
            st.info(
                "⚠️ **Demo mode** — no model loaded. "
                "Results below are illustrative only. "
                "Run the training notebook to get real weights, or ensure internet access for CLIP.",
                icon=None,
            )

    # Run inference
    raw_results = classifier.predict(image, top_k=3)  # [(class_key, confidence), ...]

    predictions = []
    for rank, (class_key, conf) in enumerate(raw_results, start=1):
        meta = species_db.get(class_key, {})

        # Image for the card: rank 1 always shows the uploaded image.
        # Ranks 2-3 try a sample image for that species; if none exists,
        # fall back to the uploaded image (better than a blank card).
        if rank == 1:
            img_uri = uploaded_uri
        else:
            sample_path = Path(__file__).parent / "samples" / f"{class_key}.jpg"
            if sample_path.exists():
                img_uri = file_to_data_uri(sample_path)
            else:
                img_uri = uploaded_uri  # reuse uploaded photo rather than showing blank

        predictions.append(Prediction(
            rank=rank,
            common_name=meta.get("common_name", class_key.replace("_", " ").title()),
            scientific_name=meta.get("scientific_name", ""),
            confidence=round(conf * 100, 1),
            status=meta.get("iucn_status", "DD"),
            description=meta.get("description", ""),
            fun_fact=meta.get("fun_fact", ""),
            native_to_india=meta.get("native_to_india", False),
            image_data_uri=img_uri,
        ))

    return predictions


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
LOGO_SVG = """
<svg viewBox="0 0 32 32" width="32" height="32" fill="none" aria-hidden="true">
  <circle cx="16" cy="19" r="7.5" fill="currentColor" fill-opacity="0.14" />
  <ellipse cx="16" cy="20" rx="5" ry="4.2" fill="currentColor" />
  <ellipse cx="9"  cy="13" rx="2.2" ry="2.6" fill="currentColor" />
  <ellipse cx="23" cy="13" rx="2.2" ry="2.6" fill="currentColor" />
  <ellipse cx="11.5" cy="8" rx="1.8" ry="2.2" fill="currentColor" />
  <ellipse cx="20.5" cy="8" rx="1.8" ry="2.2" fill="currentColor" />
</svg>
"""


def render_header():
    # Logo + theme toggle row
    col_logo, col_toggle = st.columns([6, 1])
    with col_logo:
        logo_path = Path(__file__).parent / "samples" / "Logo(Big-Cats).png"
        logo_uri = file_to_data_uri(logo_path) or ""
        
        st.markdown(
            f"""
            <div class="bc-header">
              <div class="shell bc-header-inner">
                <div class="bc-logo">
                  {f'<img src="{logo_uri}" alt="Big Cats Logo" class="header-logo-img"/>' if logo_uri else LOGO_SVG}
                </div>
                <div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # Float a real Streamlit button on top for theme toggle
    with col_toggle:
        st.write("")  # spacer
        label = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(label, key="theme_toggle", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()


def render_hero():
    samples_dir = Path(__file__).parent / "samples"
    hero_uri = file_to_data_uri(samples_dir / "hero.jpg") or ""
    hero_img_html = (
        f'<img src="{hero_uri}" alt="Bengal tiger at sunset"/>'
        if hero_uri
        else '<div style="aspect-ratio:4/3;background:linear-gradient(135deg,#3a2410,#7a4520);"></div>'
    )
    st.markdown(
        f"""
        <section class="bc-hero">
          <div class="shell bc-hero-grid">
            <div>
              <span class="bc-eyebrow"><span class="dot"></span>AI Wildlife Identification</span>
              <h1>Discover species <em>instantly</em> from your camera.</h1>
              <p class="lead">Upload a photo of any bird or wild animal. Get the top 3 matches
                with confidence scores, IUCN conservation status, and a fact you'll want to share.</p>
              <div class="bc-cta-row">
                <a class="bc-btn-primary" href="#identify">Identify a photo →</a>
                <span class="bc-meta">
                  <span style="width:6px;height:6px;border-radius:999px;background:{STATUS_META['LC'][1]};display:inline-block"></span>
                  Free · No sign-up required
                </span>
              </div>
            </div>
            <div class="bc-hero-card">
              {hero_img_html}
              <div class="bc-hero-tag">
                <div>
                  <div class="sci">Panthera tigris tigris</div>
                  <div class="name">Bengal Tiger</div>
                </div>
                <span class="bc-status" style="border:1px solid rgba(208,96,48,0.4)">
                  <span class="dot" style="background:{STATUS_META['EN'][1]}"></span>EN
                </span>
              </div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_html(status: str, position: str = "") -> str:
    label, color = STATUS_META.get(status, ("Unknown", T["muted_foreground"]))
    pos_class = "bc-status-pos" if position == "abs" else ""
    return (
        f'<span class="bc-status {pos_class}">'
        f'<span class="dot" style="background:{color}"></span>'
        f'<span class="mono">{status}</span>'
        f'<span class="lbl"> · {label}</span>'
        f"</span>"
    )


def render_card(p: Prediction, delay_ms: int) -> str:
    is_top = p.rank == 1
    if p.image_data_uri:
      image_block = (
              f'<div class="bc-card-image">'
              f'{render_status_html(p.status, "abs")}'
              f'<img src="{p.image_data_uri}" alt="{p.common_name}" loading="lazy"/>'
              f'</div>'
          )
    else:
        image_block = (
            f'<div class="bc-card-image" style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);display:flex;align-items:center;justify-content:center;">'
            f'{render_status_html(p.status, "abs")}'
            f'<div style="text-align:center;color:rgba(255,255,255,0.3);font-size:48px;">📷</div>'
            f'</div>'
        )
    ribbon = '<span class="bc-top-ribbon">Top match</span>' if is_top else ""
    india = '<span class="bc-india">📍 Native to India</span>' if p.native_to_india else ""
    return (
        f'<div class="bc-card {"top" if is_top else ""}" style="animation-delay:{delay_ms}ms">'
        f'{ribbon}'
        f'{image_block}'
        f'<div class="bc-card-body">'
        f'<div>'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span class="rank">#{p.rank}</span>'
        f'{india}'
        f'</div>'
        f'<h3>{p.common_name}</h3>'
        f'<div class="sci">{p.scientific_name}</div>'
        f'</div>'
        f'<div>'
        f'<div class="bc-conf-row">'
        f'<span>Confidence</span>'
        f'<span class="v">{p.confidence:.1f}%</span>'
        f'</div>'
        f'<div class="bc-conf-track">'
        f'<div class="bc-conf-fill" style="width:{p.confidence:.1f}%"></div>'
        f'</div>'
        f'</div>'
        f'<p class="bc-desc">{p.description}</p>'
        f'<div class="bc-fact">'
        f'<span class="label">★ Fun fact</span>'
        f'<p>{p.fun_fact}</p>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def render_skeletons():
    skeleton_html = (
        '<div class="bc-card" style="height:100%">'
        '<div class="bc-skel" style="aspect-ratio:5/3;border-radius:0"></div>'
        '<div class="bc-card-body">'
        '<div class="bc-skel" style="height:18px;width:65%"></div>'
        '<div class="bc-skel" style="height:12px;width:45%"></div>'
        '<div class="bc-skel" style="height:6px;width:100%"></div>'
        '<div class="bc-skel" style="height:12px;width:100%"></div>'
        '<div class="bc-skel" style="height:12px;width:85%"></div>'
        '</div>'
        '</div>'
    )
    cols = st.columns(3, gap="medium")
    for c in cols:
        with c:
            st.markdown(skeleton_html, unsafe_allow_html=True)


def render_results(results: List[Prediction]):
    st.markdown(
        """
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:12px;margin-top:48px;margin-bottom:20px;">
          <div>
            <div class="bc-step">Step 02 — Results</div>
            <h3 class="bc-h2" style="font-size:28px;">Top 3 predictions</h3>
          </div>
          <div class="muted" style="font-size:12px;">
            Conservation data from <span style="color:var(--fg)">IUCN Red List</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3, gap="medium")
    for col, p in zip(cols, results):
        with col:
            st.markdown(render_card(p, delay_ms=(p.rank - 1) * 90), unsafe_allow_html=True)


def render_footer():
    logo_path = Path(__file__).parent / "samples" / "Logo(Big-Cats).png"
    footer_logo_uri = file_to_data_uri(logo_path) or ""
    footer_logo_html = f'<img src="{footer_logo_uri}" alt="Big Cats Logo" class="footer-logo-img"/>' if footer_logo_uri else LOGO_SVG
    
    st.markdown(
        f"""
        <footer class="bc-footer">
          <div class="shell">
            <div class="bc-footer-inner">
              <div>
                <div class="bc-logo">{footer_logo_html}
                  <div>
                    <div class="name">Big Cats</div>
                    <div class="sub">Wildlife ID</div>
                  </div>
                </div>
                <p class="muted" style="margin-top:12px;font-size:12px;max-width:380px;">
                  A pocket field guide for the curious. Conservation data from the IUCN Red List.
                </p>
              </div>
              <div class="links">
                <a href="#" style="color:inherit;text-decoration:none">About</a>
                <a href="#" style="color:inherit;text-decoration:none">API</a>
                <a href="#" style="color:inherit;text-decoration:none">Privacy</a>
                <a href="#" style="color:inherit;text-decoration:none">Contact</a>
              </div>
            </div>
            <div class="bc-copy">© 2025 Big Cats. Built for nature lovers.</div>
          </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page composition
# ---------------------------------------------------------------------------
render_header()
render_hero()

# ---- Identify section ------------------------------------------------------
st.markdown('<div id="identify"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <section class="bc-section">
      <div class="shell" style="text-align:center;max-width:640px;margin:0 auto;">
        <div class="bc-step">Step 01 — Upload</div>
        <h2 class="bc-h2">Drop a photo. Get the top 3 matches.</h2>
        <p class="muted" style="margin-top:12px;">
          We'll analyze it against millions of wildlife images and return ranked predictions in seconds.
        </p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Uploader (centered, narrower column)
left, mid, right = st.columns([1, 3, 1])
with mid:
    uploaded = st.file_uploader(
        "Drop an image here, or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="uploader",
    )

    btn_l, btn_r = st.columns(2)
    with btn_l:
        sample_clicked = st.button("✦ Try a sample", use_container_width=True, key="sample_btn")
    with btn_r:
        clear_clicked = st.button("Clear", use_container_width=True, key="clear_btn")

    if clear_clicked:
        st.session_state.results = None
        st.session_state.preview_bytes = None
        st.rerun()

    # Decide what to process
    process_image: Optional[Image.Image] = None
    process_uri: Optional[str] = None

    if uploaded is not None:
        img = Image.open(uploaded)
        process_image = img
        process_uri = img_to_data_uri(img)
    elif sample_clicked:
        sample_path = Path(__file__).parent / "samples" / "tiger.jpg"
        if sample_path.exists():
            img = Image.open(sample_path)
            process_image = img
            process_uri = img_to_data_uri(img)
            st.session_state.last_key = None  # force re-prediction even if same image
        else:
            st.warning("Drop a `tiger.jpg` in `./samples/` to enable the sample.")

    # Show preview + run prediction
    if process_image is not None and process_uri is not None:
        # Only re-run if input changed
        cache_key = process_uri[:200]
        if st.session_state.get("last_key") != cache_key:
            st.session_state.last_key = cache_key
            st.session_state.results = None

            preview_holder = st.empty()
            preview_holder.markdown(
                f"""
                <div class="bc-uploader">
                  <div class="bc-uploader-inner" style="background:#000;">
                    <img src="{process_uri}" style="width:100%;height:100%;object-fit:cover;"/>
                  </div>
                  <p class="muted" style="text-align:center;margin-top:14px;font-size:13px;">
                    <span class="mono">●</span> Analyzing image — matching species patterns…
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Skeleton row while predicting
            skel_holder = st.empty()
            with skel_holder.container():
                render_skeletons()

            results = predict(process_image, process_uri)
            st.session_state.results = results
            preview_holder.empty()
            skel_holder.empty()
            st.rerun()

# Results section (full-width)
if st.session_state.results:
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    render_results(st.session_state.results)
    st.markdown("</div>", unsafe_allow_html=True)

render_footer()

