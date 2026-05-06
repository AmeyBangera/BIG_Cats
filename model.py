"""
model.py — Big Cats Species Classifier
=======================================

Three inference paths (auto-selected in priority order):

  1. EfficientNet-B0 fine-tuned  — if big_cats_efficientnet.pth is present
  2. CLIP ViT-B/32 zero-shot     — downloads ~600 MB on first run, then cached
  3. Demo mode                   — instant, no downloads, uses fixed sample predictions
                                   (for UI preview only — not real inference)

Usage:
    from model import get_classifier
    clf = get_classifier()
    results = clf.predict(pil_image, top_k=3)
    # -> [("tiger", 0.942), ("leopard", 0.038), ("lion", 0.012)]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Class labels (must match training folder order) ──────────────────────────
CLASS_NAMES: List[str] = [
    "cheetah",
    "clouded_leopard",
    "jaguar",
    "leopard",
    "lion",
    "tiger",
    "snow_leopard",
    "puma",
]

CLIP_PROMPTS: dict = {
    "cheetah":         "a photo of a cheetah, the fastest land animal with solid black spots",
    "clouded_leopard": "a photo of a clouded leopard with cloud-shaped markings in a forest",
    "jaguar":          "a photo of a jaguar, the large spotted cat of the Americas with rosette spots",
    "leopard":         "a photo of a leopard or Indian leopard with rosette markings",
    "lion":            "a photo of a lion or Asiatic lion, the large maned cat",
    "tiger":           "a photo of a Bengal tiger with orange coat and black stripes",
    "snow_leopard":    "a photo of a snow leopard with pale grey fur and dark rosettes in mountains",
    "puma":            "a photo of a puma, cougar or mountain lion, a tawny large cat",
}

WEIGHTS_PATH = Path(__file__).parent / "big_cats_efficientnet.pth"


def _to_tensor(output):
    """Extract a plain tensor from a CLIP output.

    Newer versions of transformers (>=4.40) may return a
    BaseModelOutputWithPooling instead of a raw tensor from
    get_text_features() / get_image_features().  This helper
    handles both cases.
    """
    if isinstance(output, torch.Tensor):
        return output
    # ModelOutput objects — try common attribute names
    for attr in ("pooler_output", "last_hidden_state"):
        val = getattr(output, attr, None)
        if val is not None:
            if val.ndim == 3:  # (batch, seq, dim) → take CLS token
                return val[:, 0]
            return val
    # Last resort: first value in the dataclass
    return next(iter(output.values()))


# ── 1. EfficientNet-B0 (fine-tuned) ─────────────────────────────────────────

class EfficientNetClassifier:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        import timm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        # Use class names saved in the checkpoint — preserves the exact training order
        # (ImageFolder sorts alphabetically, which differs from the hardcoded CLASS_NAMES order)
        self.class_names = checkpoint.get("class_names", CLASS_NAMES)
        self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(self.class_names))
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        from torchvision import transforms
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.info("EfficientNetClassifier loaded from %s", weights_path)

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        top_probs, top_idxs = probs.topk(min(top_k, len(self.class_names)))
        return [(self.class_names[i], float(p)) for i, p in zip(top_idxs, top_probs)]


# ── 2. CLIP zero-shot ────────────────────────────────────────────────────────

class CLIPClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPProcessor, CLIPModel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        texts = list(CLIP_PROMPTS.values())
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            tf = _to_tensor(self.model.get_text_features(**inputs))
            self.text_features = tf / tf.norm(dim=-1, keepdim=True)
        logger.info("CLIPClassifier loaded (%s)", model_name)

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            imf = _to_tensor(self.model.get_image_features(**inputs))
            imf = imf / imf.norm(dim=-1, keepdim=True)
            probs = torch.softmax((imf @ self.text_features.T).squeeze(0) * 100, dim=0)
        top_probs, top_idxs = probs.topk(min(top_k, len(CLASS_NAMES)))
        labels = list(CLIP_PROMPTS.keys())
        return [(labels[i], float(p)) for i, p in zip(top_idxs, top_probs)]


# ── 3. Demo mode (no model, no downloads) ────────────────────────────────────

class DemoClassifier:
    """
    Returns plausible-looking predictions without any model.
    Used for UI preview when no weights and no internet are available.
    Results are purely illustrative — not real inference.
    """
    _DEMO = [
        ("tiger",   0.942),
        ("leopard", 0.038),
        ("lion",    0.012),
    ]

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        return self._DEMO[:top_k]


# ── Factory ──────────────────────────────────────────────────────────────────

_classifier_cache = None
_load_error: Optional[str] = None  # set when EfficientNet weights exist but fail to load


def get_classifier(force_demo: bool = False):
    """
    Build and return a classifier.  Priority:
      1. EfficientNet-B0  (if .pth weights present)
      2. CLIP ViT-B/32    (downloads ~600 MB once, then cached locally)
      3. Demo mode        (instant, no downloads — UI preview only)

    NOTE: internal caching removed — Streamlit's @st.cache_resource (keyed on
    the weights file mtime) is the authoritative cache.  This prevents a stale
    DemoClassifier from surviving after the .pth file is added.
    """
    global _classifier_cache, _load_error
    _load_error = None

    if force_demo:
        _classifier_cache = DemoClassifier()
        return _classifier_cache

    # Try EfficientNet first
    if WEIGHTS_PATH.exists():
        try:
            _classifier_cache = EfficientNetClassifier(WEIGHTS_PATH)
            logger.info("Using fine-tuned EfficientNet-B0")
            return _classifier_cache
        except Exception as e:
            _load_error = str(e)
            logger.warning("EfficientNet load failed (%s); trying CLIP", e)

    # Try CLIP
    try:
        _classifier_cache = CLIPClassifier()
        logger.info("Using CLIP zero-shot classifier")
        return _classifier_cache
    except Exception as e:
        logger.warning("CLIP load failed (%s); falling back to demo mode", e)

    # Demo fallback
    logger.warning("No model available — running in demo mode (not real inference)")
    _classifier_cache = DemoClassifier()
    return _classifier_cache
