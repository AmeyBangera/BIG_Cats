"""
model.py — Big Cats Species Classifier
=======================================

Two inference paths:
  1. EfficientNet-B0 fine-tuned (if weights file present)
  2. CLIP ViT-B/32 zero-shot (automatic fallback / default when no weights)

Usage:
    from model import get_classifier
    clf = get_classifier()                        # auto-picks best available
    results = clf.predict(pil_image, top_k=3)
    # -> [("tiger", 0.942), ("leopard", 0.038), ("lion", 0.012)]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── class labels (must match training order) ────────────────────────────────
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

# CLIP text prompts — more descriptive = better zero-shot accuracy
CLIP_PROMPTS: dict = {
    "cheetah":        "a photo of a cheetah, the fastest land animal with solid black spots",
    "clouded_leopard": "a photo of a clouded leopard with cloud-shaped markings in a forest",
    "jaguar":         "a photo of a jaguar, the large spotted cat of the Americas with rosette spots",
    "leopard":        "a photo of a leopard or Indian leopard with rosette markings",
    "lion":           "a photo of a lion or Asiatic lion, the large maned cat",
    "tiger":          "a photo of a Bengal tiger with orange coat and black stripes",
    "snow_leopard":   "a photo of a snow leopard with pale grey fur and dark rosettes in mountains",
    "puma":           "a photo of a puma, cougar or mountain lion, a tawny large cat",
}

WEIGHTS_PATH = Path(__file__).parent / "big_cats_efficientnet.pth"


# ── EfficientNet-B0 classifier ───────────────────────────────────────────────

def _build_efficientnet(num_classes: int = len(CLASS_NAMES)):
    """Build EfficientNet-B0 with a custom classification head."""
    try:
        import timm
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=num_classes,
        )
        return model
    except ImportError:
        # Fallback: torchvision
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        import torch.nn as nn
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        return model


def _get_efficientnet_transforms():
    """Return torchvision transforms for EfficientNet inference."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class EfficientNetClassifier:
    """Fine-tuned EfficientNet-B0 inference wrapper."""

    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_efficientnet()
        checkpoint = torch.load(weights_path, map_location=self.device)
        # Support both raw state-dict and {"model_state_dict": ...} checkpoints
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = _get_efficientnet_transforms()
        logger.info("EfficientNetClassifier loaded from %s", weights_path)

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idxs = probs.topk(min(top_k, len(CLASS_NAMES)))
        return [(CLASS_NAMES[i], float(p)) for i, p in zip(top_idxs, top_probs)]


# ── CLIP zero-shot classifier ────────────────────────────────────────────────

class CLIPClassifier:
    """CLIP ViT-B/32 zero-shot classifier (no weights file required)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPProcessor, CLIPModel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # Pre-encode text prompts once
        texts = list(CLIP_PROMPTS.values())
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.get_text_features(**inputs)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        logger.info("CLIPClassifier loaded (%s)", model_name)

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_features = self.model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            scores = (img_features @ self.text_features.T).squeeze(0)
            probs = torch.softmax(scores * 100, dim=0)
        top_probs, top_idxs = probs.topk(min(top_k, len(CLASS_NAMES)))
        labels = list(CLIP_PROMPTS.keys())
        return [(labels[i], float(p)) for i, p in zip(top_idxs, top_probs)]


# ── Factory ──────────────────────────────────────────────────────────────────

_classifier_cache = None


def get_classifier(force_clip: bool = False):
    """
    Return a cached classifier instance.

    Priority:
      1. EfficientNet-B0 (if big_cats_efficientnet.pth exists and force_clip is False)
      2. CLIP ViT-B/32 zero-shot
    """
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache

    if not force_clip and WEIGHTS_PATH.exists():
        try:
            _classifier_cache = EfficientNetClassifier(WEIGHTS_PATH)
            logger.info("Using fine-tuned EfficientNet-B0")
            return _classifier_cache
        except Exception as e:
            logger.warning("Failed to load EfficientNet weights (%s); falling back to CLIP", e)

    _classifier_cache = CLIPClassifier()
    logger.info("Using CLIP zero-shot classifier")
    return _classifier_cache
