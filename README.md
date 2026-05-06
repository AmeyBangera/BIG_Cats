# 🐯 Big Cats — Wildlife Species Identifier

**SMAI Assignment 3 · Variant T7.5 · IIIT Hyderabad 2025–26**

Upload a photo of a big cat and get the top-3 species predictions with confidence scores, IUCN conservation status, and curated fun facts.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## Species Covered

| Species | Scientific Name | IUCN Status | Native to India |
|---------|----------------|-------------|-----------------|
| Bengal Tiger | *Panthera tigris tigris* | Endangered | ✅ |
| Asiatic Lion | *Panthera leo persica* | Endangered | ✅ |
| Indian Leopard | *Panthera pardus fusca* | Vulnerable | ✅ |
| Snow Leopard | *Panthera uncia* | Vulnerable | ✅ |
| Clouded Leopard | *Neofelis nebulosa* | Vulnerable | ✅ |
| Cheetah | *Acinonyx jubatus* | Vulnerable | ➕ (reintroduced) |
| Jaguar | *Panthera onca* | Near Threatened | ❌ |
| Puma | *Puma concolor* | Least Concern | ❌ |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/BIG_Cats.git
cd BIG_Cats
pip install -r requirements.txt
```

### 2. (Optional) Download fine-tuned weights

If you have `big_cats_efficientnet.pth` (from training the notebook), place it in the project root. Otherwise the app automatically falls back to **CLIP ViT-B/32 zero-shot** — no weights file needed.

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
BIG_Cats/
├── app.py                     # Streamlit web application
├── model.py                   # EfficientNet-B0 + CLIP inference wrappers
├── species_data.json          # Curated IUCN status, descriptions, fun facts
├── train_big_cats.ipynb       # Colab training notebook
├── requirements.txt
├── samples/
│   ├── hero.jpg               # Hero image (Bengal tiger)
│   ├── tiger.jpg              # Sample for "Try a sample" button
│   ├── leopard.jpg
│   ├── lion.jpg
│   └── Logo(Big-Cats).png
└── README.md
```

---

## Model Architecture

### Inference Pipeline

The app uses a two-tier inference strategy:

1. **EfficientNet-B0 (primary)** — Fine-tuned on the Big Cats dataset. Loaded automatically if `big_cats_efficientnet.pth` is present in the project root.
2. **CLIP ViT-B/32 (fallback)** — Zero-shot classification using descriptive text prompts. No training data or weights file needed.

### Training (EfficientNet-B0)

See `train_big_cats.ipynb` for the full training pipeline. Summary:

| Hyperparameter | Value |
|---------------|-------|
| Base model | `timm` EfficientNet-B0 (ImageNet pretrained) |
| Epochs | 10 |
| Batch size | 32 |
| Optimizer | AdamW (lr=3e-4, weight decay=1e-4) |
| LR schedule | Cosine Annealing |
| Loss | CrossEntropy + Label Smoothing (0.1) |
| Augmentation | RandomResizedCrop, HFlip, ColorJitter, RandomRotation, RandomErasing |

### Dataset

- **Source:** [Big Cats Image Classification Dataset](https://www.kaggle.com/datasets/patriciabrezeanu/big-cats-image-classification-dataset) (Kaggle)
- 8 species, ~1,000–5,000 images per class
- Split: 75% train / 15% val / 10% test

---

## Deploying to HuggingFace Spaces

1. Create a new Space (Streamlit SDK)
2. Push all files including `requirements.txt`
3. Upload `big_cats_efficientnet.pth` to the Space repo (or rely on CLIP fallback)
4. The app will be live at `https://huggingface.co/spaces/your-username/big-cats`

---

## Acknowledgements

- Dataset: Patricia Brezeanu — [Kaggle](https://www.kaggle.com/datasets/patriciabrezeanu/big-cats-image-classification-dataset)
- Models: `timm` (rwightman), Hugging Face `transformers` (CLIP by OpenAI)
- IUCN Red List for conservation status data
- LLMs used: Claude (Anthropic) for code scaffolding, report drafting, and species descriptions. All evaluation and analysis are our own.

---

## License

MIT — for educational and research purposes.
