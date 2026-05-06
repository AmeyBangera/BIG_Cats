# 🐯 Big Cats — Wildlife Species Identifier

**SMAI Assignment 3 · Variant T7.5 · IIIT Hyderabad 2025–26**  
Team: Agyeya (2023101055) · Abdul (2023102024) · Amey (2023115019)

---

## Running Locally (Windows / Mac / Linux)

### Step 1 — Clone and install

```bash
git clone https://github.com/your-username/BIG_Cats.git
cd BIG_Cats
pip install -r requirements.txt
```

> **Windows tip:** if `pip install torch` is slow, grab the CPU wheel directly from
> https://pytorch.org/get-started/locally/ and install that first.

### Step 2 — Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### What you'll see without trained weights

The app has three automatic modes — no config needed:

| Mode | When | Accuracy |
|------|------|----------|
| **EfficientNet-B0** | `big_cats_efficientnet.pth` is in the project folder | Best (~92%) |
| **CLIP zero-shot** | No `.pth` file, but internet available (downloads ~600 MB once) | ~60% |
| **Demo mode** | No `.pth` file and no internet | Shows sample predictions, UI fully functional |

A banner at the top of the results tells you which mode is active.

### Getting the trained weights

1. Open `train_big_cats.ipynb` in Google Colab
2. Upload `kaggle.json` when prompted (or place it in `My Drive/BIG_Cats/`)
3. Run all cells (~15 min on T4 GPU)
4. Download `big_cats_efficientnet.pth` from Colab (or from Drive)
5. Place it in the same folder as `app.py`
6. Restart the Streamlit app — it will automatically use EfficientNet

---

## Project Structure

```
BIG_Cats/
├── app.py                   # Streamlit web application
├── model.py                 # EfficientNet / CLIP / Demo inference wrappers
├── species_data.json        # IUCN status, descriptions, fun facts per species
├── train_big_cats.ipynb     # Colab training notebook
├── requirements.txt
├── samples/
│   ├── hero.jpg
│   ├── tiger.jpg            # Used by "Try a sample" button
│   ├── leopard.jpg
│   ├── lion.jpg
│   └── Logo(Big-Cats).png
└── README.md
```

> **Note:** `big_cats_efficientnet.pth` (~20 MB) is not included in the repo.
> Train it yourself using the notebook, or download it from the project's HF Space.

---

## Deploying to HuggingFace Spaces

1. Create a new Space → SDK: Streamlit
2. Push all files (excluding `.pth` — it's in `.gitignore`)
3. In the Space settings add a secret: `HF_TOKEN` (if needed)
4. Upload `big_cats_efficientnet.pth` via the Files tab in the Space repo
5. App goes live at `https://huggingface.co/spaces/your-username/big-cats`

Without the `.pth` file the Space will run in CLIP zero-shot mode automatically.

---

## Species Covered

| Species | Scientific Name | IUCN | Native to India |
|---------|----------------|------|-----------------|
| Bengal Tiger | *Panthera tigris tigris* | EN | ✅ |
| Asiatic Lion | *Panthera leo persica* | EN | ✅ |
| Indian Leopard | *Panthera pardus fusca* | VU | ✅ |
| Snow Leopard | *Panthera uncia* | VU | ✅ |
| Clouded Leopard | *Neofelis nebulosa* | VU | ✅ |
| Cheetah | *Acinonyx jubatus* | VU | ➕ reintroduced |
| Jaguar | *Panthera onca* | NT | ❌ |
| Puma | *Puma concolor* | LC | ❌ |

---

## Acknowledgements

- Dataset: Patricia Brezeanu — [Kaggle](https://www.kaggle.com/datasets/patriciabrezeanu/big-cats-image-classification-dataset)
- Models: `timm` (rwightman), CLIP (OpenAI via Hugging Face)
- IUCN Red List for conservation status
- LLMs used: Claude (Anthropic) for code scaffolding and report drafting. All evaluation and analysis are our own.
