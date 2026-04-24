# 🦁 Indian Wildlife & Bird Identifier

A beautiful Streamlit web application for identifying Indian wildlife and birds from photos. Get species predictions, conservation status, and interesting facts!

## Features

✨ **Key Features:**
- 📸 **Image Upload** - Simple drag-and-drop interface for wildlife photos
- 🤖 **AI-Powered Predictions** - CLIP zero-shot or fine-tuned EfficientNet models
- 🏆 **Top-3 Predictions** - Ranked species with confidence scores
- 🛡️ **IUCN Status Badges** - Conservation status with color-coded severity
- 📚 **Wikipedia Excerpts** - Detailed information about each species
- 💡 **Fun Facts** - Interesting tidbits powered by Gemini API
- 💾 **Smart Caching** - Efficient data caching for fast responses

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd BIG_Cats

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Configuration

### Environment Variables

Set up your Gemini API key (optional, for fun facts):

```bash
export GEMINI_API_KEY="your-api-key-here"
```

On Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
```

### Model Selection

Edit the configuration in `app.py` to choose your model:

```python
# Option 1: CLIP Zero-Shot (no training required)
model = WildlifeIdentifier(model_type="clip", model_name="openai/clip-vit-base-patch32")

# Option 2: Fine-Tuned EfficientNet
model = WildlifeIdentifier(model_type="efficientnet", model_name="efficientnet_b0")
```

## Project Structure

```
BIG_Cats/
├── app.py                 # Main Streamlit application
├── model.py              # CLIP & EfficientNet model wrappers
├── data_manager.py       # IUCN status & Wikipedia data management
├── config.py             # Configuration and utilities
├── requirements.txt      # Python dependencies
├── species_db.json       # Species database (auto-created)
└── species_data_cache.json  # Cached species data (auto-created)
```

## How to Integrate Your Model

### 1. Using CLIP (Zero-Shot, Recommended)

```python
from model import WildlifeIdentifier, INDIAN_SPECIES

# Initialize model
identifier = WildlifeIdentifier(model_type="clip")

# Load and process image
image = Image.open("wildlife.jpg")

# Get predictions
predictions = identifier.predict_clip_zeroshot(
    image=image,
    species_list=INDIAN_SPECIES,
    top_k=3
)

# Results: [("Bengal Tiger", 0.95), ("Leopard", 0.04), ("Bear", 0.01)]
```

### 2. Using Fine-Tuned EfficientNet

```python
# Initialize model
identifier = WildlifeIdentifier(model_type="efficientnet")

# Get predictions
predictions = identifier.predict_efficientnet(
    image=image,
    top_k=3
)
```

### 3. Integration in Frontend

Example integration in `app.py`:

```python
from model import get_model, INDIAN_SPECIES
from PIL import Image

# Get cached model
model = get_model(model_type="clip")

# Process uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Get predictions
    predictions = model.predict_clip_zeroshot(
        image=image,
        species_list=INDIAN_SPECIES,
        top_k=3
    )
    
    # predictions = [("Species", confidence), ...]
    
    # Get species info and display
    for rank, (species, confidence) in enumerate(predictions, 1):
        info = data_manager.get_species_info(species)
        display_prediction_card(rank, species, confidence, info)
```

## Data Sources

- **IUCN Red List**: Conservation status data
- **Gemini API**: Fun facts and dynamic content
- **Species Database**: Curated JSON database with Wikipedia excerpts

## UI/UX Highlights

🎨 **Design Features:**
- Gradient backgrounds and modern color scheme
- Responsive grid layout for predictions
- Hover animations on prediction cards
- Color-coded IUCN status badges
- Mobile-friendly responsive design
- Smooth transitions and visual feedback

### Color Scheme

- **Primary**: Purple gradient (#667eea → #764ba2)
- **Endangered Status**: Red to Orange (#d32f2f → #f57c00)
- **Protected Status**: Green (#388e3c)
- **Data Deficient**: Gray (#90a4ae)

## API Integration

### IUCN Status

The app includes a built-in IUCN status database. For custom updates:

```python
from data_manager import DataManager

dm = DataManager()
status = dm.get_iucn_status("Bengal Tiger")
# Returns: {"status": "Endangered", "population": "~2,500", ...}
```

### Gemini API for Fun Facts

```python
import os
from data_manager import DataManager

api_key = os.getenv("GEMINI_API_KEY")
dm = DataManager(gemini_api_key=api_key)

# Auto-fetches fun facts and caches them
info = dm.get_species_info("Bengal Tiger")
```

## Requirements

```
streamlit==1.28.1           # Web framework
Pillow==10.0.1             # Image processing
torch==2.0.1               # Deep learning
torchvision==0.15.2        # Vision models
transformers==4.33.0       # CLIP models
google-generativeai==0.3.0 # Gemini API
```

## Performance Tips

1. **Caching**: The app uses Streamlit's `@st.cache_resource` for model loading
2. **Batch Processing**: Process multiple images efficiently with proper batching
3. **GPU Acceleration**: Automatic GPU detection and usage
4. **Data Caching**: Species information cached locally in JSON

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model takes too long to load | GPU not detected. Install CUDA drivers. |
| Gemini API errors | Check API key, rate limits, and internet connection. |
| Out of memory | Reduce image size or use a smaller model variant. |
| Cache not updating | Delete `species_data_cache.json` and restart. |

## Future Enhancements

- 🌍 Expand to global wildlife species
- 🎯 Custom model fine-tuning interface
- 🎬 Video frame analysis
- 📊 Population trend charts
- 🗺️ Geographic distribution maps
- 🌐 Multi-language support
- 📱 Mobile app version

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is provided as-is for educational and research purposes.

## Contact & Support

For questions, issues, or suggestions, please open an issue in the repository.

---

**Made with ❤️ for Indian Wildlife Conservation**