# photo-style-rl

An adaptive photo editing system that learns your visual preferences 
through reinforcement learning from human feedback (RLHF).

## What It Does
1. Analyzes an input photo using a pretrained vision model (CLIP/DINOv2)
2. Predicts personalized color grading parameters (3D LUT generation)
3. Applies the transformation to produce an edited image
4. Collects your rating to improve future edits via a learned reward model

Inspired by Fujifilm's film simulation presets — but adaptive and personal.

## Tech Stack
- **PyTorch** — model training and inference
- **CLIP / DINOv2** — pretrained vision encoders
- **Pillow / OpenCV** — image processing
- **Google Colab** — GPU compute (developed in Cursor IDE)
- **RLHF** — preference learning from user ratings

## Project Structure
```
photo-style-rl/
├── notebooks/
│   ├── 01_image_embeddings.ipynb    # Explore CLIP/DINOv2 on sample photos
│   ├── 02_lut_generation.ipynb      # Learn to 3_reward_model.ipynb        # Train preference model from ratings
├── src/
│   ├── __init__.py
│   ├── embeddings.py                # Image feature extraction
│   ├── lut_predictor.py             # LUT generation network
│   ├── reward_model.py              # Preference learning
│   ├── apply_style.py               # Apply LUT to images
│   ├── models/                      # Saved model weights (gitignored)
│   └── style_templates/
│       └── fuji_classic.json        # Base Fuji-style parameters
├── data/
│   ├── sample_images/               # Test photos (gitignored)
│   └── ratings/                     # Your preference ratings (gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

## Status
🚧 Early development

## Setup
```bash
git clone https://github.com/schenn99/photo-style-rl.git
pip install -r requirements.txt
```
Open notebooks in Cursor with Google Colab kernel for GPU access
cat > requirements.txt << 'EOF'
torch
torchvision
transformers
Pillow
opencv-python
numpy
matplotlib
open_clip_torch
