# photo-style-rl

An adaptive photo editing system that learns personal visual preferences through RLHF and applies them via a fine-tuned diffusion model. Built from scratch as a learning project, evolving through 11 iterative phases from basic CLIP embeddings to LoRA fine-tuning of InstructPix2Pix.

## What It Does

1. Takes a raw/unedited photo as input
2. Optionally accepts a text prompt (e.g., "warm moody tones", "fujifilm classic chrome")
3. Applies a learned color/tone transformation that matches my personal editing style
4. Outputs a styled photo with preserved detail and no artifacts

Trained on 270 paired before/after edits from my travel photography (Sony A7III, edited in Lightroom) plus 30 Fuji X100VI reference photos defining my target aesthetic.

## Architecture Evolution

This project iterated through multiple approaches, documented in the notebooks:

| Phase | Approach | Result |
|-------|----------|--------|
| 01 | CLIP embeddings | Scene understanding works, good foundation |
| 02 | Global style parameters (7 sliders) | Learned tendencies but crude application |
| 03 | RLHF rating system + reward model | 211 ratings collected, reward model predicts preferences |
| 04 | 3D LUT prediction | Artifact-free color grading, resolution-independent |
| 05 | CLIP-conditioned U-Net | Best paired training results at 256×256 |
| 06 | Fuji X100VI style reference | Style profile extraction, distribution matching |
| 07 | High-res detail refinement (512×512) | Two-stage pipeline, sharpness loss |
| 08 | Text-conditioned U-Net | CLIP text+image blending for prompted edits |
| 09 | NAFNet from scratch | Failed — insufficient data for stable training |
| 10 | Pretrained EfficientNet decoder | Failed — same data limitation |
| 11 | **LoRA fine-tuning of InstructPix2Pix** | **Best results — sharp, artifact-free, style-aware** |

**Current best model:** InstructPix2Pix + LoRA (r=4) fine-tuned on 270 personal edits.

## Tech Stack

- **PyTorch** — model training and inference
- **Hugging Face Diffusers** — InstructPix2Pix pipeline
- **PEFT (LoRA)** — parameter-efficient fine-tuning
- **OpenCLIP (ViT-B-32)** — image/text embeddings for conditioning
- **Google Colab Pro** — GPU compute (T4/L4/A100)
- **Google Drive** — persistent data storage

## Project Structure

```
photo-style-rl/
├── notebooks/                    # Colab notebooks (run in browser)
│   ├── 01_image_embeddings.ipynb
│   ├── 02_lut_generation.ipynb
│   ├── 03_reward_model.ipynb
│   ├── 04_lut_prediction.ipynb
│   ├── 05_image_to_image.ipynb
│   ├── 06_style_reference.ipynb
│   ├── 07_high_res.ipynb
│   ├── 08_text_conditioning.ipynb
│   ├── 09_pretrained_enhancement.ipynb
│   ├── 10_pretrained_finetune.ipynb
│   └── 11_lora_finetune.ipynb
├── src/                          # Reusable Python modules
│   ├── __init__.py
│   ├── embeddings.py
│   ├── lut_predictor.py
│   ├── reward_model.py
│   ├── apply_style.py
│   └── style_templates/
│       └── fuji_classic.json
├── data/                         # Local data (gitignored)
│   └── sample_images/
├── docs/
│   └── NOTES.md                  # Development log and decisions
├── requirements.txt
├── LICENSE
└── README.md
```

**Data and model checkpoints are stored on Google Drive** (not in this repo) to keep the repo lightweight. See `docs/NOTES.md` for the full development log.

## Key Learnings

- **Data > Architecture:** 270 paired images is insufficient for training image processing from scratch. Transfer learning (LoRA on a pretrained model) solved this.
- **LUTs are underrated:** The 3D LUT approach (Notebook 04) produces artifact-free results at any resolution. Good fallback for production use.
- **RLHF needs scale:** The reward model worked well (predicted vs actual R²≈0.95) but RL fine-tuning was unstable with <500 ratings.
- **Blur is a resolution problem, not a model problem:** Every model trained at 256×256 produced blur. The pretrained InstructPix2Pix (trained at 512×512 on 450K pairs) doesn't.

## Setup

```bash
git clone https://github.com/schenn99/photo-style-rl.git
pip install -r requirements.txt
```

Notebooks are designed to run in Google Colab with GPU. Open them via Google Drive or upload directly to Colab.

## Future Work

- [ ] Increase training data to 500+ clean paired edits
- [ ] Implement text-guided style variations ("warmer", "moodier", "more film-like")
- [ ] Build a simple web UI for uploading and styling photos
- [ ] Distill the LoRA model into a lightweight model for faster inference
- [ ] Add local adjustment capabilities (selective edits via SAM segmentation)

## License

MIT License — see [LICENSE](LICENSE) for details.
