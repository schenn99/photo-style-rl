# Development Notes — photo-style-rl

A chronological log of development decisions, results, and lessons learned. Useful for context when prompting AI coding assistants (Cursor, Claude, etc.) or onboarding collaborators.

---

## Project Overview

**Goal:** Build a personal AI photo editor that learns my editing style from before/after pairs and applies it to new photos, optionally guided by text prompts.

**Data:**
- 270 paired images (raw + Lightroom-edited) from travel photography across Asia (Sony A7III)
- 30 Fuji X100VI reference photos (edited only, no raw pairs — defines target aesthetic)
- 211 human ratings (1-7 scale) with 76 text feedback entries

**Infrastructure:**
- Development: Cursor IDE (local) + Google Colab (GPU compute)
- Storage: Google Drive (`photo-style-rl/` folder) for images, checkpoints, ratings
- Version control: GitHub (`schenn99/photo-style-rl`)
- GPUs used: T4 (16GB), L4 (24GB), A100 (40GB) via Colab Pro student plan

---

## Phase 1: CLIP Embeddings (Notebook 01)

**What:** Extract 512-dimensional feature vectors from all images using OpenCLIP ViT-B-32 (pretrained on LAION-2B).

**Key results:**
- Similarity heatmap correctly clusters similar scenes (two sunsets at ~0.85 cosine similarity)
- Zero-shot text classification correctly identifies sunsets, landscapes, street scenes
- Edit shift analysis: mean 0.044 (subtle edits), max 0.256 (dramatic edits)
- Most edits are color/tone changes, confirming LUT-based approach is appropriate

**Decision:** CLIP embeddings are useful as conditioning signal — they encode scene content that can guide context-dependent editing.

---

## Phase 2: Global Style Parameters (Notebook 02)

**What:** Predict 7 Lightroom-style parameters (brightness, contrast, saturation, temperature, tint, shadow_lift, highlight_roll) from CLIP embeddings, then apply them with hand-coded pixel operations.

**Key results:**
- Model learned correct tendencies: darken images, add contrast, warm sunsets, mute street scenes
- `apply_style_params` function too crude — hand-coded multipliers produce artifacts
- Sunset photos particularly problematic (temperature delta of +0.274 is an outlier)

**Decision:** The prediction side works but the application side (hand-coded pixel math) is the bottleneck. Need a more expressive application method.

---

## Phase 3: RLHF Rating System (Notebook 03)

**What:** Rate model outputs 1-7, train a reward model on ratings, use RL to fine-tune the style predictor.

**Key results:**
- 211 images rated, mean rating 3.6/7
- Reward model predicts preferences accurately (scatter plot nearly diagonal)
- RL fine-tuning with KL constraint: predicted rating rose from 2.9 → 3.9
- BUT: RL-tuned outputs had artifacts (reward hacking despite KL constraint)
- Most common feedback: "blurry" (20), "artifacting" (20), "loss of detail" (15)

**Key insight:** Top complaints are resolution/architecture issues, not style issues. "Style is right but blurry" appeared 14 times.

**Decision:** RLHF architecture is sound but needs better base model before RL can help. Pivot to improving the base model.

---

## Phase 4: 3D LUT Prediction (Notebook 04)

**What:** Replace hand-coded `apply_style_params` with a learned 17×17×17 3D Look-Up Table predicted from CLIP embeddings. LUTs remap colors smoothly and are resolution-independent.

**Key results:**
- LUT preserves image structure perfectly (no flowers changing color, no neon artifacts)
- Color grading direction is correct
- Some banding in smooth gradients (LUT resolution limitation)
- Loss: 0.076 (L1 + SSIM + Color Histogram)

**Decision:** LUT is a good production fallback for artifact-free color grading. But can only do global color transforms, not local adjustments.

---

## Phase 5: CLIP-Conditioned U-Net (Notebook 05)

**What:** Build a U-Net that takes raw image pixels + CLIP embedding and outputs styled pixels. FiLM conditioning at the bottleneck steers style based on scene content.

**Key results:**
- Best paired training results at the time
- Loss dropped to 0.034 with 212 images
- Captures color/mood well but 256×256 resolution causes blur
- Residual architecture (predict edit, not output) stabilizes training

**Decision:** U-Net is the best approach for learning paired transforms. Blur is a resolution problem, not an architecture problem.

---

## Phase 6: Fuji X100VI Style Reference (Notebook 06)

**What:** Extract a "style DNA" profile from 30 Fuji X100VI photos and use it to fine-tune the U-Net toward Fuji's aesthetic.

**Key findings from style profile:**
- Fuji photos are cooler (more blue: +0.112 vs Lightroom edits)
- Fuji has lifted shadows (+0.062) — classic faded film look
- Fuji compresses highlights (-0.105) — soft rolloff
- Fuji is more saturated (+0.035) despite cooler tones

**Decision:** Style reference integration works. The U-Net v2 (Fuji-tuned) produces subtly improved results. Keep this as a data signal.

---

## Phase 7: High-Resolution Refinement (Notebook 07)

**What:** Two-stage pipeline — U-Net at 256×256 for style, then a lightweight refinement network at 512×512 using the raw image as a detail guide. Sharpness loss (Sobel gradient matching) to penalize blur.

**Key results:**
- Detail refinement shows improvement in crops (cherry blossoms, street scenes)
- Sharpness loss helps but can't fully overcome the 256→512 upscaling gap
- Refinement scale reached 0.29

**Decision:** Marginal improvement. The fundamental problem is training resolution, not post-processing.

---

## Phase 8: Text Conditioning (Notebook 08)

**What:** Modify U-Net to accept blended CLIP image+text embeddings. Train with 50% image-only, 25% matched text, 25% random text descriptions.

**Key results:**
- Text prompts visibly influence the edit (warm vs cool vs moody vs bright)
- Text weight slider works — smooth transition from 0.0 to 0.7
- Loss 0.028 (improved over text-free version)
- Base style preserved when no text is provided

**Decision:** Text conditioning works and adds genuine user value. Keep this feature.

---

## Phase 9: NAFNet Experiment (Notebook 09)

**What:** Train NAFNet (state-of-the-art image restoration architecture) from scratch at 384×384 with random crops.

**Key results:** FAILED
- Some images look great, others completely destroyed
- Training unstable with batch_size=2 and only 270 images
- Random crops + small batches = noisy gradients

**Lesson:** Training from scratch at higher resolution needs 500+ images minimum. Architecture isn't the bottleneck — data is.

---

## Phase 10: Pretrained EfficientNet Decoder (Notebook 10)

**What:** Freeze a pretrained EfficientNet-B0 encoder, train only a lightweight decoder for style transfer at 384×384.

**Key results:** FAILED
- Same instability as NAFNet despite frozen encoder
- Loss noisy (0.20, didn't converge cleanly)
- Some images acceptable, others show artifacts and blur

**Lesson:** Even with a frozen encoder, 270 images and batch_size=2 at 384×384 isn't enough for stable decoder training.

---

## Phase 11: LoRA Fine-tuning of InstructPix2Pix (Notebook 11) ← CURRENT BEST

**What:** Fine-tune InstructPix2Pix (pretrained on 450K+ edit pairs) using LoRA (r=4, only cross-attention layers). Conservative training: lr=1e-5, 15 epochs, early stopping at best epoch.

**Key results:** SUCCESS
- Sharp, artifact-free outputs at 512×512
- Color/tone adjustments are subtle and appropriate
- Image fidelity perfectly preserved
- Best epoch: 11 (loss 0.102)
- Text conditioning built into the base model

**Why it works:** The pretrained model already knows how to process images without artifacts. LoRA only teaches it my specific style preferences — a much easier learning problem that 270 images can handle.

**Key hyperparameters:**
- LoRA rank: 4 (r=16 caused overfitting/hallucination)
- Learning rate: 1e-5 (1e-4 was too aggressive)
- Target modules: to_q, to_v only (all attention layers caused overfitting)
- image_guidance_scale: 2.0 at inference (1.5 was too loose)
- guidance_scale: 5.0 at inference (7.0 was too strong)

---

## Current Status (March 2026)

**Best model:** InstructPix2Pix + LoRA (Notebook 11)
**Data:** 270 paired edits + 30 Fuji reference + 211 ratings
**Quality:** Artifact-free, good fidelity, style direction correct but not yet fully matching personal aesthetic

**Next priorities:**
1. Iterate LoRA training — tune hyperparameters, possibly increase rank slightly
2. More clean training data (target: 500+ pairs without Photoshop edits)
3. Text-guided style variations using the fine-tuned model
4. Rate LoRA outputs and retrain reward model
5. Build simple inference tool / web UI
6. Distill to lightweight model for fast inference

---

## File Locations (Google Drive)

```
photo-style-rl/
├── images/
│   ├── raw/          # 270 unedited JPGs (raw_01.jpg - raw_270.jpg)
│   ├── edited/       # 270 Lightroom-edited JPGs (edited_01.jpg - edited_270.jpg)
│   └── fuji_reference/  # 30 Fuji X100VI JPGs (fuji_edited_01.jpg - fuji_edited_30.jpg)
├── ratings/
│   ├── ratings_unet_v1.json      # 211 ratings with text feedback
│   └── feedback_summary.json     # Text feedback only
├── checkpoints/
│   ├── clip_embeddings_raw.pkl   # CLIP embeddings for raw images
│   ├── clip_embeddings_edited.pkl
│   ├── lut_predictor_v3.pt       # Best LUT model
│   ├── unet_v1.pt                # U-Net trained on pairs
│   ├── unet_v2_fuji.pt           # U-Net fine-tuned with Fuji reference
│   ├── text_unet_v1.pt           # Text-conditioned U-Net
│   ├── reward_model_v2.pt        # Reward model from 211 ratings
│   ├── lora_style_v1/            # LoRA weights for InstructPix2Pix
│   └── fuji_style_profile.pt     # Fuji color/tone statistics
└── notebooks/                    # Working copies of Colab notebooks
```
