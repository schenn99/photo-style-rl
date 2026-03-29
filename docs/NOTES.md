# Development Notes — photo-style-rl

A chronological log of development decisions, results, and lessons learned. Useful for context when prompting AI coding assistants (Cursor, Claude, etc.) or onboarding collaborators.

---

## Project Overview

**Goal:** Build a personal AI photo editor that learns individual visual preferences and applies them via text-conditioned pixel transforms. No diffusion artifacts — pure mathematical image processing guided by natural language.

**Data:**
- 287 paired images (raw + Lightroom-edited) from travel photography across Asia (Sony A7III)
- 30 Fuji X100VI reference photos (edited only, defines target aesthetic)
- 211 human ratings (1-7 scale) with 76 text feedback entries

**Infrastructure:**
- Development: Cursor IDE (local) + Google Colab (GPU compute)
- Storage: Google Drive (photo-style-rl/ folder) for images, checkpoints, ratings
- Version control: GitHub (schenn99/photo-style-rl)
- GPUs used: T4, L4, A100, H100 via Colab Pro

---

## Phase 1-12: Experimental Notebooks (in notebooks/experiments/)

### Key findings across 12 experimental phases:

| Phase | Approach | Result |
|-------|----------|--------|
| 01 | CLIP embeddings | Good scene understanding foundation |
| 02 | Global style parameters (7 sliders) | Too crude |
| 03 | RLHF rating system + reward model | Reward model works, RL unstable |
| 04 | 3D LUT prediction | Artifact-free but limited to global color |
| 05 | CLIP-conditioned U-Net | Best paired results at 256x256, blurry |
| 06 | Fuji X100VI style reference | Style profile extraction works |
| 07 | High-res detail refinement (512) | Marginal improvement |
| 08 | Text-conditioned U-Net | Text prompts work, style direction correct |
| 09 | NAFNet from scratch | Failed: insufficient data |
| 10 | Pretrained EfficientNet decoder | Failed: same data limitation |
| 11 | LoRA InstructPix2Pix | Sharp, artifact-free, style close but not perfect |
| 12 | FLUX.2 Klein DreamBooth LoRA | Overfitted, deep-fried outputs |

### Critical Lesson:
Diffusion models always risk introducing artifacts because they reconstruct pixels. For photo editing where preserving original detail is paramount, mathematical pixel transforms are superior. This led to the hybrid pipeline approach.

---

## Current Architecture: Hybrid Pipeline (in notebooks/pipeline/)

### Philosophy
Separate understanding what to do (LLM) from doing it (pixel math). No diffusion, no pixel reconstruction, no artifacts. Every flower petal, tooth, and text character is perfectly preserved.

### Pipeline 01: Text-to-Parameters (COMPLETE)
- LLM (Claude API) interprets natural language editing instructions
- Outputs structured JSON matching full Lightroom parameter schema
- NeuralImageProcessor applies parameters as differentiable pixel transforms
- Pair analysis: reverse-engineers style from raw-edited pairs

**Discovered Style Profile (from 287 pairs sample):**
- Temperature: +7.2 (subtle warmth)
- Shadows: +18.8 (lifted, film-like)
- Highlights: -13.3 (compressed, soft rolloff)
- Contrast: +9.3 (mild punch)
- Vibrance: +9.3 (smart saturation)
- Clarity: +8.2 (local contrast)
- Saturation: +5.0 (gentle)

### Pipeline 02: Style Profiling + Region Detection (IN PROGRESS)
- Analyze all 287 pairs to build comprehensive per-scene style profile
- SAM (Segment Anything) for face/sky/subject detection
- Enable surgical edits: brighten the right cheek, cool the sky
- Per-user style profiles: upload 10-20 photos, get personalized defaults

### Pipeline 03: Web UI (PLANNED)
- Upload photo, get auto-edit with user style profile
- Text refinement: iterate with natural language
- Parameter sliders for manual override
- Export at full resolution

### Pipeline 04: Local Model Distillation (PLANNED)
- Collect (prompt, parameters) pairs from Claude API usage
- Train small local model (Qwen 0.5B) to replace API calls
- Final product runs entirely offline

---

## Parameter Schema (Full Lightroom Coverage)

Global: exposure, contrast, highlights, shadows, whites, blacks, temperature, tint, texture, clarity, dehaze, vibrance, saturation

Detail: sharpening (amount/radius/detail/masking), noise reduction (luminance/detail/contrast), color noise reduction

Effects: grain (amount/size/roughness), vignette (amount/midpoint/roundness/feather/highlights)

Optics: defringe (purple/green amount and hue ranges), chromatic aberration removal

Tone Curve: parametric (4 regions) + RGB point curves (red/green/blue channels)

Color Mixer: 8 color channels x hue/saturation/luminance

Color Grading: shadows/midtones/highlights/global color wheels + blending/balance

Regional: linear gradient, radial gradient, brush (via region description), select subject, select sky, select face, luminance range mask, color range mask

---

## Product Vision

Core Experience: Upload a photo, get an auto-edit in your personal style, refine with text, export at full resolution.

Key Differentiator: Zero AI artifacts. Every edit is a mathematical transform on original pixels. Flowers stay sharp, hands stay normal, text stays readable. The AI understands what you want; the math does the work.

Future Features:
- Per-user style profiles (upload 10-20 reference edits)
- Creative exploration mode (optional diffusion pass for experimental looks)
- Batch editing (apply style to entire photo library)
- Mobile app with on-device inference (after model distillation)
