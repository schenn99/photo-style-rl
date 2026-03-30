# photo-style-rl

A hybrid AI photo editor that learns individual visual preferences and applies them via text-conditioned mathematical pixel transforms. No diffusion artifacts — every edit is a deterministic transform on original pixels.

Read `NOTES.md` for full development history and architecture decisions.

## Project Structure
```
photo-style-rl/
├── src/
│   └── shared.py              # THE shared module — all notebooks import from here
├── notebooks/
│   └── pipeline/
│       ├── 01_text_to_params.ipynb        # Claude API: text → Lightroom JSON params
│       ├── 02_style_profiling_and_region_detection.ipynb  # SAM + style extraction
│       ├── 03_color_extraction.ipynb      # GPU tone curve + HSL mixer extraction
│       ├── 04_inference_demo.ipynb        # End-to-end demo on own dataset
│       ├── 05_interactive_ui.ipynb        # Gradio UI + data flywheel
│       ├── 06_synthetic_data_distillation.ipynb  # Generate training data via Claude
│       └── 07_qwen_distillation.ipynb     # QLoRA fine-tune Qwen 2.5 1.5B
├── NOTES.md                   # Development log and architecture decisions
└── CLAUDE.md                  # This file
```

## Key Architecture Rules

- **One renderer:** `DeterministicRenderer` in `shared.py` is the ONLY renderer. Never duplicate rendering logic in notebooks. If a notebook defines its own image processing function, that's a bug.
- **One parameter schema:** tone_curve (5 bands: blacks/shadows/midtones/highlights/whites), color_mixer (8 HSL channels: reds/oranges/yellows/greens/aquas/blues/purples/magentas), basic params (exposure, contrast, temperature, tint, shadows, highlights, whites, blacks, clarity, vibrance, saturation, texture, dehaze), pro tools (sharpening, grain, vignette, noise reduction).
- **Skin protection:** Enabled by default on face/subject/skin regions. Temperature, tint, saturation shifts are attenuated 50-70% on detected skin pixels via `detect_skin_mask()`.
- **Region validation:** Never apply sky edits when no sky is detected. Never apply face edits when no face is detected. Use `validate_region_edit()` before every regional application.
- **Strength dampening:** Auto-style profiles use `strength=0.6` (our renderer math overshoots vs Lightroom's internal curves). Explicit text overrides use `strength=1.0`.
- **sRGB linearization:** Exposure and white balance operations are applied in linear light space via `_srgb_to_linear()` / `_linear_to_srgb()`. All other operations stay in gamma-encoded sRGB.

## Development Environment

- **Execution:** Google Colab (notebooks run in browser, not locally). GPU runtime required for SAM and kornia cells.
- **Storage:** Google Drive (`photo-style-rl/` folder) for images, checkpoints, cached labels, training data.
- **Local repo:** This directory is the git working tree. Workflow: edit in Colab → download .ipynb → commit locally.
- **GPU tiers:** T4 (free) for SAM inference and style extraction. A100/H100 for Qwen fine-tuning (NB07).
- **API:** Claude Sonnet 4 via Anthropic API. Key stored in Colab Secrets as `ANTHROPIC_API_KEY`.

## Data

- 287 paired images (raw/unedited + Lightroom-edited) that represent Simon's aesthetic in photography (Sony A7III)
- 30 Fuji X100VI reference photos (edited only, defines target aesthetic)
- Style profiles extracted per-region (SAM masks) and per-scene type (10 categories)
- SAM mask labels cached in `checkpoints/mask_labels_cache.json` to avoid redundant API calls
- Synthetic training data for Qwen distillation in `data/slm_training_dataset.jsonl`

## .gitignore

The `.gitignore` is already configured to keep the repo lightweight. Key exclusions:
- **Model weights:** `*.pt`, `*.pth`, `*.onnx`, `*.bin`, `*.safetensors` — SAM checkpoint, Qwen GGUF, reward models all live on Google Drive only
- **Image data:** `data/sample_images/*.jpg|png|heic` and `data/ratings/*.json` — the 287 paired images and rating data stay on Drive
- **Python/Jupyter caches:** `__pycache__/`, `.ipynb_checkpoints/`, `*.pyc`
- **Secrets:** `.env`, `*.key` — API keys go in Colab Secrets, never in the repo
- **IDE configs:** `.vscode/`, `.cursor/`

Note: The notebooks themselves ARE committed (they're the core deliverable). The `src/shared.py` module is also committed. Everything else that's large or sensitive is excluded.

If adding new data directories (e.g., `models/`, `checkpoints/`), make sure they have corresponding gitignore entries before committing.

## Photography Context

Simon's editing style leans toward Fujifilm colorways and vintage cinematic looks — warm shadows, compressed highlights, subtle grain, natural skin tones. Portraits and street photography can be more punchy and contrasty depending on context. The pipeline should preserve this aesthetic sensibility: no sickly yellows on skin, no blown-out neon colors, no AI-looking artifacts. 

## Code Review Standards

When reviewing this project, act as a senior ML engineer mentoring a junior engineer at a top-tier tech company. Be thorough and direct:

- **Catch everything.** Flag real bugs, logic errors, schema mismatches between notebooks, dead code, and functions that should be in `shared.py` instead of duplicated inline.
- **Discuss nitpicks too.** Variable naming, import ordering, unnecessary re-computation — mention them even if minor. Simon is building habits for production engineering.
- **Always explain why.** Don't just say "this is wrong" — explain the consequence. "This mutates the input image because PIL's `thumbnail()` operates in-place, so any downstream code using that image will get the resized version instead of the original."
- **Hold to a high standard.** This repo serves as a portfolio piece for PhD applications (CMU, UPenn, Georgia Tech) and job interviews at FAANG-tier companies. The code should reflect that.
- **Check cross-notebook consistency.** Every notebook should import from `shared.py`. If a notebook redefines `image_to_base64`, `extract_json`, `feather_mask`, or any rendering function locally, flag it as a refactoring target.

## Code Comment Style

Comments should be written for three audiences: future Simon reopening this in 6 months, a PhD advisor reviewing the repo, and a job interviewer scanning the code.

- **Natural tone, not robotic.** Write like a person explaining to a colleague, not like auto-generated documentation. "# Gaussian blur the mask edges so the edit doesn't have a hard visible seam" not "# Apply Gaussian blur to mask for feathering purposes."
- **Explain the _why_, not just the _what_.** "# We use medians instead of means here because segmentation edges introduce outlier pixels that would skew the averages" is useful. "# Calculate median" is not.
- **Inline explanations for non-obvious library calls.** If a line uses `kornia.image_to_tensor()`, `mcolors.rgb_to_hsv()`, `cv2.bilateralFilter()`, or similar, a brief comment explaining what it does and why we chose it is helpful. Example: "# kornia's rgb_to_hsv runs on GPU tensors — much faster than matplotlib's CPU version for batch processing"
- **No over-commenting.** Don't comment `img = img.copy()` with "# make a copy of the image". Only comment lines where the intent or mechanism isn't obvious from the code itself.

## Working With Simon

- **One terminal command at a time.** When giving shell commands, provide them individually — not as multi-line blocks with `#` comments between them. Zsh sometimes interprets pasted `#` characters as glob patterns and breaks.
- **Full code blocks for Colab.** Simon generally ports code manually into Google Colab notebooks. Give complete, copy-pasteable cells rather than diffs or partial snippets. Always specify which cell number or section the code belongs in.
- **Colab compatibility.** All code must run in Google Colab's environment. Use `!pip install` for dependencies, `drive.mount()` for storage, `userdata.get()` for secrets. Test imports at the top of each cell so errors surface immediately.
- **Be direct with feedback.** Simon wants honest critique, not encouragement padding. If something is architecturally wrong, say so plainly and explain the better approach.