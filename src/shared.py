"""
shared.py v2.1 — Common utilities for photo-style-rl pipeline.
Lives at: Google Drive > photo-style-rl/src/shared.py
Every notebook copies this to /content/ and imports it.

v2.1 fixes from v2.0:
- Type guards on apply_tone_curve / apply_hsl_mixer (handles list-shaped profile data)
- _dampen_params safely skips non-dict inputs
- HSL mixer attenuates orange/red shifts on detected skin pixels (fixes yellow skin bug)
- _current_skin_mask passed from render() into apply_hsl_mixer() via instance var

v2.0 changes from v1:
- Skin tone detection + attenuation on temperature/saturation/tint shifts
- sRGB linearization for exposure and white balance
- Pro tools: sharpening, grain, vignette, noise reduction
- Region validation (no sky edits when no sky detected)
- Global strength dampener for profile application
- Improved sky heuristic (rejects dark upper regions in night scenes)
"""

import os
import re
import cv2
import json
import copy
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from collections import defaultdict
import matplotlib.colors as mcolors

# =============================================================================
# Constants
# =============================================================================

PROJECT = '/content/drive/MyDrive/photo-style-rl'
RAW_DIR = f'{PROJECT}/images/raw'
EDITED_DIR = f'{PROJECT}/images/edited'
CHECKPOINTS_DIR = f'{PROJECT}/checkpoints'
DATA_DIR = f'{PROJECT}/data'

SCENE_TYPES = [
    "night_street", "daylight_landscape", "portrait", "golden_hour",
    "indoor", "overcast", "backlit", "blue_hour", "food", "architecture"
]

HUE_BINS = {
    'reds':     ((345, 360), (0, 15)),
    'oranges':  ((15, 45),),
    'yellows':  ((45, 75),),
    'greens':   ((75, 135),),
    'aquas':    ((135, 195),),
    'blues':    ((195, 255),),
    'purples':  ((255, 315),),
    'magentas': ((315, 345),),
}

TONE_BAND_CENTERS = {
    'blacks': 0.05,
    'shadows': 0.225,
    'midtones': 0.5,
    'highlights': 0.775,
    'whites': 0.95,
}

SEMANTIC_LABELS = [
    "sky", "face", "skin", "hair", "subject", "background", "ground",
    "foliage", "water", "building", "clothing", "shadow_area",
    "highlight_area", "unknown"
]

AVAILABLE_REGIONS = [
    "global", "subject", "background", "sky", "ground",
    "foliage", "water", "building", "clothing", "face"
]

# Schema validation constants for training data gating (NB06 generation + NB07 training).
# These must stay in sync with DeterministicRenderer.apply_basic_params() — if you add
# a new renderer parameter, add it here too or validate_payload() will reject it.
VALID_TONE_CURVE_KEYS = frozenset({"blacks", "shadows", "midtones", "highlights", "whites"})
VALID_COLOR_NAMES = frozenset({"reds", "oranges", "yellows", "greens", "aquas", "blues", "purples", "magentas"})
VALID_HSL_KEYS = frozenset({"h", "s", "l"})
VALID_BASIC_KEYS = frozenset({
    "exposure", "contrast", "temperature", "tint",
    "shadows", "highlights", "whites", "blacks",
    "clarity", "vibrance", "saturation", "texture", "dehaze",
})

# Skin-tone hue range in HSV degrees — covers light to dark skin
SKIN_HUE_RANGE = (8, 45)
SKIN_SAT_MIN = 0.10
SKIN_SAT_MAX = 0.75
SKIN_VAL_MIN = 0.15


# =============================================================================
# File Utilities
# =============================================================================

def get_number(filename):
    """Extract numeric ID from filenames like 'raw_042.jpg' or 'edited_042.jpg'."""
    name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    for prefix in ['raw_', 'edited_', 'raw', 'edited']:
        name = name.replace(prefix, '')
    return int(name)


def pair_files(raw_dir=RAW_DIR, edited_dir=EDITED_DIR):
    """Match raw and edited files by numeric ID."""
    raw_files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    edited_files = sorted([f for f in os.listdir(edited_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    raw_map = {get_number(f): f for f in raw_files}
    edited_map = {get_number(f): f for f in edited_files}
    common = sorted(set(raw_map.keys()) & set(edited_map.keys()))
    return [(raw_map[n], edited_map[n]) for n in common]


def load_style_profile(name='simon_master_profile.json'):
    """Load a style profile JSON from the checkpoints folder."""
    path = os.path.join(CHECKPOINTS_DIR, name)
    with open(path, 'r') as f:
        return json.load(f)


def save_style_profile(profile, name='simon_master_profile.json'):
    """Save a style profile JSON to the checkpoints folder."""
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINTS_DIR, name)
    with open(path, 'w') as f:
        json.dump(profile, f, indent=2)
    return path


# =============================================================================
# Image Encoding (for Claude API)
# =============================================================================

def image_to_base64(pil_img, max_size=512):
    """Encode a PIL image as base64 JPEG for the Claude API.
    Works on a copy so the original PIL image is never mutated
    (PIL's thumbnail() modifies in-place, which is a classic gotcha)."""
    img_copy = pil_img.copy()
    img_copy.thumbnail((max_size, max_size))
    buffer = BytesIO()
    img_copy.save(buffer, format='JPEG', quality=80)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def extract_json(text):
    """Extract JSON from an LLM response that may have markdown fences or commentary.
    Tries dict first, then array, returns None if nothing parses."""
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# Skin Tone Detection
# =============================================================================

def detect_skin_mask(img_rgb_float):
    """Detect skin-tone pixels using HSV hue/saturation/value ranges.
    Returns a soft [0, 1] mask — Gaussian-blurred so transitions are smooth
    instead of pixelated hard edges at skin boundaries."""
    hsv = mcolors.rgb_to_hsv(np.clip(img_rgb_float, 0, 1))
    hue_deg = hsv[:, :, 0] * 360.0
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    hue_mask = (hue_deg >= SKIN_HUE_RANGE[0]) & (hue_deg <= SKIN_HUE_RANGE[1])
    sat_mask = (sat >= SKIN_SAT_MIN) & (sat <= SKIN_SAT_MAX)
    val_mask = val >= SKIN_VAL_MIN

    skin_binary = (hue_mask & sat_mask & val_mask).astype(np.float32)
    skin_soft = cv2.GaussianBlur(skin_binary, (15, 15), 5)
    return skin_soft


# =============================================================================
# DeterministicRenderer v2.1
# =============================================================================

class DeterministicRenderer:
    """
    The ONLY renderer in the pipeline. Applies tone curves, 8-channel HSL mixer,
    basic Lightroom params, and pro tools via pure math. No diffusion, no artifacts.

    v2.1: skin protection now covers HSL mixer (orange/red channel attenuation),
    type guards prevent crashes on malformed profile data.
    """

    def __init__(self):
        self.hue_bins = HUE_BINS
        self.tone_centers = TONE_BAND_CENTERS
        self._current_skin_mask = None

    # --- sRGB linearization ---

    def _srgb_to_linear(self, img):
        """sRGB gamma to linear light. Needed for photometrically correct exposure/WB."""
        return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

    def _linear_to_srgb(self, img):
        """Linear light to sRGB gamma."""
        return np.where(
            img <= 0.0031308,
            img * 12.92,
            1.055 * np.power(np.clip(img, 0.0001, None), 1.0 / 2.4) - 0.055
        )

    # --- Tone curve ---

    def apply_tone_curve(self, img_hsv, curve_params):
        """Apply luminance shifts across 5 tonal bands using Gaussian falloff.
        Each band blends smoothly into neighbors so there are no hard jumps."""
        if not curve_params or not isinstance(curve_params, dict):
            return img_hsv

        v = img_hsv[:, :, 2]
        result_v = v.copy()

        for band, shift_pct in curve_params.items():
            if band not in self.tone_centers:
                continue
            if not isinstance(shift_pct, (int, float)) or shift_pct == 0:
                continue
            center = self.tone_centers[band]
            weight = np.exp(-((v - center) ** 2) / (2 * 0.15 ** 2))
            result_v += weight * (shift_pct / 100.0) * v * (1 - v)

        img_hsv[:, :, 2] = np.clip(result_v, 0, 1)
        return img_hsv

    # --- HSL mixer (with skin protection on reds/oranges) ---

    def apply_hsl_mixer(self, img_hsv, mixer_params):
        """Selectively shift hue, saturation, luminance per color bin.
        Orange and red shifts are attenuated on skin pixels to prevent
        yellow/unnatural skin tones."""
        if not mixer_params or not isinstance(mixer_params, dict):
            return img_hsv

        hue_deg = img_hsv[:, :, 0] * 360.0

        for color, shifts in mixer_params.items():
            if color not in self.hue_bins:
                continue
            if not isinstance(shifts, dict):
                continue
            if all(shifts.get(k, 0) == 0 for k in ('h', 's', 'l')):
                continue

            mask = np.zeros_like(hue_deg, dtype=bool)
            for (low, high) in self.hue_bins[color]:
                mask |= (hue_deg >= low) & (hue_deg < high)

            if not np.any(mask):
                continue

            smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)

            # Attenuate orange/red shifts on skin — this prevents the yellow skin bug.
            # Where skin is detected, reduce the color shift to 30% strength.
            if color in ('reds', 'oranges') and self._current_skin_mask is not None:
                smooth_mask = smooth_mask * (1.0 - self._current_skin_mask * 0.7)

            if shifts.get('h', 0) != 0:
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + (shifts['h'] / 360.0) * smooth_mask) % 1.0
            if shifts.get('s', 0) != 0:
                img_hsv[:, :, 1] = np.clip(
                    img_hsv[:, :, 1] * (1.0 + (shifts['s'] / 100.0) * smooth_mask), 0, 1
                )
            if shifts.get('l', 0) != 0:
                img_hsv[:, :, 2] = np.clip(
                    img_hsv[:, :, 2] * (1.0 + (shifts['l'] / 100.0) * smooth_mask), 0, 1
                )

        return img_hsv

    # --- Basic params with skin protection + linear light ---

    def apply_basic_params(self, img_rgb, params, skin_mask=None):
        """Apply Lightroom-style slider params. Exposure and WB happen in linear
        light. Temperature, tint, saturation, vibrance are attenuated on skin."""
        img = img_rgb.copy()

        needs_linear = any(params.get(k, 0) != 0 for k in ('exposure', 'temperature', 'tint'))
        if needs_linear:
            img = self._srgb_to_linear(img)

        if 'exposure' in params and params['exposure'] != 0:
            img = img * (2.0 ** params['exposure'])

        if 'temperature' in params and params['temperature'] != 0:
            t = params['temperature'] / 100.0
            if skin_mask is not None:
                attn = 1.0 - skin_mask * 0.7
                img[:, :, 0] *= (1.0 + t * attn * 0.1)
                img[:, :, 2] *= (1.0 - t * attn * 0.1)
            else:
                img[:, :, 0] *= (1.0 + t * 0.1)
                img[:, :, 2] *= (1.0 - t * 0.1)

        if 'tint' in params and params['tint'] != 0:
            tint = params['tint'] / 100.0
            if skin_mask is not None:
                img[:, :, 1] *= (1.0 - tint * (1.0 - skin_mask * 0.7) * 0.05)
            else:
                img[:, :, 1] *= (1.0 - tint * 0.05)

        if needs_linear:
            img = self._linear_to_srgb(np.clip(img, 0, None))

        if 'contrast' in params and params['contrast'] != 0:
            factor = 1.0 + params['contrast'] / 100.0
            img = (img - 0.5) * factor + 0.5

        lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        if 'shadows' in params and params['shadows'] != 0:
            s = params['shadows'] / 100.0
            img += np.clip(1.0 - lum * 2, 0, 1)[:, :, None] * s * 0.3

        if 'highlights' in params and params['highlights'] != 0:
            h = params['highlights'] / 100.0
            img += np.clip(lum * 2 - 1, 0, 1)[:, :, None] * h * 0.3

        if 'whites' in params and params['whites'] != 0:
            w = params['whites'] / 100.0
            img += np.clip((lum - 0.8) / 0.2, 0, 1)[:, :, None] * w * 0.2

        if 'blacks' in params and params['blacks'] != 0:
            b = params['blacks'] / 100.0
            img += np.clip((0.2 - lum) / 0.2, 0, 1)[:, :, None] * b * 0.2

        if 'clarity' in params and params['clarity'] != 0:
            c = params['clarity'] / 100.0
            gray = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=10).astype(np.float32) / 255.0
            detail = gray.astype(np.float32) / 255.0 - blurred
            img += detail[:, :, None] * c * 0.5

        if 'texture' in params and params['texture'] != 0:
            t = params['texture'] / 100.0
            gray = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2).astype(np.float32) / 255.0
            detail = gray.astype(np.float32) / 255.0 - blurred
            img += detail[:, :, None] * t * 0.3

        if 'dehaze' in params and params['dehaze'] != 0:
            d = params['dehaze'] / 100.0
            dark = np.min(img, axis=2)
            dark_blurred = cv2.GaussianBlur(dark, (0, 0), sigmaX=20)
            img = img + d * 0.3 * (img - dark_blurred[:, :, None])

        if 'vibrance' in params and params['vibrance'] != 0:
            v = params['vibrance'] / 100.0
            max_c = img.max(axis=2)
            min_c = img.min(axis=2)
            sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-7), 0)
            vibrance_weight = (1.0 - sat) * v
            if skin_mask is not None:
                vibrance_weight *= (1.0 - skin_mask * 0.5)
            gray_avg = img.mean(axis=2, keepdims=True)
            img += (img - gray_avg) * vibrance_weight[:, :, None] * 0.5

        if 'saturation' in params and params['saturation'] != 0:
            s = params['saturation'] / 100.0
            gray_avg = img.mean(axis=2, keepdims=True)
            if skin_mask is not None:
                s_per_pixel = s * (1.0 - skin_mask * 0.5)
                img = gray_avg + (img - gray_avg) * (1.0 + s_per_pixel[:, :, None])
            else:
                img = gray_avg + (img - gray_avg) * (1.0 + s)

        return np.clip(img, 0, 1)

    # --- Pro tools ---

    def apply_sharpening(self, img, amount=0, radius=1.0, detail=0, masking=0):
        """Unsharp mask sharpening. masking>0 uses edge detection to protect
        smooth areas (skin) from being sharpened."""
        if amount == 0:
            return img
        factor = amount / 100.0
        sigma = max(radius, 0.5)
        gray = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma).astype(np.float32)
        detail_layer = gray.astype(np.float32) - blurred
        if masking > 0:
            edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
            edge_mask = cv2.GaussianBlur(edges, (0, 0), sigmaX=2)
            blend = masking / 100.0
            detail_layer *= (blend * edge_mask + (1 - blend))
        img = img + (detail_layer / 255.0)[:, :, None] * factor * 2
        return np.clip(img, 0, 1)

    def apply_grain(self, img, amount=0, size=25, roughness=50):
        """Film grain. Generates noise at reduced resolution for coarser grain."""
        if amount == 0:
            return img
        h, w = img.shape[:2]
        intensity = amount / 100.0
        grain_scale = max(1, size // 10)
        grain_h = max(h // grain_scale, 1)
        grain_w = max(w // grain_scale, 1)
        grain = np.random.randn(grain_h, grain_w).astype(np.float32)
        if roughness < 80:
            grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=(100 - roughness) / 50.0)
        if grain_scale > 1:
            grain = cv2.resize(grain, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img + grain[:, :, None] * intensity * 0.12
        return np.clip(img, 0, 1)

    def apply_vignette(self, img, amount=0, midpoint=50, roundness=0, feather=50):
        """Radial vignette. Negative amount darkens edges."""
        if amount == 0:
            return img
        h, w = img.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        y = np.arange(h, dtype=np.float32) - cy
        x = np.arange(w, dtype=np.float32) - cx
        yy, xx = np.meshgrid(y, x, indexing='ij')
        aspect = 1.0 + roundness / 200.0
        dist = np.sqrt((xx / (cx * aspect + 1e-5)) ** 2 + (yy / (cy + 1e-5)) ** 2)
        inner_radius = midpoint / 100.0
        falloff = max(feather / 100.0 * 0.5, 0.01)
        vignette = np.clip((dist - inner_radius) / falloff, 0, 1) ** 1.5
        strength = amount / 100.0
        if strength < 0:
            img = img * (1.0 + strength * vignette[:, :, None])
        else:
            img = img + strength * vignette[:, :, None] * (1.0 - img)
        return np.clip(img, 0, 1)

    def apply_noise_reduction(self, img, luminance=0, color=0):
        """Bilateral filter for luminance noise, Lab chroma blur for color noise."""
        if luminance == 0 and color == 0:
            return img
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if luminance > 0:
            d = 5 + int(luminance / 20)
            img_uint8 = cv2.bilateralFilter(img_uint8, d, luminance * 0.75, luminance * 0.5)
        if color > 0:
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2Lab)
            sigma = color * 0.3
            lab[:, :, 1] = cv2.GaussianBlur(lab[:, :, 1], (0, 0), sigmaX=sigma)
            lab[:, :, 2] = cv2.GaussianBlur(lab[:, :, 2], (0, 0), sigmaX=sigma)
            img_uint8 = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        return img_uint8.astype(np.float32) / 255.0

    # --- Strength dampening ---

    def _dampen_params(self, params, strength):
        """Scale all numeric values by strength factor. Safely handles non-dict inputs."""
        if not isinstance(params, dict):
            return params
        dampened = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                dampened[key] = value * strength
            elif isinstance(value, dict):
                dampened[key] = self._dampen_params(value, strength)
            elif isinstance(value, list):
                dampened[key] = value
            else:
                dampened[key] = value
        return dampened

    # --- Unified render ---

    def render(self, img_rgb_float, params, protect_skin=True, strength=0.6):
        """
        Apply a complete parameter set to an image.
        strength=0.6 for auto-profiles (our math overshoots vs Lightroom).
        strength=1.0 for explicit text overrides.
        """
        if strength < 1.0:
            params = self._dampen_params(params, strength)

        result = img_rgb_float.copy()

        skin_mask = detect_skin_mask(result) if protect_skin else None
        self._current_skin_mask = skin_mask

        basic_keys = {
            'exposure', 'contrast', 'temperature', 'tint', 'shadows', 'highlights',
            'whites', 'blacks', 'clarity', 'vibrance', 'saturation', 'texture', 'dehaze'
        }
        basic_params = {k: v for k, v in params.items() if k in basic_keys and v != 0}
        if basic_params:
            result = self.apply_basic_params(result, basic_params, skin_mask)

        has_advanced = 'tone_curve' in params or 'color_mixer' in params
        if has_advanced:
            hsv = mcolors.rgb_to_hsv(np.clip(result, 0, 1))
            hsv = self.apply_hsl_mixer(hsv, params.get('color_mixer', {}))
            hsv = self.apply_tone_curve(hsv, params.get('tone_curve', {}))
            result = mcolors.hsv_to_rgb(hsv)

        result = np.clip(result, 0, 1)

        nr_lum = params.get('noise_reduction_luminance', 0)
        nr_color = params.get('color_noise_reduction', 0)
        if nr_lum > 0 or nr_color > 0:
            result = self.apply_noise_reduction(result, nr_lum, nr_color)

        sharp = params.get('sharpening_amount', params.get('sharpening', 0))
        if sharp > 0:
            result = self.apply_sharpening(
                result, amount=sharp,
                radius=params.get('sharpening_radius', 1.0),
                detail=params.get('sharpening_detail', 0),
                masking=params.get('sharpening_masking', 0),
            )

        grain = params.get('grain_amount', 0)
        if grain > 0:
            result = self.apply_grain(
                result, amount=grain,
                size=params.get('grain_size', 25),
                roughness=params.get('grain_roughness', 50),
            )

        vig = params.get('vignette_amount', 0)
        if vig != 0:
            result = self.apply_vignette(
                result, amount=vig,
                midpoint=params.get('vignette_midpoint', 50),
                roundness=params.get('vignette_roundness', 0),
                feather=params.get('vignette_feather', 50),
            )

        self._current_skin_mask = None
        return np.clip(result, 0, 1)


# =============================================================================
# SAM Segmentation + Labeling
# =============================================================================

def get_mask_properties(mask, image_np):
    """Extract spatial + color properties of a boolean mask for classification."""
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return {
        'bbox': (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        'center_y': float(ys.mean() / h),
        'center_x': float(xs.mean() / w),
        'area_ratio': float(mask.sum() / (h * w)),
        'mean_rgb': image_np[mask].mean(axis=0).tolist(),
        'mean_brightness': float(image_np[mask].mean(axis=0).mean() / 255.0),
        'top_half_ratio': float((ys < h / 2).sum() / len(ys)),
    }


def classify_mask_heuristic(props):
    """Rule-based classification. Brightness threshold 0.55 prevents dark
    upper regions in night scenes from being misidentified as sky."""
    if props is None:
        return 'unknown', 0.0
    r, g, b = props['mean_rgb']
    if (props['top_half_ratio'] > 0.8 and props['area_ratio'] > 0.05
            and props['mean_brightness'] > 0.55
            and props['mean_brightness'] < 0.95
            and b > r * 0.8):
        return 'sky', 0.7 + 0.3 * props['top_half_ratio']
    if (props['center_y'] > 0.6 and props['area_ratio'] > 0.1
            and props['top_half_ratio'] < 0.3):
        return 'ground', 0.6
    if (props['area_ratio'] < 0.15 and 0.2 < props['center_x'] < 0.8
            and 0.1 < props['center_y'] < 0.7):
        return 'subject', 0.4
    if props['area_ratio'] > 0.5:
        return 'background', 0.5
    return 'unknown', 0.0


def classify_masks_with_llm(pil_img, masks_with_props, client):
    """Use Claude to label ambiguous masks. Explicitly tells Claude that dark
    areas at top of night scenes are NOT sky."""
    img_b64 = image_to_base64(pil_img, max_size=768)
    descriptions = []
    for i, (mask, props, h_label, conf) in enumerate(masks_with_props):
        if props is None:
            continue
        descriptions.append(
            f"Mask {i}: center=({props['center_x']:.2f}, {props['center_y']:.2f}), "
            f"area={props['area_ratio']:.1%}, brightness={props['mean_brightness']:.2f}, "
            f"heuristic={h_label} (conf={conf:.2f})"
        )
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=f"""You are labeling image regions for a photo editor.
Assign each mask a label from: {', '.join(SEMANTIC_LABELS)}
IMPORTANT: Only label a region as 'sky' if it is actual sky (blue gradient, clouds, sunset).
Dark areas at the top of night scenes are NOT sky — label them 'background' or 'shadow_area'.
Respond with ONLY valid JSON: [{{"mask_id": 0, "label": "sky"}}, ...]""",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": "Label these regions:\n" + "\n".join(descriptions)},
                ]
            }],
        )
        labels = extract_json(response.content[0].text)
        if labels and isinstance(labels, list):
            return {item['mask_id']: item['label'] for item in labels}
    except Exception as e:
        print(f"  LLM labeling failed: {e}")
    return {}


def segment_and_label(pil_img, auto_generator, client=None, cache=None, image_id=None, top_k=8):
    """Full segmentation: SAM masks -> heuristic labels -> Claude fallback for ambiguous."""
    image_np = np.array(pil_img)
    raw_masks = auto_generator.generate(image_np)
    raw_masks = sorted(raw_masks, key=lambda x: x['area'], reverse=True)[:top_k]

    masks_with_props = []
    for m in raw_masks:
        mask = m['segmentation']
        props = get_mask_properties(mask, image_np)
        h_label, h_conf = classify_mask_heuristic(props)
        masks_with_props.append((mask, props, h_label, h_conf))

    llm_labels = {}
    ambiguous = [i for i, m in enumerate(masks_with_props) if m[3] < 0.6]
    if ambiguous and client is not None:
        if cache is not None and image_id and image_id in cache:
            llm_labels = {int(k): v for k, v in cache[image_id].items()}
        else:
            llm_labels = classify_masks_with_llm(pil_img, masks_with_props, client)
            if cache is not None and image_id:
                cache[image_id] = llm_labels

    results = []
    for i, (mask, props, h_label, h_conf) in enumerate(masks_with_props):
        final_label = llm_labels.get(i, h_label)
        results.append({
            'mask': mask, 'label': final_label,
            'properties': props, 'area': int(mask.sum()),
        })
    return results


def resize_for_sam(pil_img, max_dim=1024):
    """Resize preserving aspect ratio. 1024 is SAM's sweet spot."""
    w, h = pil_img.size
    if max(w, h) <= max_dim:
        return pil_img.copy()
    scale = max_dim / max(w, h)
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# =============================================================================
# Regional Editing
# =============================================================================

def feather_mask(mask, radius=15):
    """Gaussian-blur a boolean mask so edits blend smoothly at boundaries."""
    ksize = radius * 2 + 1
    return cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigmaX=radius / 2)


def validate_region_edit(region_label, segments, img_rgb_float=None):
    """Check that a region actually exists before applying edits to it."""
    labels_present = set(s['label'] for s in segments)
    if region_label == 'global':
        return True
    if region_label == 'sky' and 'sky' not in labels_present:
        return False
    if region_label == 'face' and 'face' not in labels_present:
        return False
    return region_label in labels_present


def apply_regional_edits(pil_img, segments, regional_edits, renderer, feather_radius=15, strength=1.0):
    """Apply different params to different masked regions with feathered blending.

    strength: passed through to renderer.render(). Use 0.6 for extracted style profiles
              (compensates for renderer math overshooting Lightroom values), 1.0 for
              explicit text overrides where Claude chose the values intentionally.
    """
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]
    result = img_np.copy()

    label_to_mask = {}
    for seg in segments:
        label = seg['label']
        mask = seg['mask']
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST)
            ).astype(bool)
        if label not in label_to_mask:
            label_to_mask[label] = mask
        else:
            label_to_mask[label] = label_to_mask[label] | mask

    skin_regions = {'face', 'skin', 'subject'}

    for edit in regional_edits:
        region = edit.get('region', 'global')
        params = edit.get('params', edit.get('parameters', {}))
        if not params:
            params = {k: v for k, v in edit.items() if k != 'region'}

        if not validate_region_edit(region, segments, result):
            print(f"  Skipped '{region}': not detected in this image.")
            continue

        if region == 'global':
            result = renderer.render(result, params, protect_skin=True, strength=strength)
            continue

        if region not in label_to_mask:
            continue

        protect = region in skin_regions
        edited = renderer.render(result.copy(), params, protect_skin=protect, strength=strength)
        mask_f = feather_mask(label_to_mask[region], feather_radius)[:, :, None]
        result = edited * mask_f + result * (1.0 - mask_f)

    return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))


# =============================================================================
# Profile merging
# =============================================================================

def merge_params(base, override):
    """Deep-merge override params onto base profile."""
    merged = copy.deepcopy(base)
    for key in ('tone_curve', 'color_mixer'):
        if key in override:
            if key not in merged:
                merged[key] = {}
            if key == 'color_mixer':
                for color, shifts in override[key].items():
                    if color not in merged[key]:
                        merged[key][color] = {}
                    merged[key][color].update(shifts)
            else:
                merged[key].update(override[key])
    for key in ('exposure', 'contrast', 'temperature', 'tint', 'shadows', 'highlights',
                'whites', 'blacks', 'clarity', 'vibrance', 'saturation', 'texture', 'dehaze',
                'sharpening_amount', 'sharpening_radius', 'sharpening_detail', 'sharpening_masking',
                'grain_amount', 'grain_size', 'grain_roughness',
                'vignette_amount', 'vignette_midpoint', 'vignette_roundness', 'vignette_feather',
                'noise_reduction_luminance', 'color_noise_reduction'):
        if key in override:
            merged[key] = override[key]
    return merged


# =============================================================================
# Training Data Validation
# =============================================================================

def validate_payload(payload):
    """Strict schema validation for synthetic training records.
    Rejects any payload that contains hallucinated keys or out-of-range values
    so bad data never reaches the Qwen fine-tuning run.

    Expected format (flat, region-keyed):
        {"global": {"exposure": 1.5, "tone_curve": {"shadows": 20}},
         "sky":    {"highlights": -30, "color_mixer": {"blues": {"h": -10}}}}

    Returns True if the payload is clean, False otherwise.
    """
    if not isinstance(payload, dict):
        return False

    for region, edits in payload.items():
        if not isinstance(edits, dict):
            return False

        for key, value in edits.items():
            if key == 'tone_curve':
                if not isinstance(value, dict):
                    return False
                for k, v in value.items():
                    if k not in VALID_TONE_CURVE_KEYS:
                        return False
                    if not isinstance(v, (int, float)) or not (-100 <= v <= 100):
                        return False

            elif key == 'color_mixer':
                if not isinstance(value, dict):
                    return False
                for color, shifts in value.items():
                    if color not in VALID_COLOR_NAMES:
                        return False
                    if not isinstance(shifts, dict):
                        return False
                    for axis, val in shifts.items():
                        if axis not in VALID_HSL_KEYS:
                            return False
                        if not isinstance(val, (int, float)):
                            return False
                        limit = 180 if axis == 'h' else 100
                        if not (-limit <= val <= limit):
                            return False

            elif key in VALID_BASIC_KEYS:
                if not isinstance(value, (int, float)):
                    return False
                limit = 5 if key == 'exposure' else 100
                if not (-limit <= value <= limit):
                    return False

            else:
                return False  # Unknown key — reject to prevent schema drift

    return True


# =============================================================================
# Visualization
# =============================================================================

def visualize_segments(pil_img, segments, max_show=8):
    """Display image with colored mask overlays for debugging segmentation."""
    import matplotlib.pyplot as plt
    image_np = np.array(pil_img)
    n = min(len(segments), max_show)
    cols = min(n + 1, 5)
    rows = max((n + cols) // cols, 1)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    axes[0].imshow(image_np)
    axes[0].set_title('Original')
    axes[0].axis('off')
    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 1)))
    h, w = image_np.shape[:2]
    for i, seg in enumerate(segments[:n]):
        mask = seg['mask']
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST)
            ).astype(bool)
        color = (np.array(colors[i][:3]) * 255).astype(np.uint8)
        vis = image_np.copy().astype(float)
        vis[mask] = vis[mask] * 0.4 + color * 0.6
        axes[i + 1].imshow(vis.astype(np.uint8))
        area = seg.get('properties', {}).get('area_ratio', 0)
        axes[i + 1].set_title(f"{seg['label']} ({area:.0%})")
        axes[i + 1].axis('off')
    for j in range(n + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


def show_before_after(original, edited, title=""):
    """Side-by-side comparison plot."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(edited)
    ax2.set_title(title or 'Edited')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()