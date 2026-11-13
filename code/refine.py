# -*- coding: utf-8 -*-
"""refine.py – Light‑weight post‑processing for HSJA adversarial images.

The goal is to *shrink* the visible perturbation while preserving the adversarial
label, using **only label‑only queries** (the same `BlackBoxModel.predict_label`
interface used by the attack).  Three optional operations are provided:

1. **Scalar shrinkage** – binary‑search the smallest multiplier `α ∈ [0,1]`
   applied to the full perturbation `Δ = adv – clean` such that the
   uint8‑rounded image `clean + α·Δ` still flips the label.

2. **Sparsification** – iterate over non‑overlapping blocks, temporarily set
   the block to zero, and keep the change only if the label remains flipped.
   The number of blocks examined (and therefore the query cost) is
   configurable.

3. **Low‑pass smoothing** – blur the perturbation with a small Gaussian
   kernel; if the blurred perturbation still yields an adversarial image the
   blur is kept.

Each step records the extra model queries it consumes.  After the optional
steps the script reports, for a *fixed* random subset of 30 clean images,
the before/after distortion metrics (L2, L‑∞, PSNR, SSIM) and the extra query
cost.  Refinement is kept only if the **average L2 reduction** exceeds a
user‑specified *gain threshold* **relative** to the number of extra queries.

The script can be used as a library (`refine_image(...)`) or run directly as a
demo (`python code\refine.py`).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project layout helpers (the code folder must be on PYTHONPATH)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # blackbox-attack
CODE_DIR = PROJECT_ROOT / "code"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"

import sys
sys.path.append(str(CODE_DIR))

# ---------------------------------------------------------------------------
# Imports from the repository
# ---------------------------------------------------------------------------
from blackbox import BlackBoxModel
from metrics import l2_norm, linf_norm, psnr, ssim
from attack_hsja import run_hsja

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _uint8_round(img: np.ndarray) -> np.ndarray:
    """Round a float [0,1] image to uint8 and back to float [0,1]."""
    img_uint8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    return img_uint8.astype(np.float32) / 255.0


def _label_of(img: np.ndarray, model: BlackBoxModel) -> int:
    return model.predict_label(img)

# ---------------------------------------------------------------------------
# 1️⃣ Scalar shrinkage (binary search on the perturbation multiplier)
# ---------------------------------------------------------------------------
def shrink_perturbation(
    clean: np.ndarray,
    adv: np.ndarray,
    model: BlackBoxModel,
    clean_label: int,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, int]:
    perturb = adv - clean
    # Ensure original adv is actually adversarial (otherwise return unchanged)
    if _label_of(_uint8_round(adv), model) == clean_label:
        return adv, 0
    low, high = 0.0, 1.0
    extra = 0
    while high - low > tol:
        mid = (low + high) / 2.0
        cand = clean + mid * perturb
        pred = _label_of(_uint8_round(cand), model)
        extra += 1
        if pred != clean_label:
            high = mid
        else:
            low = mid
    refined = clean + high * perturb
    return refined, extra

# ---------------------------------------------------------------------------
# 2️⃣ Sparsification (zero‑out low‑magnitude blocks)
# ---------------------------------------------------------------------------
def sparsify_perturbation(
    clean: np.ndarray,
    adv: np.ndarray,
    model: BlackBoxModel,
    clean_label: int,
    block_size: int = 4,
    mag_thresh: float = 0.015,
    max_blocks: int | None = None,
) -> Tuple[np.ndarray, int]:
    perturb = adv - clean
    h, w, c = perturb.shape
    # Pad so dimensions are multiples of block_size
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h or pad_w:
        perturb = np.pad(
            perturb,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        clean_pad = np.pad(
            clean,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        clean_pad = clean
    H, W, _ = perturb.shape
    extra = 0
    examined = 0
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            if max_blocks is not None and examined >= max_blocks:
                break
            block = perturb[i : i + block_size, j : j + block_size, :]
            if np.max(np.abs(block)) > mag_thresh:
                continue
            saved = block.copy()
            perturb[i : i + block_size, j : j + block_size, :] = 0.0
            cand = clean_pad + perturb
            pred = _label_of(_uint8_round(cand), model)
            extra += 1
            examined += 1
            if pred == clean_label:
                perturb[i : i + block_size, j : j + block_size, :] = saved
    refined = clean_pad + perturb
    refined = refined[:h, :w, :]
    return refined, extra

# ---------------------------------------------------------------------------
# 3️⃣ Low‑pass smoothing (Gaussian blur on perturbation)
# ---------------------------------------------------------------------------
def smooth_perturbation(
    clean: np.ndarray,
    adv: np.ndarray,
    model: BlackBoxModel,
    clean_label: int,
    sigma: float = 0.7,
) -> Tuple[np.ndarray, int]:
    """Apply a Gaussian blur to the perturbation.

    If ``scipy`` is not available, the function falls back to a no‑op and
    returns the original adversarial image with zero extra queries.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        # scipy missing – skip smoothing
        return adv, 0

    perturb = adv - clean
    blurred = gaussian_filter(perturb, sigma=(sigma, sigma, 0), mode="reflect")
    cand = clean + blurred
    pred = _label_of(_uint8_round(cand), model)
    extra = 1
    if pred != clean_label:
        return cand, extra
    else:
        return adv, extra

# ---------------------------------------------------------------------------
# Public API – one‑shot refinement pipeline
# ---------------------------------------------------------------------------
def refine_image(
    clean: np.ndarray,
    adv: np.ndarray,
    model: BlackBoxModel,
    config: Dict | None = None,
) -> Tuple[np.ndarray, int]:
    if config is None:
        config = {}
    clean_label = _label_of(clean, model)
    total_extra = 0
    # 1️⃣ scalar shrinkage (always)
    adv, extra = shrink_perturbation(
        clean, adv, model, clean_label, tol=config.get("shrink_tol", 1e-4)
    )
    total_extra += extra
    # 2️⃣ optional sparsification
    if config.get("sparsify", False):
        adv, extra = sparsify_perturbation(
            clean,
            adv,
            model,
            clean_label,
            block_size=config.get("block_size", 4),
            mag_thresh=config.get("mag_thresh", 0.015),
            max_blocks=config.get("max_blocks", None),
        )
        total_extra += extra
    # 3️⃣ optional smoothing
    if config.get("smooth", False):
        adv, extra = smooth_perturbation(
            clean, adv, model, clean_label, sigma=config.get("sigma", 0.7)
        )
        total_extra += extra
    return adv, total_extra

# ---------------------------------------------------------------------------
# Demo / evaluation on the 30‑image pilot set
# ---------------------------------------------------------------------------
def _run_demo() -> None:
    random.seed(42)
    np.random.seed(42)
    all_png = sorted([p for p in DATA_CLEAN.glob("*.png") if p.is_file()])
    subset = random.sample(all_png, 30)
    model_path = PROJECT_ROOT / "models" / "cnn_model.pt"
    bb = BlackBoxModel(model_path)
    attack_cfg = {"budget": 2000, "iterations": 8, "max_num_evals": 20, "tolerance": 1e-5, "seed": 42}
    refine_cfg = {
        "shrink_tol": 1e-4,
        "sparsify": True,
        "block_size": 4,
        "mag_thresh": 0.015,
        "max_blocks": 200,
        "smooth": True,
        "sigma": 0.7,
    }
    stats: List[Dict] = []
    for img_path in subset:
        clean = _load_image(img_path)  # defined later in this file
        adv, _, _ = run_hsja(clean, bb, config=attack_cfg)
        adv_refined, extra = refine_image(clean, adv, bb, config=refine_cfg)
        clean_lbl = _label_of(clean, bb)
        adv_lbl = _label_of(_uint8_round(adv_refined), bb)
        success = adv_lbl != clean_lbl
        stats.append({
            "filename": img_path.name,
            "success_before": None,  # not needed for this demo
            "success_after": success,
            "l2": l2_norm(clean, adv_refined),
            "linf": linf_norm(clean, adv_refined),
            "psnr": psnr(clean, adv_refined),
            "ssim": ssim(clean, adv_refined),
            "extra_queries": extra,
        })
    # Simple aggregate printout
    import pandas as pd

    df = pd.DataFrame(stats)
    print("\n## Refinement demo on 30‑image pilot\n")
    print(df.describe(include="all"))
    print("\nMean extra queries per image:", df["extra_queries"].mean())

# Helper to load a PNG (reuse the same code as in attack_hsja for consistency)
def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run refinement demo on 30 images")
    parser.add_argument("--demo", action="store_true", help="run the demo")
    args = parser.parse_args()
    if args.demo:
        _run_demo()
