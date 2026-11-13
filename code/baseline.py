# -*- coding: utf-8 -*-
"""baseline.py

Random‑noise baseline experiment.

* Loads the first 20 clean images.
* For each L‑inf magnitude (1/255, 2/255, 4/255, 8/255) adds uniform noise
  in ``[-mag, mag]`` to the image, clips to [0, 1], and queries the model.
* Logs:
  - total queries used
  - success rate (label flip fraction)
  - mean L2, mean L‑inf, mean PSNR, mean SSIM across the 20 images

Results are printed as a compact markdown table.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List

import numpy as np

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

# Project layout
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # blackbox-attack
CODE_DIR = PROJECT_ROOT / "code"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"

# Add code directory to import path
import sys
sys.path.append(str(CODE_DIR))

from blackbox import BlackBoxModel
from metrics import l2_norm, linf_norm, psnr, ssim


def load_image_paths(limit: int = 20) -> List[Path]:
    """Return a list of the first *limit* PNG image paths sorted alphabetically."""
    all_png = sorted([p for p in DATA_CLEAN.glob("*.png") if p.is_file()])
    return all_png[:limit]


def load_image(path: Path) -> np.ndarray:
    """Load PNG as float32 array in [0, 1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def add_uniform_noise(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Add uniform noise in ``[-magnitude, magnitude]`` to *img* and clip to [0,1]."""
    noise = np.random.uniform(-magnitude, magnitude, size=img.shape).astype(np.float32)
    noisy = np.clip(img + noise, 0.0, 1.0)
    return noisy


def main():
    # Initialise model wrapper
    model_path = PROJECT_ROOT / "models" / "cnn_model.pt"
    bb = BlackBoxModel(model_path)

    # Load clean images and their original predictions
    img_paths = load_image_paths(20)
    clean_imgs = [load_image(p) for p in img_paths]
    clean_labels = [bb.predict_label(img) for img in clean_imgs]
    # Store original query count (should be 20)
    base_queries = bb.global_query_count

    # Define magnitudes (L-inf) to test
    mags = [1/255, 2/255, 4/255, 8/255]

    # Header for table output
    print("| L-inf magnitude | Queries | Flip rate | Mean L2 | Mean Linf | Mean PSNR | Mean SSIM |")
    print("|---|---|---|---|---|---|---|")

    for mag in mags:
        flips = 0
        l2_vals = []
        linf_vals = []
        psnr_vals = []
        ssim_vals = []
        for img, clean_lbl in zip(clean_imgs, clean_labels):
            noisy = add_uniform_noise(img, mag)
            # Compute metrics against clean image
            l2_vals.append(l2_norm(img, noisy))
            linf_vals.append(linf_norm(img, noisy))
            psnr_vals.append(psnr(img, noisy))
            ssim_vals.append(ssim(img, noisy))
            # Query model
            pred = bb.predict_label(noisy)
            if pred != clean_lbl:
                flips += 1
        # After processing this magnitude, compute stats
        total_q = bb.global_query_count - base_queries  # queries for this mag
        avg_l2 = sum(l2_vals) / len(l2_vals)
        avg_linf = sum(linf_vals) / len(linf_vals)
        avg_psnr = sum(psnr_vals) / len(psnr_vals)
        avg_ssim = sum(ssim_vals) / len(ssim_vals)
        flip_rate = flips / len(clean_imgs)
        print(f"| {mag:.4f} | {total_q} | {flip_rate:.2%} | {avg_l2:.4f} | {avg_linf:.4f} | {avg_psnr:.2f} | {avg_ssim:.4f} |")

    # Final summary of total queries used
    print(f"\nTotal model queries performed: {bb.global_query_count}")

if __name__ == "__main__":
    main()
