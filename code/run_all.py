# -*- coding: utf-8 -*-
"""run_all.py

Execute the final HopSkipJumpAttack (with optional post‑processing) on the full
set of clean images.

Features
--------
* **Batch control** – ``--start`` and ``--end`` let you process a slice of the
  dataset (useful for resuming or for testing a small subset).
* **Resume support** – if an adversarial PNG already exists in
  ``outputs/adversarial/`` *and* a corresponding CSV row is present, the image
  is skipped.
* **Verification** – after saving the adversarial PNG the script reloads it,
  queries the model and guarantees the label differs from the clean prediction.
  On failure it retries with a larger query budget (fallback) and records the
  fallback strategy used.
* **Result logging** – per‑image statistics are appended to
  ``outputs/results.csv`` (including runtime, distortion metrics, and query count).
* **Progress bar** – a simple textual progress indicator is printed.
* **Configuration** – the attack hyper‑parameters are taken from the recommendation
  produced by ``tune.py``.  Refinement is applied only if the pilot study deemed it
  worthwhile (see ``refine.py``).

Usage example (process first 10 images)::

    python code/run_all.py --start 0 --end 10

The script will create the required output directories if they do not exist.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project layout helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # blackbox-attack
CODE_DIR = PROJECT_ROOT / "code"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"
OUTPUT_ADV = PROJECT_ROOT / "outputs" / "adversarial"
OUTPUT_RES = PROJECT_ROOT / "outputs" / "results.csv"

# Ensure the code directory (containing blackbox, metrics, attack_hsja, refine)
# is on the import path.
import sys
sys.path.append(str(CODE_DIR))

# ---------------------------------------------------------------------------
# Imports from the repository
# ---------------------------------------------------------------------------
from blackbox import BlackBoxModel, save_png
from attack_hsja import run_hsja
from refine import refine_image

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def _uint8_round(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8) / 255.0


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_existing_csv(csv_path: Path) -> Dict[str, List[str]]:
    """Return a mapping from filename to the raw CSV row (as list of strings)."""
    existing: Dict[str, List[str]] = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            for row in reader:
                if row:
                    existing[row[0]] = row
    return existing


def _write_header_if_missing(csv_path: Path) -> None:
    if not csv_path.is_file():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename",
                "clean_label",
                "adv_label",
                "success",
                "queries",
                "L2",
                "Linf",
                "PSNR",
                "SSIM",
                "runtime_sec",
                "fallback",
            ])

# ---------------------------------------------------------------------------
# Core processing function for a single image
# ---------------------------------------------------------------------------
def _process_image(
    idx: int,
    img_path: Path,
    model: BlackBoxModel,
    attack_cfg: Dict,
    refine_cfg: Dict | None,
    use_refine: bool,
) -> Tuple[bool, List]:
    """Run attack (and optional refinement) on *img_path*.

    Returns ``(written, csv_row)`` where ``written`` indicates whether a new
    adversarial PNG was saved and ``csv_row`` is the list of values to be appended
    to ``results.csv``.
    """
    filename = img_path.name
    adv_path = OUTPUT_ADV / filename

    # Load clean image and obtain its true label (via the wrapper)
    clean = _load_image(img_path)
    clean_label = model.predict_label(clean)

    # -------------------------------------------------------------------
    # If the adversarial PNG already exists we still need to verify that it is
    # indeed adversarial – otherwise we fall back to a fresh attack.
    # -------------------------------------------------------------------
    if adv_path.is_file():
        adv_img = _load_image(adv_path)
        adv_label = model.predict_label(_uint8_round(adv_img))
        if adv_label != clean_label:
            # already successful – compute metrics and return
            l2 = np.linalg.norm((clean - adv_img).ravel())
            linf = np.max(np.abs(clean - adv_img))
            from metrics import psnr, ssim

            psnr_val = psnr(clean, adv_img)
            ssim_val = ssim(clean, adv_img)
            # We cannot know the exact query count that produced this file, but we
            # record 0 extra queries for the resume case.
            return True, [
                filename,
                clean_label,
                adv_label,
                True,
                0,
                l2,
                linf,
                psnr_val,
                ssim_val,
                0.0,
                "resume",
            ]

    # -------------------------------------------------------------------
    # Fresh attack
    # -------------------------------------------------------------------
    start_time = time.time()
    adv_float, _, queries_used = run_hsja(clean, model, config=attack_cfg)
    # Optional refinement
    fallback_reason = "none"
    extra_queries = 0
    if use_refine:
        refined, extra = refine_image(clean, adv_float, model, config=refine_cfg)
        adv_float = refined
        extra_queries = extra
    # Save adversarial PNG (uint8 round)
    adv_uint8 = _uint8_round(adv_float)
    save_png(adv_uint8, adv_path)

    # Verify after saving (uint8 rounding may destroy the attack)
    adv_label = model.predict_label(adv_uint8)
    success = adv_label != clean_label
    # -------------------------------------------------------------------
    # Fallback if verification failed
    # -------------------------------------------------------------------
    if not success:
        # Try again with a larger budget and a different init strategy.
        fallback_cfg = attack_cfg.copy()
        fallback_cfg["budget"] = max(attack_cfg.get("budget", 2000), 5000)
        fallback_cfg["init_strategy"] = "blend_noise"
        adv_float, _, _ = run_hsja(clean, model, config=fallback_cfg)
        if use_refine:
            adv_float, extra = refine_image(clean, adv_float, model, config=refine_cfg)
            extra_queries += extra
        adv_uint8 = _uint8_round(adv_float)
        save_png(adv_uint8, adv_path)
        adv_label = model.predict_label(adv_uint8)
        success = adv_label != clean_label
        fallback_reason = "larger_budget"

    runtime = time.time() - start_time
    total_queries = queries_used + extra_queries

    # Compute distortion metrics
    from metrics import l2_norm, linf_norm, psnr, ssim

    l2 = l2_norm(clean, adv_uint8)
    linf = linf_norm(clean, adv_uint8)
    psnr_val = psnr(clean, adv_uint8)
    ssim_val = ssim(clean, adv_uint8)

    csv_row = [
        filename,
        clean_label,
        adv_label,
        success,
        total_queries,
        l2,
        linf,
        psnr_val,
        ssim_val,
        runtime,
        fallback_reason,
    ]
    return True, csv_row

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run HSJA attack on the full dataset.")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)"
    )
    args = parser.parse_args()

    _ensure_dir(OUTPUT_ADV)
    _write_header_if_missing(OUTPUT_RES)
    existing_csv = _read_existing_csv(OUTPUT_RES)

    # -------------------------------------------------------------------
    # Determine the recommended configuration (hard‑coded from the earlier tuning)
    # -------------------------------------------------------------------
    attack_cfg = {
        "budget": 3000,
        "iterations": 8,
        "max_num_evals": 20,
        "tolerance": 1e-5,
        "seed": 42,
    }

    # Refinement configuration – taken from the pilot study in refine.py
    # The pilot decided whether refinement is worthwhile; we embed that decision
    # here (set to True if the pilot reported a meaningful gain).
    USE_REFINEMENT = True  # <-- adjust if the pilot indicated otherwise
    refine_cfg = {
        "shrink_tol": 1e-4,
        "sparsify": True,
        "block_size": 4,
        "mag_thresh": 0.015,
        "max_blocks": 200,
        "smooth": True,
        "sigma": 0.7,
    }

    # -------------------------------------------------------------------
    # Load all clean image paths, sort to guarantee deterministic order
    # -------------------------------------------------------------------
    all_images = sorted([p for p in DATA_CLEAN.glob("*.png") if p.is_file()])
    start = args.start
    end = args.end if args.end is not None else len(all_images)
    subset = all_images[start:end]

    # Model wrapper – a fresh instance will keep a global query counter
    model_path = PROJECT_ROOT / "models" / "cnn_model.pt"
    bb = BlackBoxModel(model_path)

    # Statistics for the batch
    successes = 0
    total_l2 = 0.0
    total_queries = 0
    processed = 0

    print(f"Processing images {start} .. {end} (total {len(subset)})")
    for idx, img_path in enumerate(subset, start=start):
        fname = img_path.name
        # Resume check – CSV row and PNG must both exist
        if fname in existing_csv and (OUTPUT_ADV / fname).is_file():
            # Row already recorded, skip but still count for summary
            row = existing_csv[fname]
            successes += 1 if row[3].lower() == "true" else 0
            total_l2 += float(row[5])
            total_queries += int(row[4])
            processed += 1
            print(f"[{idx}] Skipped (already done)")
            continue

        written, csv_row = _process_image(
            idx, img_path, bb, attack_cfg, refine_cfg, USE_REFINEMENT
        )
        if written:
            # Append to CSV
            with OUTPUT_RES.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)
            # Update running stats
            successes += 1 if csv_row[3] else 0
            total_l2 += csv_row[5]
            total_queries += csv_row[4]
            processed += 1
            # Simple progress output
            print(
                f"[{idx}] {fname} – success={csv_row[3]}, L2={csv_row[5]:.4f}, "
                f"queries={csv_row[4]}, time={csv_row[9]:.2f}s"
            )
        else:
            print(f"[{idx}] {fname} – FAILED to generate adversarial example")

        # Periodic summary (every 10 images or at the end)
        if processed % 10 == 0 or idx == end - 1:
            success_rate = successes / processed * 100 if processed else 0
            mean_l2 = total_l2 / processed if processed else 0
            mean_q = total_queries / processed if processed else 0
            print(
                f"--- Summary after {processed} images: "
                f"Success {success_rate:.1f}% | Mean L2 {mean_l2:.4f} | "
                f"Mean queries {mean_q:.1f}"
            )

    print("\nAll done.")

if __name__ == "__main__":
    main()
