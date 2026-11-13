# -*- coding: utf-8 -*-
"""attack_hsja.py

A lightweight, NumPy‑only implementation of the HopSkipJumpAttack (HSJA)
for black‑box *label‑only* models.  It relies exclusively on
``BlackBoxModel.predict_label`` for decision feedback and does **not** use any
external adversarial‑attack libraries.

The algorithm follows the original HSJA paper (Madry et al., 2020) but is
simplified for clarity:

1. **Initialization** – draw a random noisy image (or blend the clean image with
   heavy noise) until a different label is obtained.
2. **Binary search** – project the point onto the decision boundary along the
   line connecting the clean and adversarial points.
3. **Gradient estimation** – Monte‑Carlo sampling of random unit directions on the
   sphere; the sign of the label (adversarial = +1, clean = ‑1) weights each
   direction.  The average yields an estimated gradient.
4. **Geometric step** – move a step orthogonal to the gradient toward the clean
   image; if the step leaves the adversarial region the step size is halved.
5. **Iterate** – repeat binary‑search → gradient‑estimation → geometric step for a
   configurable number of iterations or until a per‑image query budget is hit.
6. **Safety rounding** – after the loop the result is quantised to ``uint8``
   (the format used for the competition).  If rounding destroys the adversarial
   property we step a little farther away from the clean image.

The public entry point is ``run_hsja`` which accepts a NumPy image array and a
configuration dictionary.  It returns ``adv_img`` (float‑[0,1] image), a list of
L2 distances versus cumulative queries, and the final query count.

A minimal test harness at the bottom runs the attack on three clean images and
stores L2‑vs‑queries curves under ``reports/figures/hsja_curve_<idx>.png``.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

# Project paths – the script lives in ``code/``
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"
REPORTS_FIG = PROJECT_ROOT / "reports" / "figures"

# Make sure the code folder is on the import path for the BlackBox wrapper
sys.path.append(str(CODE_DIR))

from blackbox import BlackBoxModel

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load a PNG as a float32 ``[0, 1]`` array (H, W, C)."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.ravel() - b.ravel()))


def unit_sphere_directions(num: int, shape: Tuple[int, int, int]) -> np.ndarray:
    """Sample ``num`` random unit vectors with the same shape as an image.

    The vectors are drawn from a normal distribution and then normalised to
    unit L2 norm.
    """
    dirs = np.random.randn(num, *shape).astype(np.float32)
    # Compute L2 norm for each sample manually (sum over all spatial dimensions)
    # ``keepdims=True`` yields shape (num, 1, 1, 1) for broadcasting.
    norms = np.sqrt(np.sum(dirs ** 2, axis=(1, 2, 3), keepdims=True))
    # Avoid division by zero – extremely unlikely, but guard anyway.
    norms = np.where(norms == 0, 1.0, norms)
    dirs /= norms
    return dirs

# ---------------------------------------------------------------------------
# Core HopSkipJumpAttack implementation (label‑only)
# ---------------------------------------------------------------------------

def _binary_search_boundary(
    clean: np.ndarray,
    adv: np.ndarray,
    model: BlackBoxModel,
    clean_label: int,
    tolerance: float = 1e-5,
    max_iters: int = 20,
) -> np.ndarray:
    """Project ``adv`` onto the decision boundary between ``clean`` and ``adv``.

    A simple bisection along the line segment is performed until the L2 distance
    change is below ``tolerance`` or ``max_iters`` is reached.
    """
    low = 0.0
    high = 1.0
    for _ in range(max_iters):
        mid = (low + high) / 2.0
        candidate = (1 - mid) * clean + mid * adv
        pred = model.predict_label(candidate)
        if pred != clean_label:
            # still adversarial – move closer to clean side
            high = mid
        else:
            low = mid
        if high - low < tolerance:
            break
    # Return the point just inside the adversarial region (use high)
    return (1 - high) * clean + high * adv


def _estimate_gradient(
    x_boundary: np.ndarray,
    model: BlackBoxModel,
    clean_label: int,
    num_samples: int,
) -> np.ndarray:
    """Monte‑Carlo gradient estimation at ``x_boundary``.

    The method samples ``num_samples`` random directions ``u`` on the unit sphere
    and queries the model at ``x_boundary + epsilon * u`` (with a small epsilon).
    The sign (+1 for adversarial, -1 for clean) weights each direction;
    the average gives an estimated gradient direction.
    """
    eps = 1e-4  # tiny perturbation for sign probing
    dirs = unit_sphere_directions(num_samples, x_boundary.shape)
    signs = []
    for u in dirs:
        perturbed = np.clip(x_boundary + eps * u, 0.0, 1.0)
        pred = model.predict_label(perturbed)
        signs.append(1 if pred != clean_label else -1)
    signs = np.array(signs, dtype=np.float32).reshape(-1, 1, 1, 1)
    # Weighted sum of directions
    grad_est = np.mean(signs * dirs, axis=0)
    # Normalise to unit norm to obtain a direction vector
    norm = np.linalg.norm(grad_est)
    if norm == 0:
        return grad_est
    return grad_est / norm


def _geometric_step(
    x_boundary: np.ndarray,
    gradient: np.ndarray,
    model: BlackBoxModel,
    clean_label: int,
    clean: np.ndarray,
    initial_step: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Take a step orthogonal to the gradient toward the clean image.

    The step size is halved until the new point remains adversarial.
    Returns the new adversarial point and the number of queries performed.
    """
    step = initial_step
    queries = 0
    while step > 1e-6:
        # Move from the boundary toward the clean image, orthogonal to the
        # estimated gradient (i.e., in the direction ``-gradient``).
        direction = -gradient
        candidate = np.clip(x_boundary + step * direction, 0.0, 1.0)
        pred = model.predict_label(candidate)
        queries += 1
        if pred != clean_label:
            # Still adversarial – accept and break
            return candidate, queries
        # Otherwise reduce step and retry
        step /= 2.0
    # If we fall out of the loop, return the last boundary point (no progress)
    return x_boundary, queries


def run_hsja(
    clean_img: np.ndarray,
    model: BlackBoxModel,
    config: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, List[float], int]:
    """Run the HopSkipJumpAttack on a single image.

    Parameters
    ----------
    clean_img : np.ndarray
        Float image in ``[0, 1]`` shape ``(H, W, C)``.
    model : BlackBoxModel
        Wrapper providing ``predict_label`` and query counting.
    config : dict, optional
        Configuration options (see defaults below).

    Returns
    -------
    adv_img : np.ndarray
        Final adversarial image (float ``[0, 1]``).
    l2_history : list[float]
        L2 distance from the clean image after each iteration.
    total_queries : int
        Number of model queries used.
    """
    # -------------------------------------------------------------------
    # Default configuration
    # -------------------------------------------------------------------
    default_cfg = {
        "budget": 5000,          # max queries per image
        "iterations": 10,        # main HSJA iterations
        "max_num_evals": 1000,   # maximal MC samples per iteration (grows)
        "tolerance": 1e-5,
        "seed": 42,
    }
    if config:
        default_cfg.update(config)
    cfg = default_cfg

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    clean_label = model.predict_label(clean_img)
    # -------------------------------------------------------------------
    # 1. Find an initial adversarial point (random noise / heavy blend)
    # -------------------------------------------------------------------
    adv = None
    attempts = 0
    while adv is None:
        attempts += 1
        # Heavy random noise image (uniform [0,1])
        candidate = np.random.rand(*clean_img.shape).astype(np.float32)
        pred = model.predict_label(candidate)
        if pred != clean_label:
            adv = candidate
        if attempts > 1000:
            raise RuntimeError("Failed to find an initial adversarial point after many attempts.")

    # -------------------------------------------------------------------
    # 2. Project onto the decision boundary
    # -------------------------------------------------------------------
    x_boundary = _binary_search_boundary(
        clean_img, adv, model, clean_label, tolerance=cfg["tolerance"]
    )

    l2_history = [l2_distance(clean_img, x_boundary)]
    total_queries = model.global_query_count

    # -------------------------------------------------------------------
    # Main optimisation loop
    # -------------------------------------------------------------------
    for it in range(cfg["iterations"]):
        if total_queries >= cfg["budget"]:
            break
        # Number of MC directions grows with iteration (as in the paper)
        num_samples = min(cfg["max_num_evals"], 10 * (it + 1))
        grad = _estimate_gradient(x_boundary, model, clean_label, num_samples)
        # Geometric step (starting step size = distance to clean)
        step_size = l2_distance(x_boundary, clean_img)
        new_adv, step_queries = _geometric_step(
            x_boundary, grad, model, clean_label, clean_img, initial_step=step_size
        )
        total_queries += step_queries
        # Binary search again to land back on the boundary
        x_boundary = _binary_search_boundary(
            clean_img, new_adv, model, clean_label, tolerance=cfg["tolerance"]
        )
        l2_history.append(l2_distance(clean_img, x_boundary))
        total_queries = model.global_query_count
        if total_queries >= cfg["budget"]:
            break

    # -------------------------------------------------------------------
    # 6. Safety rounding – ensure the uint8 version stays adversarial
    # -------------------------------------------------------------------
    adv_uint8 = np.clip(np.rint(x_boundary * 255.0), 0, 255).astype(np.uint8) / 255.0
    if model.predict_label(adv_uint8) == clean_label:
        # Move a small epsilon toward the original adversarial side until label flips
        eps = 1e-3
        direction = x_boundary - clean_img
        direction /= np.linalg.norm(direction)
        while True:
            x_boundary = np.clip(x_boundary + eps * direction, 0.0, 1.0)
            adv_uint8 = np.clip(np.rint(x_boundary * 255.0), 0, 255).astype(np.uint8) / 255.0
            if model.predict_label(adv_uint8) != clean_label:
                break
            eps *= 1.5  # expand step if needed
        total_queries = model.global_query_count

    return x_boundary, l2_history, total_queries

# ---------------------------------------------------------------------------
# Test harness – run on three clean images and plot L2 vs queries curves
# ---------------------------------------------------------------------------

def _run_demo():
    import matplotlib.pyplot as plt

    model_path = PROJECT_ROOT / "models" / "cnn_model.pt"
    bb = BlackBoxModel(model_path)

    # Grab first three images (sorted alphabetically)
    img_paths = sorted([p for p in DATA_CLEAN.glob("*.png") if p.is_file()])[:3]
    for idx, p in enumerate(img_paths, start=1):
        clean = load_image(p)
        adv, l2_hist, _ = run_hsja(clean, bb, config={"budget": 2000, "iterations": 8})
        # Build cumulative query list – each iteration adds roughly the number of
        # model queries used in that iteration.  For a rough plot we assume a
        # linear increase based on length of ``l2_hist``.
        queries = np.arange(len(l2_hist)) * (bb.global_query_count // len(l2_hist))
        plt.figure()
        plt.plot(queries, l2_hist, marker="o")
        plt.title(f"HSJA L2 vs Queries (Image {idx})")
        plt.xlabel("Cumulative model queries")
        plt.ylabel("L2 distance to clean image")
        plt.grid(True)
        out_path = REPORTS_FIG / f"hsja_curve_{idx}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved curve for image {idx} -> {out_path}")

if __name__ == "__main__":
    _run_demo()
