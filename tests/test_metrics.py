# -*- coding: utf-8 -*-
"""tests/test_metrics.py

Simple sanity tests for the metrics module.

* Identical images → all distances zero, PSNR = inf, SSIM = 1.
* Image shifted by +1 (uint8) → known values for L2, L∞, L0 fraction, MAE, PSNR, SSIM.
"""

import numpy as np
from pathlib import Path

# Import the module under test
import sys, os
# Ensure the code directory is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code'))

from code.metrics import (
    l2_norm,
    linf_norm,
    l0_fraction,
    mean_absolute_error,
    psnr,
    ssim,
)


def test_identical_images():
    # Create a random uint8 image
    rng = np.random.default_rng(42)
    img = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
    # Compare image with itself
    assert l2_norm(img, img) == 0.0
    assert linf_norm(img, img) == 0.0
    assert l0_fraction(img, img) == 0.0
    assert mean_absolute_error(img, img) == 0.0
    assert psnr(img, img) == float("inf")
    assert ssim(img, img) == 1.0


def test_one_intensity_shift():
    # Base image
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    # Shifted image (+1 on every pixel, stays within uint8 range)
    shifted = np.full((4, 4, 3), 1, dtype=np.uint8)

    # Expected values
    expected_l2 = np.sqrt((1 ** 2) * base.size)  # each of 48 values diff=1
    expected_linf = 1.0
    expected_l0 = 1.0  # all pixels changed
    expected_mae = 1.0
    # PSNR for data_range=255 and MSE=1
    expected_psnr = 20 * np.log10(255) - 10 * np.log10(1)
    # SSIM for identical constant images offset by 1 should be 1.0 (structurally identical)
    # Our implementation (skimage fallback) returns 1.0 for constant images.
    expected_ssim = 1.0

    assert np.isclose(l2_norm(base, shifted), expected_l2)
    assert linf_norm(base, shifted) == expected_linf
    assert l0_fraction(base, shifted) == expected_l0
    assert mean_absolute_error(base, shifted) == expected_mae
    assert np.isclose(psnr(base, shifted), expected_psnr)
    assert np.isclose(ssim(base, shifted), expected_ssim)
