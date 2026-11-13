# -*- coding: utf-8 -*-
"""blackbox.py

Utility wrapper around the provided pretrained CNN model for the black‑box
adversarial attack.  The wrapper hides all model internals and only exposes a
`predict_label` method that returns integer class predictions.

Features:
- Global query counter and per‑image query counters.
- Optional per‑image query budget (raises ``QueryBudgetExceeded``).
- Helpers to load PNG images as float arrays in ``[0, 1]`` range and to save
  float arrays back to PNG with proper uint8 rounding.
- ``verify_saved`` validates that a saved adversarial image is indeed *adversarial*
  by checking that the model prediction differs from the supplied clean label.

The module can be executed directly to run a quick smoke test on a few clean
images.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QueryBudgetExceeded(RuntimeError):
    """Raised when the per‑image query budget is exceeded."""

# ---------------------------------------------------------------------------
# Helper functions (image I/O)
# ---------------------------------------------------------------------------

def _pil_to_float_array(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a ``float32`` numpy array in ``[0, 1]``.

    The returned array has shape ``(H, W, C)`` and dtype ``float32``.
    """
    arr = np.asarray(pil_img).astype(np.float32) / 255.0
    # Ensure ``H, W, C`` ordering (PIL already uses this ordering)
    if arr.ndim == 2:  # grayscale → expand channel dim
        arr = np.expand_dims(arr, axis=2)
    return arr


def load_png(path: Union[str, os.PathLike]) -> np.ndarray:
    """Load a PNG file and return a ``float32`` array in ``[0, 1]``.

    Parameters
    ----------
    path:
        Path to the PNG image.

    Returns
    -------
    np.ndarray
        Float array with shape ``(H, W, C)``.
    """
    pil_img = Image.open(path).convert("RGB")
    return _pil_to_float_array(pil_img)


def save_png(img: np.ndarray, path: Union[str, os.PathLike]) -> None:
    """Save a float image ``[0, 1]`` to PNG.

    The image is rounded to the nearest ``uint8`` value.
    """
    if img.dtype != np.float32 and img.dtype != np.float64:
        raise TypeError("Image array must be of float dtype.")
    # Clip and convert to uint8
    img_uint8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    # ``PIL`` expects ``(H, W, C)``
    pil_img = Image.fromarray(img_uint8)
    pil_img.save(path)

# ---------------------------------------------------------------------------
# Black‑box model wrapper
# ---------------------------------------------------------------------------

class BlackBoxModel:
    """A strict black‑box wrapper around the pretrained CNN.

    Only the ``predict_label`` method is exposed.  Internals such as the raw
    ``torch`` model, weights, logits or probabilities are kept private.
    """

    def __init__(
        self,
        model_path: Union[str, os.PathLike],
        mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
        std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
        device: Optional[str] = None,
    ) -> None:
        """Load the TorchScript model and initialise counters.

        Parameters
        ----------
        model_path:
            Path to ``cnn_model.pt``.
        mean, std:
            Normalisation parameters used by the original notebook.
        device:
            ``"cuda"`` or ``"cpu"``.  If ``None`` the script decides based on
            ``torch.cuda.is_available()``.
        """
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Load the model – keep it private
        self._model = torch.jit.load(str(model_path), map_location=self._device)
        self._model.eval()

        # Normalisation tensors (shape C,1,1) as used in the notebook
        self._mean = torch.tensor(mean, dtype=torch.float32, device=self._device).view(3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32, device=self._device).view(3, 1, 1)

        # Counters
        self.global_query_count: int = 0
        self._per_image_counts: Dict[bytes, int] = {}
        self._query_budget: Optional[int] = None  # same budget for *all* images if set

    # ---------------------------------------------------------------------
    # Counter utilities
    # ---------------------------------------------------------------------
    def _hash_image(self, img_tensor: torch.Tensor) -> bytes:
        """Return a deterministic hash for an image tensor.

        The image tensor is expected to be in ``(C, H, W)`` format on CPU.
        """
        # Ensure deterministic representation – use ``numpy`` bytes
        arr = img_tensor.detach().cpu().numpy().tobytes()
        return hashlib.sha256(arr).digest()

    def set_query_budget(self, budget: int) -> None:
        """Set a per‑image query budget (same for every image)."""
        if budget <= 0:
            raise ValueError("Budget must be a positive integer.")
        self._query_budget = budget

    # ---------------------------------------------------------------------
    # Core prediction method
    # ---------------------------------------------------------------------
    def _preprocess(self, img: Union[np.ndarray, torch.Tensor, Image.Image]) -> torch.Tensor:
        """Convert a single image to a normalised ``torch`` tensor.

        The output tensor has shape ``(3, H, W)`` on ``self._device``.
        """
        if isinstance(img, Image.Image):
            arr = _pil_to_float_array(img)
        elif isinstance(img, np.ndarray):
            if img.dtype != np.float32 and img.dtype != np.float64:
                raise TypeError("Numpy image must be float in [0, 1].")
            arr = img
        elif isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Ensure ``H, W, C`` → ``C, H, W`` and convert to torch tensor
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Image must have shape (H, W, 3).")
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(self._device, dtype=torch.float32)
        # Normalise as in the notebook
        tensor = (tensor - self._mean) / self._std
        return tensor

    def predict_label(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor, List[Any], Tuple[Any, ...]],
    ) -> Union[int, torch.Tensor]:
        """Predict integer label(s) for the supplied image(s).

        * **Single image** – returns an ``int``.
        * **Batch** (list/tuple/torch Tensor) – returns a 1‑D ``torch.Tensor`` of
          ``int64`` labels.

        The query counters are updated accordingly.  If a per‑image budget has
        been set and the budget is exceeded, a ``QueryBudgetExceeded`` exception
        is raised.
        """
        # Detect batch vs single image
        if isinstance(img, (list, tuple)):
            # Batch of images – preprocess each one individually
            tensors = [self._preprocess(i) for i in img]
            batch = torch.stack(tensors, dim=0)  # shape (N, C, H, W)
            batch_size = batch.shape[0]
        elif isinstance(img, torch.Tensor) and img.dim() == 4:
            # Already a batch tensor (N, C, H, W)
            batch = img.to(self._device)
            batch_size = batch.shape[0]
        else:
            # Single image
            batch = self._preprocess(img).unsqueeze(0)  # add batch dim
            batch_size = 1

        # Update counters before the forward pass (so even failed queries are counted)
        self.global_query_count += batch_size
        # Per‑image counters (hash each image in the batch)
        for i in range(batch_size):
            img_hash = self._hash_image(batch[i])
            self._per_image_counts[img_hash] = self._per_image_counts.get(img_hash, 0) + 1
            if self._query_budget is not None and self._per_image_counts[img_hash] > self._query_budget:
                raise QueryBudgetExceeded(f"Query budget of {self._query_budget} exceeded for an image.")

        # Forward pass – the model returns integer class IDs (as a tensor)
        with torch.no_grad():
            preds = self._model(batch)
        # Ensure integer type
        if preds.dtype != torch.int64:
            preds = preds.long()

        if batch_size == 1:
            return int(preds.item())
        return preds.squeeze()

    # ---------------------------------------------------------------------
    # Verification helper
    # ---------------------------------------------------------------------
    def verify_saved(self, adv_path: Union[str, os.PathLike], clean_label: int) -> bool:
        """Load a saved adversarial PNG and confirm its prediction differs.

        Returns ``True`` if the prediction **differs** from ``clean_label``,
        ``False`` otherwise.
        """
        adv_img = load_png(adv_path)
        pred = self.predict_label(adv_img)
        return int(pred) != int(clean_label)

# ---------------------------------------------------------------------------
# Smoke test (executed when running the module directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob

    # Paths – adjust if the script is moved
    ROOT = Path(__file__).parents[1]  # ``blackbox-attack`` directory
    MODEL_PATH = ROOT / "models" / "cnn_model.pt"
    CLEAN_DIR = ROOT / "data" / "clean"

    # Initialise the wrapper (seed is set globally by Python's RNG – not needed here)
    bb = BlackBoxModel(MODEL_PATH)

    # Grab first 5 PNG files (sorted for reproducibility)
    sample_paths = sorted(glob.glob(str(CLEAN_DIR / "*.png")))[:5]
    print("--- Smoke test: predictions on 5 clean images ---")
    for p in sample_paths:
        img_arr = load_png(p)
        pred_label = bb.predict_label(img_arr)
        # Filename format: <class>_<idx>.png – extract class name
        basename = os.path.basename(p)
        class_name = basename.split("_")[0]
        print(f"{basename:30} -> predicted label: {pred_label} (class from name: {class_name})")
    print("\nTotal queries made:", bb.global_query_count)
    # The model is 100 % accurate on the clean set according to the notebook.
    # If the extracted class name does not match the numeric label, it is due to
    # the mapping ``class_names`` used in the notebook – the numeric label is the
    # index of that name in the list.
    # This smoke test merely shows that the wrapper works and that the query
    # counter is updated.
