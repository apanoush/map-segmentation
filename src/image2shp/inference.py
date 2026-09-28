"""Inference with an mmsegmentation model.

This is the **only** module of the package that imports torch/mmseg, and it
does so lazily inside functions: importing :mod:`image2shp` must not require a
GPU stack, otherwise nothing can be unit-tested without one.

Input convention: mmsegmentation models are fed **BGR** arrays (the model
config's ``data_preprocessor`` sets ``bgr_to_rgb=True``, and the test pipeline
does not resize), so pixels are handed over exactly as the raster stores them,
channel-reversed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from image2shp.processing import RasterInfo, open_raster, read_window

logger = logging.getLogger(__name__)


class MissingBackendError(ImportError):
    """Raised when torch/mmsegmentation are not installed."""


class InferenceError(RuntimeError):
    """Raised when the model cannot be loaded or produces unexpected output."""


def _backend() -> tuple[Any, Any, Any, Any]:
    """Import torch + mmseg lazily, with an actionable error message."""
    try:
        import torch
        from mmengine.model.utils import revert_sync_batchnorm
        from mmseg.apis import inference_model, init_model
    except ImportError as exc:
        raise MissingBackendError(
            "mmsegmentation (and torch) are required to run inference.\n"
            "  * Docker (recommended): see doc/usage.md\n"
            "  * Locally:             uv sync --extra inference"
        ) from exc
    return torch, revert_sync_batchnorm, init_model, inference_model


def resolve_device(requested: str) -> str:
    """Turn a config ``device`` value into a concrete torch device string."""
    torch, *_ = _backend()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise InferenceError(
            "device='cuda' is configured but CUDA is not available; "
            'set [segmentation] device = "cpu" or "auto"'
        )
    return requested


def load_model(config_path: str | Path, checkpoint_path: str | Path, device: str = "auto"):
    """Build the segmentor from its config file and load the checkpoint."""
    torch, revert_sync_batchnorm, init_model, _ = _backend()
    resolved = resolve_device(device)
    if not Path(config_path).is_file():
        raise InferenceError(f"model config not found: {config_path}")
    if not Path(checkpoint_path).is_file():
        raise InferenceError(f"model checkpoint not found: {checkpoint_path}")

    logger.info("loading model %s (device=%s)", Path(checkpoint_path).name, resolved)
    model = init_model(str(config_path), str(checkpoint_path), device=resolved)
    if resolved == "cpu":
        model = revert_sync_batchnorm(model)
    return model


def predict_logits(model: Any, path: str | Path, window_size: int = 2048) -> np.ndarray:
    """Run segmentation on ``path`` and return raw logits shaped ``(C, H, W)``.

    Large rasters are read and inferred window by window to keep memory use
    bounded; ``window_size <= 0`` processes the whole image in one go.
    """
    return predict_logits_from_info(model, open_raster(path), window_size=window_size)


def predict_logits_from_info(model: Any, info: RasterInfo, window_size: int = 2048) -> np.ndarray:
    """Windowed :func:`predict_logits` on an already-opened raster."""
    _, _, _, inference_model = _backend()

    height, width = info.height, info.width
    if height <= 0 or width <= 0:
        raise InferenceError(f"{info.path}: raster has no pixels")

    step = window_size if window_size and window_size > 0 else max(height, width)
    prediction: np.ndarray | None = None

    for row_off in range(0, height, step):
        for col_off in range(0, width, step):
            win_h = min(step, height - row_off)
            win_w = min(step, width - col_off)

            rgb = read_window(info, col_off, row_off, win_w, win_h)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])

            result = inference_model(model, bgr)
            try:
                logits = result.seg_logits.data.detach().cpu().numpy()
            except AttributeError as exc:
                raise InferenceError(
                    "unexpected inference result: no seg_logits; "
                    "is the checkpoint compatible with mmsegmentation?"
                ) from exc

            if logits.ndim != 3:
                raise InferenceError(f"expected logits shaped (C, H, W), got {logits.shape}")
            if logits.shape[1:] != (win_h, win_w):
                raise InferenceError(
                    f"model returned {logits.shape[1:]} for a {(win_h, win_w)} "
                    "window; the model's test pipeline must not resize inputs"
                )

            if prediction is None:
                prediction = np.zeros((logits.shape[0], height, width), dtype=np.float32)
            prediction[:, row_off : row_off + win_h, col_off : col_off + win_w] = logits

            logger.debug("window (%d, %d) -> %s", row_off, col_off, tuple(logits.shape))

    assert prediction is not None  # loops always execute at least once
    return prediction


def labels_from_logits(logits: np.ndarray) -> np.ndarray:
    """Argmax over the class axis: ``(C, H, W)`` -> ``(H, W)`` int64."""
    logits = np.asarray(logits)
    if logits.ndim != 3:
        raise ValueError(f"expected logits shaped (C, H, W), got {logits.shape}")
    return np.argmax(logits, axis=0)
