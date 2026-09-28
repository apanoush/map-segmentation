"""Rendering of class-label maps using the palette defined in ``config.toml``."""

from __future__ import annotations

import numpy as np

from image2shp.config import ClassSpec

RGB = tuple[int, int, int]


def palette_lut(classes: tuple[ClassSpec, ...] | list[ClassSpec]) -> np.ndarray:
    """Build a ``(max_index + 1, 3)`` uint8 RGB lookup table from ``classes``.

    Indices that are not declared in the configuration map to black, so a model
    emitting more classes than the config describes still renders (in black).
    """
    if not classes:
        return np.zeros((1, 3), dtype=np.uint8)
    size = max(spec.index for spec in classes) + 1
    lut = np.zeros((size, 3), dtype=np.uint8)
    for spec in classes:
        lut[spec.index] = spec.color
    return lut


def render_labels(labels: np.ndarray, classes: tuple[ClassSpec, ...]) -> np.ndarray:
    """Map a ``(H, W)`` label map to a ``(H, W, 3)`` uint8 **RGB** image.

    Indices that are not declared in ``classes`` render as class 0 (the usual
    background class), which is black unless class 0 declares another color.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"label map must be 2-D, got shape {labels.shape}")
    lut = palette_lut(classes)
    known = (labels >= 0) & (labels < len(lut))
    return lut[np.where(known, labels, 0)]


def color_for_index(classes: tuple[ClassSpec, ...], index: int) -> RGB:
    """RGB color configured for ``index`` (black when undeclared)."""
    for spec in classes:
        if spec.index == index:
            return spec.color
    return (0, 0, 0)


def name_for_index(classes: tuple[ClassSpec, ...], index: int) -> str:
    """Human readable name for ``index`` (falls back to ``class_<n>``)."""
    for spec in classes:
        if spec.index == index:
            return spec.name
    return f"class_{index}"
