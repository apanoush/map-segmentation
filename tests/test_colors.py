"""Palette rendering and label-map coloring."""

from __future__ import annotations

import numpy as np
import pytest

from image2shp.colors import color_for_index, name_for_index, palette_lut, render_labels
from image2shp.config import ClassSpec

CLASSES = (
    ClassSpec(index=0, name="background", color=(1, 2, 3)),
    ClassSpec(index=2, name="built", color=(0, 255, 0)),
)


def test_palette_lut_size_and_content():
    lut = palette_lut(CLASSES)
    assert lut.shape == (3, 3)
    assert lut.dtype == np.uint8
    assert list(lut[0]) == [1, 2, 3]
    assert list(lut[1]) == [0, 0, 0]  # undeclared index 1 falls back to black
    assert list(lut[2]) == [0, 255, 0]


def test_palette_lut_empty():
    assert palette_lut(()).shape == (1, 3)


def test_render_labels():
    labels = np.array([[0, 2], [2, 0]], dtype=np.int64)
    rgb = render_labels(labels, CLASSES)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert list(rgb[0, 0]) == [1, 2, 3]
    assert list(rgb[1, 0]) == [0, 255, 0]


def test_render_labels_unknown_indices_fall_back_to_class_zero():
    """Undeclared indices render as class 0 (the usual background class)."""
    labels = np.array([[99, -4]], dtype=np.int64)
    rgb = render_labels(labels, CLASSES)

    assert list(rgb[0, 0]) == [1, 2, 3]
    assert list(rgb[0, 1]) == [1, 2, 3]


def test_render_labels_black_when_class_zero_is_undeclared():
    classes = (ClassSpec(index=2, name="built", color=(0, 255, 0)),)
    rgb = render_labels(np.array([[99]]), classes)
    assert list(rgb[0, 0]) == [0, 0, 0]


def test_render_labels_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        render_labels(np.zeros((2, 2, 3)), CLASSES)


def test_index_helpers():
    assert color_for_index(CLASSES, 2) == (0, 255, 0)
    assert color_for_index(CLASSES, 7) == (0, 0, 0)
    assert name_for_index(CLASSES, 2) == "built"
    assert name_for_index(CLASSES, 7) == "class_7"
