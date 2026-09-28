"""Raster IO and mask/density helpers (no model involved)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio

from image2shp.processing import (
    class_density,
    class_mask,
    normalize01,
    open_raster,
    read_image,
    save_geotiff,
    save_png,
    to_uint8,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_to_uint8_passes_uint8_through():
    data = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert to_uint8(data) is data


def test_to_uint8_scales_uint16_with_percentile_clipping():
    data = np.arange(0, 10_000, dtype=np.uint16).reshape(100, 100)
    out = to_uint8(data)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255
    # values below the 2nd percentile are flattened to 0: without clipping row 1
    # (values 100-199) would start at ~5
    assert (out[1] == 0).all()
    assert (out[-1] == 255).all()


def test_to_uint8_handles_constant_data():
    data = np.full((4, 4), 7, dtype=np.uint16)
    assert (to_uint8(data) == 0).all()


def test_to_uint8_scales_floats():
    data = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    out = to_uint8(data)
    assert out.max() == 255


def test_normalize01():
    flat = np.ones((3, 3), dtype=np.float32)
    assert (normalize01(flat) == 0).all()

    values = np.array([[0.0, 5.0], [10.0, 2.0]])
    scaled = normalize01(values)
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0


def test_class_mask():
    labels = np.array([[0, 1], [2, 1]])
    mask = class_mask(labels, [1, 2])
    assert mask.tolist() == [[False, True], [True, True]]


def test_class_density_is_normalised_and_smooth():
    labels = np.zeros((40, 40), dtype=np.int64)
    labels[10:20, 10:20] = 2

    density = class_density(labels, 2, sigma=2)
    assert density.dtype == np.float32
    assert density.shape == (40, 40)
    assert 0.0 <= density.min() <= density.max() <= 1.0
    assert density.max() == pytest.approx(1.0)
    assert density[0, 0] == pytest.approx(0.0)

    absent = class_density(labels, 9, sigma=2)
    assert absent.max() == 0.0


def test_open_raster_on_georeferenced_image(geotiff: Path):
    info = open_raster(geotiff)
    assert (info.height, info.width) == (64, 48)
    assert info.count == 3
    assert info.dtype == "uint8"
    assert info.is_georeferenced
    assert info.crs.to_epsg() == 2056
    assert info.transform.a == 1.0


def test_read_image_returns_rgb(geotiff: Path):
    image, info = read_image(geotiff)
    assert image.shape == (64, 48, 3)
    assert image.dtype == np.uint8
    assert info.path == geotiff


def test_plain_png_has_no_georeferencing(plain_image: Path):
    image, info = read_image(plain_image)
    assert image.shape == (32, 40, 3)
    assert not info.is_georeferenced
    assert info.crs is None


def test_save_png_roundtrip(tmp_path: Path):
    expected = np.zeros((5, 7, 3), dtype=np.uint8)
    expected[2, 3] = (255, 0, 255)

    path = save_png(tmp_path / "out" / "image.png", expected)
    assert path.is_file()

    loaded = cv2.imread(str(path))
    assert loaded is not None
    assert loaded.shape == (5, 7, 3)
    assert list(loaded[2, 3][::-1]) == [255, 0, 255]  # BGR on disk


def test_save_geotiff_keeps_georeferencing(tmp_path: Path, geotiff: Path):
    info = open_raster(geotiff)
    data = np.full((64, 48, 3), 42, dtype=np.uint8)

    path = save_geotiff(tmp_path / "rgb.tif", data, info)
    with rasterio.open(path) as src:
        assert src.count == 3
        assert src.dtypes[0] == "uint8"
        assert src.crs.to_epsg() == 2056
        assert src.transform == info.transform
        assert src.read(1)[0, 0] == 42


def test_save_geotiff_single_band_float(tmp_path: Path, geotiff: Path):
    info = open_raster(geotiff)
    density = np.linspace(0, 1, 64 * 48, dtype=np.float32).reshape(64, 48)

    path = save_geotiff(tmp_path / "density.tif", density, info)
    with rasterio.open(path) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert src.crs.to_epsg() == 2056
        assert src.read(1).max() == pytest.approx(1.0)


def test_save_geotiff_without_reference_uses_identity(tmp_path: Path):
    path = save_geotiff(tmp_path / "bare.tif", np.zeros((4, 4), dtype=np.uint8))
    with rasterio.open(path) as src:
        assert src.crs is None
        assert src.transform.is_identity


def test_save_geotiff_rejects_shape_mismatch(tmp_path: Path, geotiff: Path):
    info = open_raster(geotiff)
    with pytest.raises(ValueError, match="does not match raster"):
        save_geotiff(tmp_path / "bad.tif", np.zeros((3, 3), dtype=np.uint8), info)


@pytest.mark.parametrize("name", ["test_1.png", "test_2.png"])
def test_shipped_fixture_images_are_readable(name: str):
    image, info = read_image(FIXTURES / name)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    assert not info.is_georeferenced
