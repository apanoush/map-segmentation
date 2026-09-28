"""Mask -> polygon -> shapefile conversion."""

from __future__ import annotations

from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin

from image2shp.processing import open_raster
from image2shp.vectorize import (
    mask_to_polygons,
    mask_to_shapefile,
    polygons_to_gdf,
    segmented_image_to_shp,
)


def square_mask(size: int = 40, from_: int = 10, to: int = 30) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[from_:to, from_:to] = True
    return mask


def test_mask_to_polygons_without_buffer():
    polygons = mask_to_polygons(square_mask(), buffer_distance=0)

    assert len(polygons) == 1
    # area is in pixel units: the 20x20 block is ~400 px
    assert polygons[0].area == pytest.approx(400, rel=0.1)
    bounds = polygons[0].bounds
    assert bounds[0] == pytest.approx(10, abs=1)
    assert bounds[1] == pytest.approx(10, abs=1)


def test_buffer_distance_grows_the_polygon():
    plain = mask_to_polygons(square_mask(), buffer_distance=0)[0]
    buffered = mask_to_polygons(square_mask(), buffer_distance=3)[0]
    assert buffered.area > plain.area


def test_empty_mask_yields_no_polygons():
    assert mask_to_polygons(np.zeros((20, 20), dtype=bool)) == []


def test_mask_to_shapefile_writes_georeferenced_polygons(tmp_path: Path):
    mask = square_mask()
    transform = from_origin(2_480_000.0, 1_120_000.0, 1.0, 1.0)

    output = tmp_path / "nested" / "buildings.shp"
    gdf = mask_to_shapefile(mask, output, transform, "EPSG:2056", buffer_distance=0)

    assert output.is_file()
    assert len(gdf) == 1
    assert gdf.crs.to_epsg() == 2056
    assert {"area", "perimeter"}.issubset(gdf.columns)

    reloaded = gpd.read_file(output)
    assert len(reloaded) == len(gdf)
    # transform is north-up: column 10 -> x = origin + 10, row 10 -> y = origin - 10
    minx, miny, maxx, maxy = reloaded.total_bounds
    assert minx == pytest.approx(2_480_010.0, abs=1)
    assert maxy == pytest.approx(1_120_000.0 - 10.0, abs=1)
    assert miny == pytest.approx(1_120_000.0 - 30.0, abs=1)


def test_mask_to_shapefile_without_crs_uses_pixel_coordinates(tmp_path: Path):
    output = tmp_path / "pixel.shp"
    gdf = mask_to_shapefile(square_mask(), output, None, None, buffer_distance=0)
    assert gdf.crs is None
    minx, _, maxx, _ = gdf.total_bounds
    assert 9 < minx < 11


def test_mask_to_shapefile_without_polygons_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="no polygons"):
        mask_to_shapefile(
            np.zeros((10, 10), dtype=bool),
            tmp_path / "nothing.shp",
            from_origin(0, 10, 1, 1),
            None,
        )


def test_polygons_to_gdf_applies_transform():
    from shapely.geometry import Polygon

    unit = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    gdf = polygons_to_gdf(unit, from_origin(100.0, 200.0, 2.0, 2.0), "EPSG:2056")

    minx, miny, maxx, maxy = gdf.total_bounds
    assert (minx, maxy) == (100.0, 200.0)
    assert (maxx - minx, maxy - miny) == (2.0, 2.0)


def test_segmented_image_to_shp_wrapper(tmp_path: Path, geotiff: Path):
    png = tmp_path / "segmented.png"
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[8:24, 8:24] = (255, 0, 255)  # pink buildings
    cv2.imwrite(str(png), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    output = tmp_path / "wrapper.shp"
    gdf = segmented_image_to_shp(png, geotiff, output, buffer_distance=0)

    assert output.is_file()
    assert len(gdf) == 1
    assert gdf.crs.to_epsg() == open_raster(geotiff).crs.to_epsg()


def test_segmented_image_to_shp_requires_building_pixels(tmp_path: Path, geotiff: Path):
    png = tmp_path / "empty.png"
    cv2.imwrite(str(png), np.zeros((16, 16, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="no pixels match"):
        segmented_image_to_shp(png, geotiff, tmp_path / "out.shp")


def test_segmented_image_to_shp_requires_readable_png(tmp_path: Path, geotiff: Path):
    with pytest.raises(ValueError, match="could not load image"):
        segmented_image_to_shp(tmp_path / "missing.png", geotiff, tmp_path / "out.shp")


def test_render_then_vectorize_round_trip(tmp_path: Path, geotiff: Path):
    """A rendered class map stays consumable by the PNG-based API."""
    from image2shp.colors import render_labels
    from image2shp.config import ClassSpec
    from image2shp.processing import save_png

    classes = (
        ClassSpec(index=0, name="background", color=(0, 0, 0)),
        ClassSpec(index=2, name="built", color=(255, 0, 255)),
    )
    labels = np.zeros((48, 48), dtype=np.int64)
    labels[8:24, 8:24] = 2

    png = tmp_path / "rendered.png"
    save_png(png, render_labels(labels, classes))

    gdf = segmented_image_to_shp(png, geotiff, tmp_path / "roundtrip.shp", buffer_distance=0)
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].area == pytest.approx(256, rel=0.2)
