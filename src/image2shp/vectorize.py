"""Vectorization: class mask -> polygons -> shapefile.

The pipeline works on the label mask directly instead of hunting for an exact
"building color" in a rendered PNG (which silently broke as soon as the
rendering became a 6-class palette). :func:`segmented_image_to_shp` is kept as a
thin wrapper for callers that only have a colored PNG.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from skimage import measure

from image2shp.processing import open_raster


def mask_to_polygons(mask: np.ndarray, buffer_distance: int = 0) -> list[Polygon]:
    """Extract polygons from a boolean mask.

    ``buffer_distance`` dilates the mask by that many pixels first, which
    bridges small gaps between neighbouring building pixels before contour
    extraction.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []

    binary = mask.astype(np.uint8) * 255
    if buffer_distance > 0:
        kernel = np.ones((2 * buffer_distance + 1, 2 * buffer_distance + 1), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

    # Pad with background: contours that reach the image border (or a mask that
    # covers the whole image) are otherwise dropped or truncated.
    padded = np.pad(binary, 1, mode="constant", constant_values=0)

    polygons: list[Polygon] = []
    for contour in measure.find_contours(padded, level=128):
        contour = contour - 1.0  # undo the padding offset
        if len(contour) < 3:
            continue
        points = [(col, row) for row, col in contour]
        polygon = Polygon(points)
        if polygon.is_valid and polygon.area > 0:
            polygons.append(polygon)
    return polygons


def polygons_to_gdf(polygons: Sequence[Polygon], transform, crs) -> gpd.GeoDataFrame:
    """Assign ``transform``'s pixel-to-world mapping and ``crs`` to polygons."""
    if transform is None:
        from affine import Affine

        transform = Affine.identity()
    geoms = [Polygon([transform * point for point in poly.exterior.coords]) for poly in polygons]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length
    return gdf


def mask_to_shapefile(
    mask: np.ndarray,
    output_path: str | Path,
    transform,
    crs,
    *,
    buffer_distance: int = 20,
) -> gpd.GeoDataFrame:
    """Extract polygons from ``mask`` and write them to ``output_path``."""
    polygons = mask_to_polygons(mask, buffer_distance=buffer_distance)
    if not polygons:
        raise ValueError("no polygons found in mask (nothing to vectorize)")
    gdf = polygons_to_gdf(polygons, transform, crs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output)
    return gdf


def segmented_image_to_shp(
    png_path: str | Path,
    geotiff_path: str | Path,
    output_shp_path: str | Path,
    building_color: tuple[int, int, int] = (255, 0, 255),
    buffer_distance: int = 20,
) -> gpd.GeoDataFrame:
    """Convert a colored segmentation PNG to a shapefile.

    Kept for backward compatibility: pixels exactly equal to ``building_color``
    (RGB) are vectorized and georeferenced with ``geotiff_path``. Prefer
    :func:`mask_to_shapefile` when the label map is available.

    Raises:
        ValueError: if the PNG cannot be read or contains no building pixels.
    """
    image = cv2.imread(str(png_path))
    if image is None:
        raise ValueError(f"could not load image from {png_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.all(rgb == np.asarray(building_color), axis=2)
    if not mask.any():
        raise ValueError(f"no pixels match building_color {building_color} in {png_path}")
    info = open_raster(geotiff_path)
    return mask_to_shapefile(
        mask,
        output_shp_path,
        info.transform,
        info.crs,
        buffer_distance=buffer_distance,
    )
