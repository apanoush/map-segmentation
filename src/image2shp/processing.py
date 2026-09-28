"""Raster IO and mask post-processing.

This module is intentionally free of torch/mmsegmentation imports so that it
can be imported (and tested) without a GPU stack installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

SUPPORTED_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass(frozen=True)
class RasterInfo:
    """Georeferencing and layout of one input image."""

    path: Path
    height: int
    width: int
    count: int
    dtype: str
    transform: Any
    crs: Any
    profile: dict[str, Any] = field(repr=False)

    @property
    def is_georeferenced(self) -> bool:
        return self.crs is not None


# -----------------------------------------------------------------------------
# Reading
# -----------------------------------------------------------------------------


def open_raster(path: str | Path) -> RasterInfo:
    """Open ``path`` and return its metadata.

    GeoTIFFs (and anything else GDAL can open) keep their transform and CRS.
    Plain photos get an identity transform and no CRS, which means downstream
    outputs fall back to pixel coordinates.
    """
    raster_path = Path(path)
    with rasterio.open(raster_path) as src:
        return RasterInfo(
            path=raster_path,
            height=src.height,
            width=src.width,
            count=src.count,
            dtype=src.dtypes[0],
            transform=src.transform,
            crs=src.crs,
            profile=src.profile.copy(),
        )


def to_uint8(data: np.ndarray, percentile_clip: Sequence[float] = (2, 98)) -> np.ndarray:
    """Convert arbitrary raster data to uint8.

    * ``uint8`` passes through untouched.
    * Integers wider than 8 bits are clipped to ``percentile_clip`` percentiles
      and then scaled, which suppresses scanning artefacts/outliers.
    * Everything else (float, small integers) is min-max scaled.
    """
    data = np.asarray(data)
    if data.dtype == np.uint8:
        return data
    work = data.astype(np.float32)
    low, high = float(work.min()), float(work.max())
    if high <= low:
        return np.zeros(work.shape, dtype=np.uint8)
    if np.issubdtype(data.dtype, np.integer) and high > 255:
        p_low, p_high = np.percentile(work, percentile_clip)
        if p_high > p_low:
            low, high = float(p_low), float(p_high)
    scaled = np.clip((work - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _bands_to_rgb(bands: np.ndarray) -> np.ndarray:
    """(bands, H, W) -> (H, W, 3) uint8 RGB, padding/truncating as needed."""
    if bands.shape[0] == 1:
        bands = np.repeat(bands, 3, axis=0)
    elif bands.shape[0] == 2:
        bands = np.stack([bands[0], bands[0], bands[0]], axis=0)
    elif bands.shape[0] > 3:
        bands = bands[:3]
    return np.transpose(to_uint8(bands), (1, 2, 0))


def read_window(
    info: RasterInfo, col_off: int, row_off: int, width: int, height: int
) -> np.ndarray:
    """Read a window as ``(h, w, 3)`` uint8 RGB."""
    window = Window(col_off, row_off, width, height)
    with rasterio.open(info.path) as src:
        bands = src.read(window=window)
    return _bands_to_rgb(bands)


def read_image(path: str | Path) -> tuple[np.ndarray, RasterInfo]:
    """Read a whole image as ``(H, W, 3)`` uint8 RGB plus its metadata."""
    info = open_raster(path)
    return read_window(info, 0, 0, info.width, info.height), info


# -----------------------------------------------------------------------------
# Writing
# -----------------------------------------------------------------------------


def save_png(path: str | Path, image: np.ndarray) -> Path:
    """Write an RGB (or RGBA) image as PNG."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    code = cv2.COLOR_RGBA2BGRA if image.shape[2] == 4 else cv2.COLOR_RGB2BGR
    if not cv2.imwrite(str(output), cv2.cvtColor(image, code)):
        raise OSError(f"could not write PNG to {output}")
    return output


def save_geotiff(
    path: str | Path,
    data: np.ndarray,
    info: RasterInfo | None = None,
    *,
    compress: str = "lzw",
) -> Path:
    """Write ``data`` as a GeoTIFF, reusing ``info``'s georeferencing when given.

    ``data`` may be ``(H, W)`` (single band) or ``(H, W, C)``. Values are
    written as-is, so pass uint8 imagery or float32 densities.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(data)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"expected 2-D or 3-D data, got shape {array.shape}")

    count = array.shape[2]
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": count,
        "dtype": array.dtype,
        "transform": rasterio.Affine.identity(),
        "crs": None,
        "compress": compress,
    }
    if info is not None:
        profile.update(
            height=info.height,
            width=info.width,
            transform=info.transform,
            crs=info.crs,
        )
        # Keep source creation options (tiling, predictor, ...) when the dtype
        # still matches; otherwise they can be invalid for the new dtype.
        if info.profile.get("dtype") == array.dtype.name:
            for key in ("tiled", "blockxsize", "blockysize", "predictor", "interleave"):
                if key in info.profile:
                    profile[key] = info.profile[key]
            nodata = info.profile.get("nodata")
            if nodata is not None:
                profile["nodata"] = nodata

    if profile["width"] != array.shape[1] or profile["height"] != array.shape[0]:
        raise ValueError(
            f"data shape {array.shape[:2]} does not match raster "
            f"{profile['height']}x{profile['width']}"
        )

    with rasterio.open(output, "w", **profile) as dst:
        for band in range(count):
            dst.write(array[:, :, band], band + 1)
    return output


# -----------------------------------------------------------------------------
# Mask / density helpers
# -----------------------------------------------------------------------------


def normalize01(array: np.ndarray) -> np.ndarray:
    """Min-max scale ``array`` to ``[0, 1]`` (returns it unchanged if flat)."""
    array = np.asarray(array, dtype=np.float32)
    low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def class_mask(labels: np.ndarray, class_indices: Sequence[int]) -> np.ndarray:
    """Boolean mask of pixels whose label is one of ``class_indices``."""
    return np.isin(np.asarray(labels), tuple(class_indices))


def class_density(
    labels: np.ndarray, class_index: int, sigma: float = 25.0, *, normalize: bool = True
) -> np.ndarray:
    """Gaussian-smoothed density of ``class_index``, as float32 in ``[0, 1]``."""
    mask = class_mask(labels, [class_index]).astype(np.float32)
    density = gaussian_filter(mask, sigma=sigma)
    return normalize01(density) if normalize else density
