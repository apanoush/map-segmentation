"""Shared fixtures: synthetic GeoTIFFs and a ready-to-use config file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from image2shp.config import ClassSpec, load_config

# Three classes are enough to exercise palette handling and target selection.
CLASSES = (
    ClassSpec(index=0, name="background", color=(0, 0, 0)),
    ClassSpec(index=1, name="contours", color=(0, 0, 255)),
    ClassSpec(index=2, name="built", color=(0, 255, 0)),
)


@pytest.fixture
def classes() -> tuple[ClassSpec, ...]:
    return CLASSES


@pytest.fixture
def geotiff(tmp_path: Path) -> Path:
    """A small georeferenced RGB raster (EPSG:2056, 1 m pixels)."""
    path = tmp_path / "input.tif"
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(3, 64, 48), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=48,
        count=3,
        dtype="uint8",
        crs="EPSG:2056",
        transform=from_origin(2_480_000.0, 1_120_000.0, 1.0, 1.0),
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def plain_image(tmp_path: Path) -> Path:
    """A georeference-free photo (PNG) to check the non-GIS path."""
    import cv2

    path = tmp_path / "photo.png"
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[5:15, 5:25] = (10, 200, 30)
    cv2.imwrite(str(path), image)
    return path


def write_config(
    directory: Path,
    *,
    data_dir: Path,
    results_dir: Path,
    model_config: Path,
    checkpoint: Path,
    extra: str = "",
) -> Path:
    """Write a minimal but complete config file inside ``directory``."""
    config_path = directory / "config.toml"
    config_path.write_text(
        f"""
[paths]
data_dir = "{data_dir}"
results_dir = "{results_dir}"

[model]
config = "{model_config}"
checkpoint = "{checkpoint}"

[inputs]
patterns = ["*.tif", "*.tiff", "*.png"]

[segmentation]
device = "cpu"
window_size = 32
target_class = "built"
heatmap_sigma = 3
buffer_distance = 2

[outputs]
segmented_png = true
segmented_geotiff = true
heatmap_geotiff = true
shapefile = true

[[classes]]
index = 0
name = "background"
color = "#000000"

[[classes]]
index = 1
name = "contours"
color = "#0000ff"

[[classes]]
index = 2
name = "built"
color = "#00ff00"
{extra}
""",
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def config_path(tmp_path: Path, geotiff: Path) -> Path:
    """A config whose model files and data directory all exist."""
    data_dir = geotiff.parent / "data"
    data_dir.mkdir(exist_ok=True)
    data_dir.joinpath(geotiff.name).write_bytes(geotiff.read_bytes())

    results_dir = tmp_path / "results"
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    model_config = model_dir / "model.py"
    model_config.write_text("# stand-in for an mmsegmentation config\n", encoding="utf-8")
    checkpoint = model_dir / "weights.pth"
    checkpoint.write_bytes(b"not a real checkpoint")

    return write_config(
        tmp_path,
        data_dir=data_dir,
        results_dir=results_dir,
        model_config=model_config,
        checkpoint=checkpoint,
    )


@pytest.fixture
def config(config_path: Path):
    return load_config(config_path)


@pytest.fixture
def fake_logits():
    """Build a synthetic ``(C, H, W)`` logit cube from a raster's shape."""

    def _make(info, class_index: int = 2, num_classes: int = 3) -> np.ndarray:
        logits = np.full((num_classes, info.height, info.width), -5.0, dtype=np.float32)
        # Make `class_index` win over a rectangular region, everything else
        # stays with class 0 elsewhere.
        logits[0, :, :] = 0.0
        logits[class_index, 8 : info.height // 2, 6 : info.width // 2] = 10.0
        return logits

    return _make
