"""image2shp: segment map images with an mmsegmentation model and export the
result as colored rasters, heatmaps and vector polygons.

The package deliberately keeps mmsegmentation (and therefore torch) out of the
import path: only :mod:`image2shp.inference` touches it, and it does so lazily.
Everything else is importable — and testable — without a GPU stack installed.
"""

from image2shp.config import Config, ConfigError, load_config

__version__ = "0.1.0"

__all__ = ["Config", "ConfigError", "load_config", "__version__"]
