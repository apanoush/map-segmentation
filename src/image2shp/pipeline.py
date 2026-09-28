"""End-to-end pipeline: discover inputs, segment them, write all outputs."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from image2shp import colors, inference, processing, vectorize
from image2shp.config import Config

logger = logging.getLogger(__name__)

# output kind -> subdirectory under results_dir
OUTPUT_DIRS = {
    "segmented_png": "segmented",
    "segmented_geotiff": "segmented",
    "heatmap_geotiff": "heatmap",
    "shapefile": "vector",
}

_OUTPUT_SUFFIX = {
    "segmented_png": ".png",
    "segmented_geotiff": ".tif",
    "heatmap_geotiff": ".tif",
    "shapefile": ".shp",
}


@dataclass
class FileReport:
    """Outcome of processing a single input image."""

    path: Path
    ok: bool
    outputs: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def output_names(self) -> list[str]:
        return [str(path) for path in self.outputs]


@dataclass
class RunReport:
    """Outcome of a whole run."""

    files: list[FileReport] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(report.ok for report in self.files)

    @property
    def succeeded(self) -> int:
        return sum(1 for report in self.files if report.ok)

    @property
    def failed(self) -> int:
        return sum(1 for report in self.files if not report.ok)

    def summary(self) -> str:
        return f"{self.succeeded} succeeded, {self.failed} failed, {len(self.files)} total"


def discover_inputs(config: Config) -> list[Path]:
    """Return the sorted, de-duplicated inputs matching ``[inputs] patterns``."""
    data_dir = config.paths.data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    found: dict[Path, None] = {}
    for pattern in config.inputs.patterns:
        globber = f"**/{pattern}" if config.inputs.recursive else pattern
        for candidate in data_dir.glob(globber):
            if candidate.is_file():
                found.setdefault(candidate.resolve(), None)
    return sorted(found, key=lambda path: path.name)


def output_paths(config: Config, source: Path) -> dict[str, Path]:
    """Where each enabled output for ``source`` will be written."""
    results_dir = config.paths.results_dir
    stem = source.stem
    paths: dict[str, Path] = {}
    for kind, enabled in (
        ("segmented_png", config.outputs.segmented_png),
        ("segmented_geotiff", config.outputs.segmented_geotiff),
        ("heatmap_geotiff", config.outputs.heatmap_geotiff),
        ("shapefile", config.outputs.shapefile),
    ):
        if enabled:
            paths[kind] = results_dir / OUTPUT_DIRS[kind] / f"{stem}{_OUTPUT_SUFFIX[kind]}"
    return paths


def process_file(
    model: Any,
    config: Config,
    source: Path,
    *,
    skip_existing: bool = False,
) -> FileReport:
    """Segment one image and write every output enabled in the config."""
    targets = output_paths(config, source)
    report = FileReport(path=source, ok=False)

    if skip_existing and targets and all(path.exists() for path in targets.values()):
        report.notes.append("all outputs already exist, skipped")
        report.ok = True
        return report

    try:
        info = processing.open_raster(source)
        if not info.is_georeferenced:
            report.notes.append(
                "input has no georeferencing; geotiff/shapefile outputs use pixel coordinates"
            )

        logits = inference.predict_logits_from_info(
            model, info, window_size=config.segmentation.window_size
        )
        labels = inference.labels_from_logits(logits)

        target = config.target_class_spec
        mask = processing.class_mask(labels, [target.index])

        rgb: np.ndarray | None = None
        if "segmented_png" in targets or "segmented_geotiff" in targets:
            rgb = colors.render_labels(labels, config.classes)

        if "segmented_png" in targets:
            report.outputs.append(processing.save_png(targets["segmented_png"], rgb))
        if "segmented_geotiff" in targets:
            report.outputs.append(processing.save_geotiff(targets["segmented_geotiff"], rgb, info))
        if "heatmap_geotiff" in targets:
            density = processing.class_density(
                labels, target.index, config.segmentation.heatmap_sigma
            )
            report.outputs.append(
                processing.save_geotiff(targets["heatmap_geotiff"], density, info)
            )
        if "shapefile" in targets:
            gdf = vectorize.mask_to_shapefile(
                mask,
                targets["shapefile"],
                info.transform,
                info.crs,
                buffer_distance=config.segmentation.buffer_distance,
            )
            report.outputs.append(targets["shapefile"])
            report.notes.append(f"{len(gdf)} polygons")

        report.ok = True
    except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
        report.error = f"{type(exc).__name__}: {exc}"
        logger.exception("failed to process %s", source)

    return report


def run(
    config: Config,
    files: Sequence[Path],
    *,
    model: Any | None = None,
    device: str | None = None,
    skip_existing: bool = False,
) -> RunReport:
    """Process ``files`` with a single model instance.

    ``model`` may be injected (used by tests) to avoid loading mmsegmentation.
    """
    report = RunReport()

    duplicates: dict[str, Path] = {}
    unique: list[Path] = []
    for path in files:
        previous = duplicates.get(path.stem)
        if previous is not None:
            report.files.append(
                FileReport(
                    path=path,
                    ok=False,
                    error=(
                        f"output name collision: {path.name} and {previous.name} "
                        "share a file stem, rename one of them"
                    ),
                )
            )
            continue
        duplicates[path.stem] = path
        unique.append(path)

    if unique and model is None:
        model = inference.load_model(
            config.model.config_path,
            config.model.checkpoint_path,
            device=config.segmentation.device if device is None else device,
        )

    results_dir = config.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(unique, start=1):
        logger.info("[%d/%d] %s", index, len(unique), path.name)
        report.files.append(process_file(model, config, path, skip_existing=skip_existing))
        for output in report.files[-1].outputs:
            logger.info("  wrote %s", output)

    return report


def run_from_paths(config: Config, files: Iterable[Path], **kwargs: Any) -> RunReport:
    """Convenience wrapper taking any iterable of paths."""
    return run(config, list(files), **kwargs)
