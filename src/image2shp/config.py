from dataclasses import dataclass
from pathlib import Path
import tomli


@dataclass
class DataPaths:
    images_dir: Path
    georeferenced_tiff_1973_path: Path
    georeferenced_tiff_1983_path: Path


@dataclass
class ResultPaths:
    segmented_png_results_dir: Path
    shp_results_dir: Path


@dataclass
class ModelPaths:
    config_path: Path
    checkpoint_path: Path


@dataclass
class SegmentationConfig:
    building_class_idx: int
    buffer_distance: int


@dataclass
class PathsConfig:
    data: DataPaths
    model: ModelPaths
    results: ResultPaths


@dataclass
class Config:
    paths: PathsConfig
    segmentation: SegmentationConfig


def load_config(path = "config.toml") -> Config:
    path = Path(path)

    with path.open("rb") as f:
        data = tomli.load(f)

    base_dir = path.parent
    paths_data = data["paths"]

    data_paths_raw = paths_data["data"]

    images_dir = (base_dir / Path(data_paths_raw["images_dir"]).expanduser()).resolve()

    tiff_1973 = Path(data_paths_raw["georeferenced_tiff_1973_path"]).expanduser()
    tiff_1983 = Path(data_paths_raw["georeferenced_tiff_1983_path"])

    # If relative : interpret relative to images_dir
    if not tiff_1973.is_absolute():
        tiff_1973 = images_dir / tiff_1973

    if not tiff_1983.is_absolute():
        tiff_1983 = images_dir / tiff_1983

    data_paths = DataPaths(
        images_dir=images_dir,
        georeferenced_tiff_1973_path=tiff_1973.resolve(),
        georeferenced_tiff_1983_path=tiff_1983.resolve(),
    )

    model_raw = paths_data["model"]

    model_paths = ModelPaths(
        config_path=(base_dir / model_raw["config_path"]).resolve(),
        checkpoint_path=(base_dir / model_raw["checkpoint_path"]).resolve(),
    )

    results_raw = paths_data["results"]

    results_paths = ResultPaths(
        segmented_png_results_dir=(
            base_dir / results_raw["segmented_png_results_dir"]
        ).resolve(),
        shp_results_dir=(base_dir / results_raw["shp_results_dir"]).resolve(),
    )

    segmentation_raw = data["segmentation"]

    segmentation_config = SegmentationConfig(
        building_class_idx=segmentation_raw["building_class_idx"],
        buffer_distance=segmentation_raw["buffer_distance"],
    )

    return Config(
        paths=PathsConfig(data=data_paths, model=model_paths, results=results_paths),
        segmentation=segmentation_config,
    )
