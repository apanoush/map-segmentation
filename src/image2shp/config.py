"""Configuration loading for image2shp.

The configuration lives in a single ``config.toml`` whose paths are relative to
the file itself (the repository root). See ``doc/configuration.md``.
"""

from __future__ import annotations

try:  # stdlib since Python 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAME = "config.toml"

RGB = tuple[int, int, int]


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ClassSpec:
    """One segmentable class as described by ``[[classes]]``."""

    index: int
    name: str
    color: RGB

    @property
    def hex_color(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.color)


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class ModelPaths:
    config_path: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class InputConfig:
    patterns: tuple[str, ...]
    recursive: bool


@dataclass(frozen=True)
class SegmentationConfig:
    device: str
    window_size: int
    target_class: str
    heatmap_sigma: float
    buffer_distance: int


@dataclass(frozen=True)
class OutputsConfig:
    segmented_png: bool
    segmented_geotiff: bool
    heatmap_geotiff: bool
    shapefile: bool


@dataclass(frozen=True)
class Config:
    """Fully resolved configuration."""

    paths: PathsConfig
    model: ModelPaths
    inputs: InputConfig
    segmentation: SegmentationConfig
    outputs: OutputsConfig
    classes: tuple[ClassSpec, ...]
    source_path: Path

    # -- class lookup helpers -------------------------------------------------

    def class_by_name(self, name: str) -> ClassSpec:
        for spec in self.classes:
            if spec.name == name:
                return spec
        raise ConfigError(
            f"Unknown class name {name!r}; known classes: "
            f"{', '.join(sorted(spec.name for spec in self.classes))}"
        )

    def class_by_index(self, index: int) -> ClassSpec | None:
        for spec in self.classes:
            if spec.index == index:
                return spec
        return None

    @property
    def target_class_spec(self) -> ClassSpec:
        return self.class_by_name(self.segmentation.target_class)

    # -- validation -----------------------------------------------------------

    def problems(self) -> list[str]:
        """Return a list of human readable problems (empty when usable).

        This never raises: the CLI uses it to print a friendly report before
        failing, and tests use it to check that a config is *consistent*
        without requiring the model files to exist.
        """
        issues: list[str] = []

        if not self.paths.data_dir.is_dir():
            issues.append(f"data directory not found: {self.paths.data_dir}")
        if not self.model.config_path.is_file():
            issues.append(f"model config not found: {self.model.config_path}")
        if not self.model.checkpoint_path.is_file():
            issues.append(f"model checkpoint not found: {self.model.checkpoint_path}")
        if not self.classes:
            issues.append("no classes defined ([[classes]] tables)")

        indices = [spec.index for spec in self.classes]
        if len(indices) != len(set(indices)):
            issues.append(f"duplicate class indices: {sorted(indices)}")

        names = [spec.name for spec in self.classes]
        if len(names) != len(set(names)):
            issues.append(f"duplicate class names: {sorted(names)}")

        if self.classes:
            try:
                self.class_by_name(self.segmentation.target_class)
            except ConfigError as exc:
                issues.append(str(exc))

        if not any(
            (
                self.outputs.segmented_png,
                self.outputs.segmented_geotiff,
                self.outputs.heatmap_geotiff,
                self.outputs.shapefile,
            )
        ):
            issues.append("[outputs] disables every output; nothing would be written")

        return issues


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "paths": {"data_dir": "data", "results_dir": "results"},
    "inputs": {
        "patterns": ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"],
        "recursive": False,
    },
    "segmentation": {
        "device": "auto",
        "window_size": 2048,
        "target_class": "built",
        "heatmap_sigma": 25.0,
        "buffer_distance": 20,
    },
    "outputs": {
        "segmented_png": True,
        "segmented_geotiff": True,
        "heatmap_geotiff": True,
        "shapefile": True,
    },
}


def _section(data: dict[str, Any], name: str, *, required: bool = True) -> dict[str, Any]:
    value = data.get(name)
    if value is None:
        if required:
            raise ConfigError(f"missing [{name}] section")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _value(section: dict[str, Any], defaults: dict[str, Any], key: str, kind: type) -> Any:
    raw = section.get(key, defaults.get(key))
    if raw is None:
        raise ConfigError(f"missing required key {key!r}")
    if kind is float and isinstance(raw, int) and not isinstance(raw, bool):
        return float(raw)
    if not isinstance(raw, kind) or isinstance(raw, bool) and kind is not bool:
        raise ConfigError(f"key {key!r} must be of type {kind.__name__}, got {type(raw).__name__}")
    return raw


def _resolve(base_dir: Path, raw: str) -> Path:
    """Resolve ``raw`` relative to ``base_dir`` (absolute/``~`` paths win)."""
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded
    return (base_dir / expanded).resolve()


def _parse_color(raw: str, *, where: str) -> RGB:
    text = raw.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ConfigError(f"{where}: color {raw!r} is not a #RRGGBB hex value")
    try:
        r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise ConfigError(f"{where}: color {raw!r} is not a #RRGGBB hex value") from exc
    return (r, g, b)


def _parse_classes(raw: Iterable[Any]) -> tuple[ClassSpec, ...]:
    specs: list[ClassSpec] = []
    for position, entry in enumerate(raw):
        where = f"[[classes]] #{position + 1}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a table")
        for key in ("index", "name", "color"):
            if key not in entry:
                raise ConfigError(f"{where} is missing required key {key!r}")
        index = entry["index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ConfigError(f"{where}: index must be a non-negative integer")
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{where}: name must be a non-empty string")
        color = entry["color"]
        if not isinstance(color, str):
            raise ConfigError(f'{where}: color must be a string such as "#ff00ff"')
        specs.append(ClassSpec(index=index, name=name, color=_parse_color(color, where=where)))
    return tuple(sorted(specs, key=lambda spec: spec.index))


def load_config(path: str | Path = DEFAULT_CONFIG_NAME) -> Config:
    """Load and resolve ``path``.

    Raises:
        ConfigError: if the file is missing, unparsable, or inconsistent.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(
            f"config file not found: {config_path} "
            f"(create one from the template at the repository root)"
        )
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{config_path} is not valid TOML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"{config_path}: top level must be tables")

    base_dir = config_path.expanduser().resolve().parent

    paths_raw = _section(data, "paths", required=False)
    paths = PathsConfig(
        data_dir=_resolve(base_dir, _value(paths_raw, _DEFAULTS["paths"], "data_dir", str)),
        results_dir=_resolve(base_dir, _value(paths_raw, _DEFAULTS["paths"], "results_dir", str)),
    )

    model_raw = _section(data, "model")
    for key in ("config", "checkpoint"):
        if key not in model_raw:
            raise ConfigError(f"[model] is missing required key {key!r}")
    model = ModelPaths(
        config_path=_resolve(base_dir, str(model_raw["config"])),
        checkpoint_path=_resolve(base_dir, str(model_raw["checkpoint"])),
    )

    inputs_raw = _section(data, "inputs", required=False)
    patterns_raw = _value(inputs_raw, _DEFAULTS["inputs"], "patterns", list)
    patterns: list[str] = []
    for pattern in patterns_raw:
        if not isinstance(pattern, str) or not pattern:
            raise ConfigError("[inputs] patterns must be non-empty strings")
        patterns.append(pattern)
    inputs = InputConfig(
        patterns=tuple(patterns),
        recursive=bool(_value(inputs_raw, _DEFAULTS["inputs"], "recursive", bool)),
    )

    segmentation_raw = _section(data, "segmentation", required=False)
    defaults = _DEFAULTS["segmentation"]
    target_class = _value(segmentation_raw, defaults, "target_class", str)
    window_size = _value(segmentation_raw, defaults, "window_size", int)
    buffer_distance = _value(segmentation_raw, defaults, "buffer_distance", int)
    device = _value(segmentation_raw, defaults, "device", str)
    heatmap_sigma = _value(segmentation_raw, defaults, "heatmap_sigma", float)
    if window_size < 0:
        raise ConfigError("[segmentation] window_size must be >= 0")
    if buffer_distance < 0:
        raise ConfigError("[segmentation] buffer_distance must be >= 0")
    if heatmap_sigma <= 0:
        raise ConfigError("[segmentation] heatmap_sigma must be > 0")
    if device not in ("auto", "cpu", "cuda"):
        raise ConfigError("[segmentation] device must be one of: auto, cpu, cuda")
    segmentation = SegmentationConfig(
        device=device,
        window_size=window_size,
        target_class=target_class,
        heatmap_sigma=heatmap_sigma,
        buffer_distance=buffer_distance,
    )

    outputs_raw = _section(data, "outputs", required=False)
    outputs_defaults = _DEFAULTS["outputs"]
    outputs = OutputsConfig(
        segmented_png=bool(_value(outputs_raw, outputs_defaults, "segmented_png", bool)),
        segmented_geotiff=bool(_value(outputs_raw, outputs_defaults, "segmented_geotiff", bool)),
        heatmap_geotiff=bool(_value(outputs_raw, outputs_defaults, "heatmap_geotiff", bool)),
        shapefile=bool(_value(outputs_raw, outputs_defaults, "shapefile", bool)),
    )

    classes = _parse_classes(data.get("classes", []))

    return Config(
        paths=paths,
        model=model,
        inputs=inputs,
        segmentation=segmentation,
        outputs=outputs,
        classes=classes,
        source_path=config_path.expanduser().resolve(),
    )
