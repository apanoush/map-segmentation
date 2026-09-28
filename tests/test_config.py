"""Configuration loading, validation and class palette handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from image2shp.config import (
    ClassSpec,
    ConfigError,
    load_config,
)
from tests.conftest import write_config


def test_load_config_resolves_paths_relative_to_the_file(config_path: Path):
    config = load_config(config_path)

    assert config.source_path == config_path.resolve()
    assert config.paths.data_dir.is_absolute()
    assert config.paths.results_dir.is_absolute()
    assert config.model.config_path.is_file()
    assert config.model.checkpoint_path.is_file()
    assert config.segmentation.device == "cpu"
    assert config.segmentation.window_size == 32
    assert config.segmentation.target_class == "built"
    assert config.segmentation.heatmap_sigma == 3
    assert config.segmentation.buffer_distance == 2
    assert config.inputs.patterns == ("*.tif", "*.tiff", "*.png")


def test_default_config_has_no_problems(config_path: Path):
    config = load_config(config_path)
    assert config.problems() == []


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml_raises(tmp_path: Path):
    broken = tmp_path / "broken.toml"
    broken.write_text("[paths\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(broken)


def test_missing_model_section_raises(tmp_path: Path, geotiff: Path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[paths]\ndata_dir = "."\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\[model\]"):
        load_config(config_file)


def test_missing_model_key_raises(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[model]\nconfig = "model.py"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="checkpoint"):
        load_config(config_file)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("segmentation", "device", '"gpu"', "device must be one of"),
        ("segmentation", "window_size", "-1", "window_size must be >= 0"),
        ("segmentation", "buffer_distance", "-3", "buffer_distance must be >= 0"),
        ("segmentation", "heatmap_sigma", "0", "heatmap_sigma must be > 0"),
    ],
)
def test_invalid_values_raise(
    tmp_path: Path, geotiff: Path, section: str, key: str, value: str, match: str
):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        f"""
[paths]
data_dir = "{data_dir}"
results_dir = "{tmp_path / "results"}"
[model]
config = "{tmp_path / "m.py"}"
checkpoint = "{tmp_path / "m.pth"}"
[{section}]
{key} = {value}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=match):
        load_config(config_file)


def test_unknown_target_class_is_reported(tmp_path: Path, geotiff: Path):
    config_file = write_config(
        tmp_path,
        data_dir=geotiff.parent,
        results_dir=tmp_path / "results",
        model_config=tmp_path / "m.py",
        checkpoint=tmp_path / "m.pth",
    )
    text = config_file.read_text(encoding="utf-8").replace(
        'target_class = "built"', 'target_class = "castle"'
    )
    config_file.write_text(text, encoding="utf-8")

    config = load_config(config_file)
    problems = "\n".join(config.problems())
    assert "castle" in problems
    assert "built" in problems  # lists the known classes


def test_missing_model_files_are_reported(tmp_path: Path, geotiff: Path):
    config_file = write_config(
        tmp_path,
        data_dir=geotiff.parent,
        results_dir=tmp_path / "results",
        model_config=tmp_path / "missing.py",
        checkpoint=tmp_path / "missing.pth",
    )
    problems = load_config(config_file).problems()
    assert any("model config not found" in problem for problem in problems)
    assert any("model checkpoint not found" in problem for problem in problems)


def test_duplicate_class_indices_are_reported(tmp_path: Path, geotiff: Path):
    config_file = write_config(
        tmp_path,
        data_dir=geotiff.parent,
        results_dir=tmp_path / "results",
        model_config=tmp_path / "m.py",
        checkpoint=tmp_path / "m.pth",
        extra="""
[[classes]]
index = 2
name = "castle"
color = "#ff00ff"
""",
    )
    problems = load_config(config_file).problems()
    assert any("duplicate class indices" in problem for problem in problems)


def test_disabled_outputs_are_reported(tmp_path: Path, geotiff: Path):
    config_file = write_config(
        tmp_path,
        data_dir=geotiff.parent,
        results_dir=tmp_path / "results",
        model_config=tmp_path / "m.py",
        checkpoint=tmp_path / "m.pth",
    )
    config_file.write_text(
        config_file.read_text(encoding="utf-8")
        .replace("[outputs]\n", "[outputs]\n")
        .replace("segmented_png = true", "segmented_png = false")
        .replace("segmented_geotiff = true", "segmented_geotiff = false")
        .replace("heatmap_geotiff = true", "heatmap_geotiff = false")
        .replace("shapefile = true", "shapefile = false"),
        encoding="utf-8",
    )
    problems = load_config(config_file).problems()
    assert any("disables every output" in problem for problem in problems)


def test_class_lookup_helpers(config):
    spec = config.class_by_name("contours")
    assert spec == ClassSpec(index=1, name="contours", color=(0, 0, 255))
    assert config.class_by_index(2).name == "built"
    assert config.class_by_index(99) is None
    assert config.target_class_spec.index == 2
    assert config.target_class_spec.hex_color == "#00ff00"

    with pytest.raises(ConfigError, match="known classes"):
        config.class_by_name("nope")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("#00ff00", (0, 255, 0)),
        ("#f00", (255, 0, 0)),
        ("ff00ff", (255, 0, 255)),
    ],
)
def test_color_parsing(tmp_path: Path, raw: str, expected: tuple[int, int, int]):
    config_file = write_config(
        tmp_path,
        data_dir=tmp_path,
        results_dir=tmp_path / "out",
        model_config=tmp_path / "m.py",
        checkpoint=tmp_path / "m.pth",
    )
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace('"#00ff00"', f'"{raw}"'),
        encoding="utf-8",
    )
    assert load_config(config_file).class_by_name("built").color == expected


def test_invalid_color_raises(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        f"""
[model]
config = "{tmp_path / "m.py"}"
checkpoint = "{tmp_path / "m.pth"}"
[[classes]]
index = 0
name = "background"
color = "#zz0000"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not a #RRGGBB"):
        load_config(config_file)


def test_shipped_config_parses():
    """The config committed at the repository root must stay loadable."""
    repo_config = Path(__file__).resolve().parents[1] / "config.toml"
    config = load_config(repo_config)

    assert len(config.classes) == 6
    assert config.target_class_spec.name == "built"
    assert config.paths.data_dir.name == "data"
    assert config.model.checkpoint_path.name.endswith(".pth")
    # Only "file not found" issues are acceptable in a fresh checkout.
    assert all("not found" in problem for problem in config.problems())
