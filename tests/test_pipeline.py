"""End-to-end pipeline with a stubbed model (mmsegmentation is never loaded)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

import image2shp.inference as inference_module
from image2shp import pipeline
from image2shp.config import load_config
from image2shp.processing import open_raster


def stub_inference(monkeypatch: pytest.MonkeyPatch, fake_logits, *, fail_for=None):
    """Replace windowed prediction with a deterministic stand-in."""

    def fake_predict(model, info, window_size=0):
        if fail_for is not None and info.path.name == fail_for:
            raise RuntimeError("simulated inference failure")
        return fake_logits(info)

    monkeypatch.setattr(inference_module, "predict_logits_from_info", fake_predict)
    monkeypatch.setattr(inference_module, "load_model", lambda *args, **kwargs: object())
    return fake_predict


# -- input discovery ---------------------------------------------------------


def test_discover_inputs_respects_patterns(tmp_path: Path, config):
    data_dir = config.paths.data_dir
    for name in ("a.tif", "b.png", "c.jpg", "notes.txt"):
        (data_dir / name).write_bytes(b"x")
    (data_dir / "sub").mkdir()
    (data_dir / "sub" / "nested.tif").write_bytes(b"x")

    found = {path.name for path in pipeline.discover_inputs(config)}
    assert {"a.tif", "b.png"}.issubset(found)  # in the config's patterns
    assert "c.jpg" not in found  # *.jpg is not in this test config
    assert "notes.txt" not in found  # no pattern matches
    assert "nested.tif" not in found  # recursive = false


def test_discover_inputs_recursive(config, tmp_path: Path):
    data_dir = config.paths.data_dir
    (data_dir / "sub").mkdir()
    (data_dir / "sub" / "nested.tif").write_bytes(b"x")

    import dataclasses

    recursive_config = load_config(config.source_path)
    object.__setattr__(
        recursive_config,
        "inputs",
        dataclasses.replace(recursive_config.inputs, recursive=True),
    )
    names = [path.name for path in pipeline.discover_inputs(recursive_config)]
    assert "nested.tif" in names


def test_discover_inputs_missing_directory(config):
    import dataclasses

    broken = load_config(config.source_path)
    object.__setattr__(
        broken,
        "paths",
        dataclasses.replace(broken.paths, data_dir=config.paths.data_dir / "nope"),
    )
    with pytest.raises(FileNotFoundError, match="data directory not found"):
        pipeline.discover_inputs(broken)


def test_output_paths_only_contains_enabled_outputs(config, tmp_path: Path):
    targets = pipeline.output_paths(config, Path("/tmp/map.tif"))
    assert set(targets) == {
        "segmented_png",
        "segmented_geotiff",
        "heatmap_geotiff",
        "shapefile",
    }
    assert targets["segmented_png"].name == "map.png"
    assert targets["heatmap_geotiff"] == (config.paths.results_dir / "heatmap" / "map.tif")


# -- single file processing --------------------------------------------------


def test_process_file_writes_every_output(monkeypatch: pytest.MonkeyPatch, config, fake_logits):
    stub_inference(monkeypatch, fake_logits)

    source = next(iter(pipeline.discover_inputs(config)))
    report = pipeline.process_file(object(), config, source)

    assert report.ok, report.error
    assert report.error is None
    assert {path.suffix for path in report.outputs} == {".png", ".tif", ".shp"}
    for path in report.outputs:
        assert path.is_file(), path

    # segmented geotiff keeps the input's georeferencing
    segmented = next(path for path in report.outputs if path.suffix == ".png")
    with rasterio.open(segmented.parent / "input.tif") as src:
        assert src.crs.to_epsg() == 2056

    # heatmap is a single float32 band scaled to [0, 1]
    heatmap = next(path for path in report.outputs if path.parts[-2] == "heatmap")
    with rasterio.open(heatmap) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert 0 <= src.read(1).min() <= src.read(1).max() <= 1

    # vector output is a readable shapefile with polygons
    shapefile = next(path for path in report.outputs if path.suffix == ".shp")
    assert shapefile.is_file()
    assert any("polygons" in note for note in report.notes)


def test_process_file_on_plain_photo_still_reports(
    monkeypatch: pytest.MonkeyPatch, config, plain_image: Path, fake_logits
):
    stub_inference(monkeypatch, fake_logits)
    plain_image.rename(config.paths.data_dir / plain_image.name)

    source = config.paths.data_dir / plain_image.name
    report = pipeline.process_file(object(), config, source)

    assert report.ok, report.error
    assert any("pixel coordinates" in note for note in report.notes)


def test_skip_existing_short_circuits(monkeypatch: pytest.MonkeyPatch, config, fake_logits):
    stub_inference(monkeypatch, fake_logits)
    source = next(iter(pipeline.discover_inputs(config)))

    first = pipeline.process_file(object(), config, source)
    assert first.ok

    calls: list[int] = []

    def counting_predict(*args, **kwargs):
        calls.append(1)
        return fake_logits(args[1])

    monkeypatch.setattr(inference_module, "predict_logits_from_info", counting_predict)

    second = pipeline.process_file(object(), config, source, skip_existing=True)
    assert second.ok
    assert calls == []
    assert any("already exist" in note for note in second.notes)


def test_process_file_records_errors(monkeypatch: pytest.MonkeyPatch, config, fake_logits):
    stub_inference(monkeypatch, fake_logits, fail_for="input.tif")
    source = next(iter(pipeline.discover_inputs(config)))

    report = pipeline.process_file(object(), config, source)
    assert not report.ok
    assert "simulated inference failure" in report.error
    assert report.outputs == []


# -- whole runs --------------------------------------------------------------


def test_run_uses_a_single_model_and_reports(monkeypatch: pytest.MonkeyPatch, config, fake_logits):
    loaded: list[tuple] = []
    stub_inference(monkeypatch, fake_logits)
    monkeypatch.setattr(
        inference_module,
        "load_model",
        lambda *args, **kwargs: loaded.append(args) or object(),
    )

    files = pipeline.discover_inputs(config)
    report = pipeline.run(config, files, device="cpu")

    assert report.ok
    assert report.succeeded == len(files)
    assert len(loaded) == 1  # model loaded once for the whole batch
    assert "succeeded" in report.summary()


def test_run_continues_after_a_failing_file(monkeypatch: pytest.MonkeyPatch, config, fake_logits):
    data_dir = config.paths.data_dir
    (data_dir / "second.tif").write_bytes((data_dir / "input.tif").read_bytes())

    stub_inference(monkeypatch, fake_logits, fail_for="input.tif")
    report = pipeline.run(config, pipeline.discover_inputs(config), model=object())

    assert not report.ok
    assert report.failed == 1
    assert report.succeeded == 1
    assert {f.path.name for f in report.files} == {"input.tif", "second.tif"}


def test_run_rejects_output_name_collisions(config, fake_logits, monkeypatch):
    stub_inference(monkeypatch, fake_logits)
    data_dir = config.paths.data_dir
    (data_dir / "input.png").write_bytes(b"")

    files = [data_dir / "input.tif", data_dir / "input.png"]
    report = pipeline.run(config, files, model=object())

    assert not report.ok
    collision = next(f for f in report.files if not f.ok)
    assert "collision" in collision.error


def test_run_creates_results_directory(monkeypatch, config, fake_logits):
    stub_inference(monkeypatch, fake_logits)
    results = config.paths.results_dir
    assert not results.exists()

    pipeline.run(config, pipeline.discover_inputs(config), model=object())
    assert results.is_dir()


def test_injected_model_skips_loading(monkeypatch, config, fake_logits):
    stub_inference(monkeypatch, fake_logits)

    def explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("load_model must not be called when a model is given")

    monkeypatch.setattr(inference_module, "load_model", explode)
    report = pipeline.run(config, pipeline.discover_inputs(config), model=object())
    assert report.ok


def test_labels_drive_the_target_class(monkeypatch, config):
    """The configured target class (not a hardcoded index) picks the polygons."""

    def only_class_1(info, *args, **kwargs):
        logits = np.full((3, info.height, info.width), -5.0, dtype=np.float32)
        logits[1] = 5.0
        return logits

    stub_inference(monkeypatch, only_class_1)
    source = next(iter(pipeline.discover_inputs(config)))

    # target_class = "built" (index 2) -> nothing selected -> no polygons
    report = pipeline.process_file(object(), config, source)
    assert not report.ok
    assert "no polygons" in report.error

    import dataclasses

    retargeted = load_config(config.source_path)
    object.__setattr__(
        retargeted,
        "segmentation",
        dataclasses.replace(retargeted.segmentation, target_class="contours"),
    )
    report = pipeline.process_file(object(), retargeted, source)
    assert report.ok, report.error


def test_open_raster_matches_the_pipeline_view(monkeypatch, config, fake_logits):
    stub_inference(monkeypatch, fake_logits)
    source = next(iter(pipeline.discover_inputs(config)))
    info = open_raster(source)
    assert (info.height, info.width) == (64, 48)
