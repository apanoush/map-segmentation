"""CLI smoke tests: argument handling, dry-run and exit codes."""

from __future__ import annotations

from pathlib import Path

import pytest

import image2shp.inference as inference_module
from image2shp.cli import (
    EXIT_BACKEND,
    EXIT_CONFIG,
    EXIT_FAILURE,
    EXIT_OK,
    build_parser,
    main,
)
from image2shp.inference import MissingBackendError
from tests.test_pipeline import stub_inference


def test_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0


def test_version_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0


def test_dry_run_prints_the_plan(config_path: Path, capsys):
    code = main(["--dry-run", "--config", str(config_path)])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "config        :" in out
    assert "target class  : built" in out
    assert "images (1):" in out
    assert "-> shapefile:" in out


def test_dry_run_reports_missing_model_files_without_failing(tmp_path: Path, geotiff: Path, capsys):
    from tests.conftest import write_config

    config_file = write_config(
        tmp_path,
        data_dir=geotiff.parent,
        results_dir=tmp_path / "results",
        model_config=tmp_path / "missing.py",
        checkpoint=tmp_path / "missing.pth",
    )
    code = main(["--dry-run", "--config", str(config_file)])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "warning: model config not found" in out
    assert "warning: model checkpoint not found" in out


def test_missing_config_file_exits_with_config_code(tmp_path: Path, capsys):
    code = main(["--config", str(tmp_path / "absent.toml")])
    assert code == EXIT_CONFIG
    assert "config file not found" in capsys.readouterr().out


def test_broken_config_exits_with_config_code(tmp_path: Path, capsys):
    broken = tmp_path / "broken.toml"
    broken.write_text("[paths\n", encoding="utf-8")
    code = main(["--config", str(broken)])
    assert code == EXIT_CONFIG
    assert "not valid TOML" in capsys.readouterr().out


def test_problems_block_a_real_run(tmp_path: Path, geotiff: Path, capsys):
    from tests.conftest import write_config

    config_file = write_config(
        tmp_path,
        data_dir=geotiff.parent,
        results_dir=tmp_path / "results",
        model_config=tmp_path / "missing.py",
        checkpoint=tmp_path / "missing.pth",
    )
    code = main(["--config", str(config_file)])
    out = capsys.readouterr().out

    assert code == EXIT_CONFIG
    assert "warning: model config not found" in out
    assert "problem(s) in" in out


def test_no_matching_inputs_exits_with_failure(config_path: Path, capsys, tmp_path):
    from tests.conftest import write_config

    empty = tmp_path / "empty"
    empty.mkdir()
    model = tmp_path / "m.py"
    checkpoint = tmp_path / "m.pth"
    model.write_text("# stub\n", encoding="utf-8")
    checkpoint.write_bytes(b"stub")
    config_file = write_config(
        tmp_path,
        data_dir=empty,
        results_dir=tmp_path / "results",
        model_config=model,
        checkpoint=checkpoint,
    )
    code = main(["--config", str(config_file)])
    assert code == EXIT_FAILURE
    assert "no inputs matched" in capsys.readouterr().out


def test_missing_explicit_image_exits_with_failure(config_path: Path, capsys):
    code = main(["--config", str(config_path), "does-not-exist.tif"])
    assert code == EXIT_FAILURE
    assert "no such image" in capsys.readouterr().out


def test_missing_backend_exits_with_backend_code(
    monkeypatch: pytest.MonkeyPatch, config_path: Path, capsys
):
    def raise_backend(*args, **kwargs):
        raise MissingBackendError("mmsegmentation (and torch) are required")

    monkeypatch.setattr(inference_module, "load_model", raise_backend)

    code = main(["--config", str(config_path)])
    assert code == EXIT_BACKEND
    assert "required" in capsys.readouterr().out


def test_full_run_with_stubbed_model(
    monkeypatch: pytest.MonkeyPatch, config_path: Path, config, capsys, fake_logits
):
    stub_inference(monkeypatch, fake_logits)

    code = main(["--config", str(config_path), "-v"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "1 succeeded, 0 failed" in out
    results = config.paths.results_dir
    assert (results / "segmented" / "input.png").is_file()
    assert (results / "heatmap" / "input.tif").is_file()
    assert (results / "vector" / "input.shp").is_file()


def test_limit_and_explicit_files(
    monkeypatch: pytest.MonkeyPatch, config_path: Path, config, fake_logits, tmp_path
):
    stub_inference(monkeypatch, fake_logits)
    second = config.paths.data_dir / "second.tif"
    second.write_bytes(next(iter(config.paths.data_dir.iterdir())).read_bytes())

    assert main(["--config", str(config_path), "--limit", "1"]) == EXIT_OK
    assert not (config.paths.results_dir / "vector" / "second.shp").exists()

    code = main(["--config", str(config_path), str(second)])
    assert code == EXIT_OK
    assert (config.paths.results_dir / "vector" / "second.shp").exists()


def test_skip_existing_option(
    monkeypatch: pytest.MonkeyPatch, config_path: Path, config, fake_logits
):
    stub_inference(monkeypatch, fake_logits)
    assert main(["--config", str(config_path)]) == EXIT_OK

    calls: list[int] = []

    def counting(*args, **kwargs):
        calls.append(1)
        return fake_logits(args[1])

    monkeypatch.setattr(inference_module, "predict_logits_from_info", counting)
    assert main(["--config", str(config_path), "--skip-existing"]) == EXIT_OK
    assert calls == []


def test_device_override_is_passed_through(
    monkeypatch: pytest.MonkeyPatch, config_path: Path, config, fake_logits
):
    seen: dict[str, str | None] = {}

    def fake_load(*args, **kwargs):
        seen["device"] = kwargs.get("device") if "device" in kwargs else args[-1]
        return object()

    def fake_predict(model, info, window_size=0):
        return fake_logits(info)

    monkeypatch.setattr(inference_module, "load_model", fake_load)
    monkeypatch.setattr(inference_module, "predict_logits_from_info", fake_predict)

    assert main(["--config", str(config_path), "--device", "cpu"]) == EXIT_OK
    assert seen["device"] == "cpu"
