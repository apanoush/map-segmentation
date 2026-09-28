"""Command line interface: ``python -m image2shp`` (or the ``image2shp`` script)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from image2shp import __version__, pipeline
from image2shp.config import Config, ConfigError, load_config
from image2shp.inference import MissingBackendError

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
EXIT_BACKEND = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image2shp",
        description=(
            "Segment map images with an mmsegmentation model and export colored "
            "rasters, heatmaps and vector polygons."
        ),
        epilog=(
            "examples:\n"
            "  image2shp                     process every input in [paths] data_dir\n"
            "  image2shp data/map.tif         process specific images\n"
            "  image2shp --dry-run            validate the config and list the plan\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="images to process (default: everything matching [inputs] patterns)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="configuration file (default: ./config.toml)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="override [segmentation] device",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved configuration and the processing plan, load no model",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip files whose outputs are all already present",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="process at most N images"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v for info, -vv for debug logging",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _print_config(config: Config, files: Sequence[Path]) -> None:
    enabled = [
        name
        for name, on in (
            ("segmented PNG", config.outputs.segmented_png),
            ("segmented GeoTIFF", config.outputs.segmented_geotiff),
            ("heatmap GeoTIFF", config.outputs.heatmap_geotiff),
            ("shapefile", config.outputs.shapefile),
        )
        if on
    ]
    print(f"config        : {config.source_path}")
    print(f"data dir      : {config.paths.data_dir}")
    print(f"results dir   : {config.paths.results_dir}")
    print(f"model config  : {config.model.config_path}")
    print(f"checkpoint    : {config.model.checkpoint_path}")
    print(
        f"device        : {config.segmentation.device}, window {config.segmentation.window_size}px"
    )
    print(
        f"target class  : {config.segmentation.target_class} "
        f"(index {config.target_class_spec.index}, "
        f"color {config.target_class_spec.hex_color})"
    )
    print(
        f"buffer/sigma  : {config.segmentation.buffer_distance}px / "
        f"{config.segmentation.heatmap_sigma}px"
    )
    print(f"outputs       : {', '.join(enabled) if enabled else 'none'}")
    print(
        "classes       : "
        + ", ".join(f"{spec.index}={spec.name} {spec.hex_color}" for spec in config.classes)
    )
    print(f"images ({len(files)}):")
    if not files:
        print("  (none)")
    for path in files:
        targets = pipeline.output_paths(config, path)
        print(f"  {path}")
        for kind, target in targets.items():
            print(f"    -> {kind}: {target}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", flush=True)
        return EXIT_CONFIG

    problems = config.problems()
    if problems:
        for problem in problems:
            print(f"warning: {problem}", flush=True)
        if not args.dry_run:
            print(
                f"error: {len(problems)} problem(s) in {config.source_path}; "
                "fix them or run with --dry-run to inspect the configuration",
                flush=True,
            )
            return EXIT_CONFIG

    if args.images:
        files: list[Path] = []
        missing = False
        for image in args.images:
            if not image.is_file():
                print(f"error: no such image: {image}", flush=True)
                missing = True
            else:
                files.append(image.resolve())
        if missing:
            return EXIT_FAILURE
    else:
        try:
            files = pipeline.discover_inputs(config)
        except FileNotFoundError as exc:
            print(f"error: {exc}", flush=True)
            return EXIT_CONFIG

    if args.limit is not None:
        files = files[: max(args.limit, 0)]

    if args.dry_run:
        _print_config(config, files)
        return EXIT_OK

    if not files:
        print(
            f"error: no inputs matched in {config.paths.data_dir} "
            f"(patterns: {', '.join(config.inputs.patterns)})",
            flush=True,
        )
        return EXIT_FAILURE

    try:
        report = pipeline.run(
            config,
            files,
            device=args.device,
            skip_existing=args.skip_existing,
        )
    except MissingBackendError as exc:
        print(f"error: {exc}", flush=True)
        return EXIT_BACKEND

    print(f"\n{report.summary()}")
    for file_report in report.files:
        if file_report.ok:
            continue
        print(f"  failed: {file_report.path}: {file_report.error}")
    return EXIT_OK if report.ok else EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
