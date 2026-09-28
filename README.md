# image2shp

Segment map images with an [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) model and turn the result into things you can actually use: colored class maps, a georeferenced heatmap, and vector polygons.

Typical use: you have a trained model (a `.py` config + a `.pth` checkpoint) and a folder of scanned or georeferenced maps, and you want building footprints as a shapefile plus a QGIS-friendly density layer.

## Sources

This project builds on the paper and pretrained model of Rémi Petitpierre
(EPFL), and on OpenMMLab's mmsegmentation:

- **Paper** — Rémi Petitpierre, *Generalizable Multiscale Segmentation of
  Heterogeneous Map Collections* (2026): <https://arxiv.org/abs/2603.05037>
- **Model & dataset (Semap)** — Rémi Petitpierre, Damien Gomez Donoso and Ben
  Kriesel, EPFL: <https://zenodo.org/records/19048095>
- **Library** — OpenMMLab, *MMSegmentation*: <https://github.com/open-mmlab/mmsegmentation>

## What it produces

For every image found in `data/`:

| output | file | purpose |
|---|---|---|
| segmented class map | `results/segmented/<name>.png` | quick look, palette from `config.toml` |
| segmented class map | `results/segmented/<name>.tif` | same, georeferenced for GIS |
| class heatmap | `results/heatmap/<name>.tif` | float32 Gaussian density of the target class, drop straight into QGIS |
| vector polygons | `results/vector/<name>.shp` | polygons of the target class, georeferenced (`.shp/.shx/.dbf/.prj`) |

## Quick start (Docker)

```bash
# 1. inputs: any .tif/.tiff/.png/.jpg (see [inputs] patterns)
cp your_maps/*.tif data/

# 2. model: the mmsegmentation config and the trained checkpoint
cp /path/to/mask2former_....py model/
cp /path/to/best_mIoU_iter_*.pth model/

# 3. tell the tool which model to use and which class/colours to render
$EDITOR config.toml

# 4. check the plan without loading the model
docker compose run --rm image2shp --dry-run

# 5. run it
docker compose run --rm image2shp
```

`./data`, `./model` and `./config.toml` are mounted read-only; everything is written to `./results`.

Requirements: Docker with the compose plugin. Add the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and uncomment the `deploy` block in `docker-compose.yml` to use a GPU (`[segmentation] device = "auto"` picks it up).

## Configuration

One file, `config.toml`, used identically on the host and in the container: **all paths are relative to the file itself**, and the container's layout mirrors the repository (`/app/data`, `/app/model`, `/app/results`). That is why there is no separate "docker config" any more.

```toml
[paths]
data_dir = "data"
results_dir = "results"

[model]
config     = "model/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py"
checkpoint = "model/best_mIoU_iter_138828.pth"

[segmentation]
target_class   = "built"   # class used for the heatmap and the shapefile
device         = "auto"    # auto | cpu | cuda
window_size    = 2048      # tiled inference for large rasters (0 = whole image)
heatmap_sigma  = 25
buffer_distance = 20

[[classes]]                # index, name and colour are all yours to change
index = 2
name  = "built"
color = "#00ff00"
```

Every key, its type and its default are documented in [doc/configuration.md](doc/configuration.md).

## Command line

```bash
python -m image2shp                     # every input matching [inputs] patterns
python -m image2shp data/map.tif        # specific files
python -m image2shp --dry-run           # validate config + print the plan, no model
python -m image2shp --device cpu        # override [segmentation] device
python -m image2shp --skip-existing     # resume an interrupted run
python -m image2shp --limit 2 -v        # first two images, info logging
```

Exit codes: `0` success, `1` at least one image failed, `2` configuration problem, `3` mmsegmentation not installed.

## Running without Docker

Not needed for the pipeline itself, but handy for development:

```bash
uv sync                       # core dependencies + dev tools
uv sync --extra inference     # ... + torch/mmsegmentation (heavy)
uv run python -m image2shp --dry-run
```

## Development

```bash
uv run pytest                 # 80+ smoke tests, none of them loads a model
uv run ruff format src tests
uv run ruff check src tests
```

The test suite deliberately stubs out inference (`tests/test_pipeline.py`, `tests/test_cli.py`), so it runs on a machine without torch or mmsegmentation: it exercises configuration loading, raster IO, palette rendering, vectorization and the CLI end to end against small synthetic GeoTIFFs.

## Repository layout

```
config.toml        single configuration file (tracked, safe to edit)
docker-compose.yml mounts ./data ./model ./results and runs the CLI
Dockerfile         image with mmsegmentation + mmdetection
data/              input images            (ignored by git)
model/             model .py + .pth        (ignored by git)
results/           outputs                 (ignored by git)
src/image2shp/     the library
  config.py        config.toml loader + validation
  processing.py    raster IO, normalisation, heatmaps
  colors.py        label map -> RGB using the configured palette
  vectorize.py     mask -> polygons -> shapefile
  inference.py     the only module that touches torch/mmseg (lazy)
  pipeline.py      discover inputs, run, write outputs
  cli.py           argparse entry point
tests/             pytest smoke tests + small fixtures
doc/               architecture, configuration, usage, models
old/               archived legacy workflow (not imported; see its README)
```

## Documentation

- [doc/architecture.md](doc/architecture.md) — data flow, module responsibilities, design decisions and alternatives
- [doc/configuration.md](doc/configuration.md) — every `config.toml` key
- [doc/usage.md](doc/usage.md) — Docker and native usage, CLI reference, troubleshooting
- [doc/models.md](doc/models.md) — putting a new mmsegmentation model into `model/`

## Status

The pipeline, configuration and tests are verified. The `Dockerfile` could **not** be rebuilt during the generalization work (no Docker daemon was available): it carries the original base image (`pytorch/pytorch:1.11.0`) and clones mmsegmentation/mmdetection from `main`, so if `pip install .` fails on your machine, bump the `PYTORCH`/`CUDA`/`MMCV` build args as described in [doc/usage.md](doc/usage.md).
