# Usage

## Docker (recommended)

The image contains mmsegmentation, mmdetection, mmcv and the `image2shp`
package. Inputs, model and outputs are bind-mounted, so rebuilding is only
needed when the code changes.

```bash
# build once
docker compose build

# validate the configuration and print the plan (loads no model)
docker compose run --rm image2shp --dry-run

# process every input in data/
docker compose run --rm image2shp

# a single file, on the CPU, with progress logs
docker compose run --rm image2shp -v data/1907_decoupe.tif --device cpu

# resume an interrupted run
docker compose run --rm image2shp --skip-existing
```

`docker-compose.yml` sets `entrypoint: ["python", "-m", "image2shp"]`, so any
flag you type after the service name goes straight to the CLI. To get a shell
instead: `docker compose run --rm --entrypoint bash image2shp`.

### Mounts

| host | container | mode |
|---|---|---|
| `./data` | `/app/data` | read-only |
| `./model` | `/app/model` | read-only |
| `./config.toml` | `/app/config.toml` | read-only (edit on the host, no rebuild) |
| `./results` | `/app/results` | read-write |

### GPU

1. install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
2. uncomment the `deploy:` block at the bottom of `docker-compose.yml`;
3. leave `[segmentation] device = "auto"` (or set `"cuda"`).

`nvidia-smi` inside the container confirms it:
`docker compose run --rm --entrypoint bash image2shp -c "nvidia-smi"`.

### Plain `docker run`

Equivalent, without compose:

```bash
docker build -t image2shp:latest .

docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/model:/app/model:ro" \
  -v "$PWD/config.toml:/app/config.toml:ro" \
  -v "$PWD/results:/app/results" \
  -w /app image2shp:latest python -m image2shp --dry-run
```

(the image has no `ENTRYPOINT`; the compose file supplies it)

## Without Docker

Only needed to develop or to run the pipeline natively:

```bash
uv sync                    # core dependencies + pytest/ruff (dev group)
uv sync --extra inference  # adds torch + mmsegmentation (large)
uv run python -m image2shp --dry-run
```

With plain pip instead of uv:

```bash
pip install -e .
pip install -e ".[inference]"   # optional
pip install pytest ruff
```

Without the `inference` extra the CLI still works for `--dry-run`, `--help` and
`--version`; a real run stops with exit code `3` and an explanation.

## CLI reference

```
usage: image2shp [-h] [-c CONFIG] [--device {auto,cpu,cuda}] [--dry-run]
                 [--skip-existing] [--limit N] [-v] [--version]
                 [images ...]
```

| option | effect |
|---|---|
| `images...` | explicit files to process (default: everything matching `[inputs] patterns` in `data_dir`) |
| `-c, --config` | configuration file (default `./config.toml`) |
| `--device` | override `[segmentation] device` |
| `--dry-run` | print resolved configuration + per-file output plan, load no model |
| `--skip-existing` | skip a file when all of its outputs are already on disk |
| `--limit N` | process at most `N` images |
| `-v` / `-vv` | info / debug logging |
| `--version` | print the version and exit |

### Exit codes

| code | meaning |
|---|---|
| `0` | every image succeeded (or a successful `--dry-run`) |
| `1` | at least one image failed, no input matched, or a named file does not exist |
| `2` | configuration problem (unparsable, inconsistent, or missing files on a real run) |
| `3` | mmsegmentation/torch not installed |

A failing file never aborts the batch: the run continues and the summary lists
every failure at the end.

## Outputs

```
results/
├── segmented/<name>.png    class map, palette from [[classes]]
├── segmented/<name>.tif    same, georeferenced (input's CRS + transform)
├── heatmap/<name>.tif      float32 density of target_class, [0, 1], single band
└── vector/<name>.shp       polygons of target_class (+ .shx .dbf .prj)
```

Open the heatmap in QGIS: *Layer → Add Layer → Add Raster Layer*, then set a
single-band pseudocolor ramp. The shapefile loads as a vector layer with the
source raster's projection.

Inputs without georeferencing (a plain `png`/`jpg`) still produce all
outputs, but in **pixel coordinates**; the per-file report says
`input has no georeferencing; geotiff/shapefile outputs use pixel
coordinates`.

## Troubleshooting

**`error: mmsegmentation (and torch) are required to run inference` (exit 3)**
Run through Docker, or `uv sync --extra inference`.

**`error: N problem(s) in config.toml` (exit 2)**
The warnings above it say exactly what is wrong (missing model file, unknown
`target_class`, no classes, …). Use `--dry-run` to see the report without
failing.

**`pip install .` fails during `docker build` with a Requires-Python error**
The Dockerfile still uses its original base image
(`pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel`), which predates the project's
`requires-python = ">=3.9"`, and it clones mmsegmentation/mmdetection from
their `main` branches. Bump the build args at the top of the `Dockerfile`, for
example:

```dockerfile
ARG PYTORCH="2.1.1"
ARG CUDA="11.8"
ARG CUDNN="8"
```

This image has **not** been rebuilt since the repository was generalized (no
Docker daemon was available), so treat build failures there as expected and
report/adjust the pins. Building it is also slow (it compiles mmcv).

**GPU not used**
Check `deploy:` is uncommented in `docker-compose.yml`, that
`nvidia-smi` works on the host, and that `[segmentation] device` is `auto` or
`cuda`. `device = "cuda"` with no CUDA raises an explicit error rather than
silently falling back.

**Out of memory on a large map**
Lower `[segmentation] window_size` (e.g. `1024`). Memory use scales with the
window area, not the map area.

**Split structures along a grid**
Windows do not overlap (see [architecture.md](architecture.md)); reduce
`window_size` to reduce the number of seams, or accept them for now.

**Weird colors / wrong class selected**
`[[classes]] index` values must match the model's class list, and
`target_class` must name one of them. See [models.md](models.md).

## Tests

```bash
uv run pytest                 # 83 tests, no model, no GPU
uv run ruff format src tests
uv run ruff check src tests
```

The suite stubs inference, so it covers configuration, raster IO, palette
rendering, vectorization and the CLI end to end without loading mmsegmentation.
