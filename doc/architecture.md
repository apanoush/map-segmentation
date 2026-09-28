# Architecture

## Goals

- one entry point (`python -m image2shp`) that works the same on the host and in the container;
- drop any mmsegmentation model into `model/` and any image into `data/` — no code changes;
- everything except inference importable and testable without torch installed;
- one configuration file instead of a host/container pair.

## Data flow

```
config.toml ──► config.load_config() ──► Config (resolved paths, palette, options)
                                             │
data/*.{tif,png,…} ──► pipeline.discover_inputs()
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │  per file                                    │
                     │                                              │
                     │  processing.open_raster()  → RasterInfo      │
                     │        (size, dtype, transform, CRS)         │
                     │  inference.predict_logits_from_info()        │
                     │        windowed mmseg inference → (C,H,W)    │
                     │  inference.labels_from_logits() → (H,W)      │
                     │        argmax                                 │
                     │  colors.render_labels() → RGB using palette  │
                     │  processing.class_density() → float32 [0,1]  │
                     │  vectorize.mask_to_shapefile() → polygons    │
                     └───────────────────────┬───────────────────────┘
                                             ▼
        results/segmented/<name>.png|.tif   results/heatmap/<name>.tif
        results/vector/<name>.shp(.shx,.dbf,.prj)
```

## Modules

| module | responsibility | imports mmseg? |
|---|---|---|
| `config.py` | parse/validate `config.toml`, resolve paths relative to the file, class lookup helpers, `problems()` report | no |
| `processing.py` | raster metadata/windowed reading, `uint8` normalisation, PNG/GeoTIFF writing, Gaussian class density | no |
| `colors.py` | build the palette LUT from `[[classes]]`, map a label map to RGB | no |
| `vectorize.py` | label mask → buffered contours → polygons → GeoDataFrame/shapefile | no |
| `inference.py` | lazy torch/mmseg import, model loading, windowed prediction, argmax | **yes, lazily** |
| `pipeline.py` | input discovery, output paths, per-file orchestration, per-file error isolation, batch reports | no |
| `cli.py` | argparse, logging setup, dry-run report, exit codes | no |

`image2shp/__init__.py` re-exports only `load_config` and the config dataclasses, so `import image2shp` is cheap.

### Why the split

`processing.py`, `colors.py` and `vectorize.py` used to sit next to `torch`/`mmseg` imports in the same modules (`segment_image.py`, `processing.py`, `heatmap.py`), which made the whole package unimportable — and therefore untestable — without a GPU stack. Keeping every mmseg reference inside `inference.py`, and importing it *inside functions*, is what allows the 80+ tests to run on a plain laptop.

## Design decisions

### 1. One config file instead of `config.toml` + `config.docker.toml`

*Alternatives considered*: keep the pair and generate one from the other (the retired `old/run_docker.sh` did this with `sed`); use environment variables; use two different path styles.

The two files differed only in `images_dir` and the two GeoTIFF paths, because the host used `~/epfl_local/shs/...` while the container used `/app/data/...`. The difference was never really "host vs container" — it was "paths that live outside the project vs paths inside it". Normalising everything to **paths relative to the config file**, and mirroring the repository layout inside the container (`WORKDIR /app` + `./data → /app/data` etc.), makes the two layouts identical, so one file suffices. It also makes the shipped `config.toml` committable (no personal home directory in it).

The 1973/1983 fields are gone: the pipeline now globs `data_dir` with `[inputs] patterns`, so adding a map means copying a file.

### 2. Vectorize from the label mask, not from a color in a PNG

*Alternatives considered*: keep the "find pixels equal to `(255,0,255)`" approach from `segmented_image_to_shp`.

The old flow was: model → render PNG where *only* building pixels were pink → detect pink pixels → polygonize. Once rendering became a six-class palette (`preds2colors`), pink stopped meaning "building" and started meaning "whatever class 5 is": the pipeline would have silently vectorized the wrong class. Operating on the label array removes the ambiguity entirely and costs nothing.

`segmented_image_to_shp(png, geotiff, ...)` is still exported as a thin wrapper for callers that only have a colored PNG.

Related fix: contours are extracted from a zero-padded mask. Without padding, `skimage.measure.find_contours` drops a mask that covers the whole image (0 contours) and truncates any mask touching the image border — a building at the edge of a tile would have come out clipped.

### 3. Palette and target class come from the configuration

*Alternatives considered*: hardcode the six training colours; ship a `palette.json`.

The old code had two divergent palettes (one in `processing.py`, one in `old/tests/inference-upscale.py`) and hardcoded class indices (`class_id=2`, `built_ids = [2, 1, 5]`) in at least three places. `[[classes]]` entries carry `index`, `name` and `color`; `[segmentation] target_class` refers to a class *by name*, and the same entry drives the heatmap, the shapefile and the "which pixels are they" question. Swapping in a model with a different class list is then a config edit.

### 4. Three outputs instead of five near-duplicates

The archived `full_job.py` wrote `segmented_image`, `buildings_heatmap`, `buildings_predictions`, `built_predictions` and `built_heatmap` per input — five trees (ten directories) that were the same red-translucent overlay with different class selections and with/without blur. They are now:

- `segmented` — the full class map (what `segmented_image` was),
- `heatmap` — Gaussian density of `target_class`, as a float32 GeoTIFF you can style in QGIS (what the `*_heatmap` variants approximated, and what `todo.md` asked for),
- `vector` — polygons of `target_class`.

The remaining knobs (`heatmap_sigma`, `buffer_distance`, `[[classes]]`) cover what was previously hardcoded. A "built = classes 2+1+5" union is not expressible any more; if that matters, extending `target_class` to accept a list of names is a small, contained change.

### 5. Windowed inference by default (`window_size = 2048`)

The input GeoTIFFs are tens of megabytes; whole-image inference was the memory-fragile path (`get_pred_mask`). Reading and predicting window by window bounds memory use at the cost of seam artifacts — windows do not overlap, so a structure crossing a window boundary can be split. Overlapping windows with weighted blending (the pattern in `old/tests/inference-upscale.py`) is the natural next improvement; it needs a correctness test, so it was left out of this pass. `window_size = 0` restores single-pass inference.

The model's own `test_cfg = dict(mode='whole')` performs no resizing at test time, which is what makes per-window logits directly assemblable; `inference.predict_logits_from_info` asserts the returned shape rather than assuming it.

### 6. Installed package, absolute imports

The package used to be imported as `src.image2shp...` with `sys.path.insert(0, ".")` in three different places, and shipped no `__init__.py`. It is now a normal hatchling package installed editable by `uv sync`, imported as `image2shp.*`; the console script `image2shp` and `python -m image2shp` both point at `cli.main`.

### 7. Per-file error isolation

`pipeline.run` records a failure and continues, because a batch of 50 maps should not die on one unreadable file. The CLI prints a summary and exits `1` if anything failed. Exit codes are `0/1/2/3` (ok / file failure / config problem / missing backend) so scripts can branch on them. Config problems are collected by `Config.problems()` and reported *before* any model is loaded.

## Archived code

`old/` is local-only (`/old/*` is ignored) **except** `old/original_workflow/`,
which is kept under version control so the pre-refactor pipeline stays
recoverable:

- `old/original_workflow/` — `full_job.py`, `process_geotiffs.py`, a
  `src_snapshot/` of the modules as they were, and a README;
- `old/TL/` — one-off scripts for the SHS MA4 group (its own config loader,
  data copier, compose file);
- `old/tests/` — the ad-hoc mmseg experiments that used to live in `tests/`;
- the retired Dockerfiles, install scripts and `run_docker.sh` that used to
  generate the Docker config file.

Everything that *was* tracked is also recoverable from git history
(`git show <commit>:src/image2shp/...`).

## Known limitations / future work

- Windows do not overlap → possible seams (see decision 5).
- Plain photos (`png`/`jpg` without georeferencing) still produce all four outputs, but in **pixel coordinates**; the run notes say so per file.
- `results/` uses the input file stem, so `a.tif` and `a.png` in `data/` collide; `pipeline.run` detects this and fails the second file with an explicit message rather than overwriting.
- The `Dockerfile` is inherited unchanged apart from the `COPY` fixes needed to actually contain the code; it has not been rebuilt (see `doc/usage.md`).
- `todo.md` ideas not yet covered: KDE instead of a Gaussian blur, choosing per-class colours for the *heatmap* fill (currently one `target_class` colour comes from `[[classes]]`).
