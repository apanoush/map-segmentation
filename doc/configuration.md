# Configuration reference

There is exactly one configuration file: `config.toml` at the repository root.
Point the tool at another one with `--config path/to/other.toml`.

## Path resolution

Every path in the file is resolved **relative to the directory containing the
config file** (the repository root for the shipped one):

```toml
[paths]
data_dir = "data"      # → <repo>/data
```

* `~/...` is expanded to the user's home directory first;
* an absolute path is used as-is;
* relative paths are joined onto the config file's directory and normalised.

This is what makes one file work both on the host and in the container: the
container's working directory is `/app` and `docker-compose.yml` mounts
`./data`, `./model` and `./results` at `/app/data`, `/app/model`,
`/app/results`, so the relative paths resolve to the same places.

## Sections

### `[paths]`

| key | type | default | meaning |
|---|---|---|---|
| `data_dir` | string | `"data"` | directory searched for input images |
| `results_dir` | string | `"results"` | directory receiving all outputs |

### `[model]` *(required)*

| key | type | default | meaning |
|---|---|---|---|
| `config` | string | — | mmsegmentation model definition (`.py`) |
| `checkpoint` | string | — | trained weights (`.pth`) |

Both are interpreted relative to the config file, so `model/...` works
everywhere. See [models.md](models.md).

### `[inputs]`

| key | type | default | meaning |
|---|---|---|---|
| `patterns` | list of strings | `["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]` | glob patterns matched against `data_dir` |
| `recursive` | bool | `false` | also search subdirectories (`**/<pattern>`) |

```toml
[inputs]
patterns = ["*.tif"]
recursive = true
```

Case matters for globs on Linux: add `"*.TIF"` if your files are uppercase.
Files are de-duplicated across patterns and sorted by name. If two inputs share
the same file stem (e.g. `map.tif` and `map.png`), their outputs would collide
and `pipeline.run` fails the second one instead of overwriting.

### `[segmentation]`

| key | type | default | constraints | meaning |
|---|---|---|---|---|
| `device` | string | `"auto"` | `auto`, `cpu`, `cuda` | inference device; `auto` = CUDA when available |
| `window_size` | int | `2048` | `>= 0` | side of the square window for tiled inference; `0` = whole image in one pass |
| `target_class` | string | `"built"` | must match a `[[classes]]` `name` | class used for the heatmap **and** the shapefile |
| `heatmap_sigma` | float | `25` | `> 0` | Gaussian blur radius (pixels) of the heatmap density |
| `buffer_distance` | int | `20` | `>= 0` | dilation (pixels) applied before polygon extraction — bridges small gaps between pixels of the same class |

`window_size = 2048` bounds memory on large rasters; windows do **not**
overlap, so structures crossing a boundary can be split (see
[architecture.md](architecture.md), decision 5).

### `[outputs]`

| key | type | default | file written |
|---|---|---|---|
| `segmented_png` | bool | `true` | `results/segmented/<name>.png` — RGB class map |
| `segmented_geotiff` | bool | `true` | `results/segmented/<name>.tif` — same, georeferenced |
| `heatmap_geotiff` | bool | `true` | `results/heatmap/<name>.tif` — single-band float32 density in `[0, 1]` |
| `shapefile` | bool | `true` | `results/vector/<name>.shp` (+ `.shx`, `.dbf`, `.prj`) |

Setting every key to `false` is reported as a configuration problem (the run
would do nothing).

### `[[classes]]`

An array of tables — one entry per class emitted by the model. **`index` must
match the class indices of the checkpoint you configured.**

| key | type | constraints | meaning |
|---|---|---|---|
| `index` | int | `>= 0`, unique | class index in the model's output |
| `name` | string | non-empty, unique | human-readable name; `target_class` refers to it |
| `color` | string | `#RGB` or `#RRGGBB` | colour used to render that class |

```toml
[[classes]]
index = 0
name = "background"
color = "#000000"

[[classes]]
index = 2
name = "built"
color = "#00ff00"
```

Rendering rules:

* the segmented map is `color[index]`, one entry per class;
* an index the config does not declare renders as class 0 (black, unless class
  0 declares another colour) instead of raising;
* the heatmap is a plain float32 band — style it in QGIS, colours are not
  applied to it;
* the shapefile selects pixels equal to `target_class`'s index.

## Validation

`load_config()` raises `ConfigError` for anything that makes the file
unusable — missing file, invalid TOML, missing `[model]`, wrong types, out of
range values, malformed colours.

`Config.problems()` returns a list of *environment* issues that should not
prevent printing a dry-run: data directory missing, model files missing, no
`[[classes]]`, duplicate indices/names, unknown `target_class`, all outputs
disabled. The CLI prints them as `warning:` lines, aborts a real run with exit
code `2`, and lets `--dry-run` continue with exit code `0`.

## Example

A complete file for a two-class model, only producing a heatmap:

```toml
[paths]
data_dir = "data/scans"
results_dir = "results/2026-09"

[model]
config = "model/my_model.py"
checkpoint = "model/my_model.pth"

[inputs]
patterns = ["*.tiff"]
recursive = true

[segmentation]
device = "cpu"
window_size = 1024
target_class = "water"
heatmap_sigma = 10
buffer_distance = 4

[outputs]
segmented_png = false
segmented_geotiff = false
heatmap_geotiff = true
shapefile = false

[[classes]]
index = 0
name = "background"
color = "#000000"

[[classes]]
index = 1
name = "water"
color = "#0000ff"
```
