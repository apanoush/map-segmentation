# Adding a model

`model/` holds two things: the **mmsegmentation model config** (a `.py` file)
and the **trained checkpoint** (a `.pth` file). Nothing else needs changing.

```
model/
├── mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py   ← [model] config
├── best_mIoU_iter_138828.pth                                          ← [model] checkpoint
└── ...                                                                (directories are fine)
```

Point `config.toml` at them:

```toml
[model]
config     = "model/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py"
checkpoint = "model/best_mIoU_iter_138828.pth"
```

Both paths are relative to the config file. `docker-compose.yml` mounts the
whole `model/` directory read-only, so adding files there needs no rebuild.

## Where the files come from

| you have | what to copy |
|---|---|
| a training run (`mmseg train ...`) | `work_dirs/<experiment>/*.py` (the resolved config) + the chosen `.pth` |
| a model-zoo entry (e.g. Mask2Former ADE20K) | the config from mmsegmentation's `configs/<family>/` **and** its `_base_` files, plus the downloaded checkpoint |
| only a checkpoint | the matching config from [mmsegmentation configs](https://github.com/open-mmlab/mmsegmentation/tree/main/configs) |

### Configs that use `_base_`

mmsegmentation configs are often split:

```python
_base_ = ['./ade20k_640x640.py', './default_runtime.py']
```

`mmengine` resolves those paths **relative to the config file**, so copy the
whole directory (as `model/default/` does) rather than the single file. The
shipped `mask2former_swin-l-...py` is a flattened, self-contained config and
works on its own.

## Class indices and colours

The checkpoint fixes the number and order of classes; the config declares how
they are *named* and *rendered*. They have to agree.

1. Read `num_classes` from the model config (for the shipped model:
   `num_classes = 6`).
2. Establish the index → meaning mapping. It comes from the dataset the model
   was trained on (mmseg configs usually define `classes`/`palette` in the
   dataset definition; for a custom run it is whatever your label rasters used).
3. Declare one `[[classes]]` entry per index in `config.toml`, with the name and
   colour you want:

```toml
[[classes]]
index = 0
name  = "background"
color = "#000000"

[[classes]]
index = 2
name  = "built"
color = "#00ff00"
```

4. Set `[segmentation] target_class` to one of those **names** — that class
   feeds both `results/heatmap/` and `results/vector/`.

The shipped model (trained on `synth_v5`) uses:

| index | name | color |
|---|---|---|
| 0 | `background` | `#000000` |
| 1 | `contours` | `#0000ff` |
| 2 | `built` | `#00ff00` |
| 3 | `non-built` | `#ff0000` |
| 4 | `water` | `#00ffff` |
| 5 | `road_network` | `#ff00ff` |

Indexes that are missing from `[[classes]]` render as class 0 rather than
crashing, but `target_class` must exist — that mismatch is reported by
`--dry-run`.

### Checking your class list quickly

```bash
python -m image2shp --dry-run | grep classes
```

## Requirements on the model itself

* The config must be loadable by the installed mmsegmentation version
  (the Docker image installs mmsegmentation from `main`; if you pin a
  different version, keep config and version together).
* The model's **test pipeline must not resize** inputs and its `test_cfg`
  should be `dict(mode='whole')`; otherwise the per-window logits would not
  line up with the input grid, and `inference.predict_logits_from_info`
  raises an explicit error instead of writing a misaligned raster.
* Input channels: the preprocessor converts BGR→RGB (`bgr_to_rgb=True`), which
  is what the pipeline assumes when it hands over raster windows.
* Three-band inputs are expected. Single-band images are replicated to three
  channels; more than three bands are truncated to the first three.

## Using several models

Nothing stops you from keeping several `.py`/`.pth` pairs in `model/` and
switching with a second config file:

```bash
docker compose run --rm image2shp --config config_water.toml
```

Each config keeps its own `[model]` and `[[classes]]` (see
[configuration.md](configuration.md)).
