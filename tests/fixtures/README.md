# Test fixtures

| file | what it is |
|---|---|
| `test_1.png`, `test_2.png` | two 2362×3826 segmentation outputs produced by the **legacy** pipeline (6-class palette, buildings in pink `#ff00ff`). Kept as realistic sample data for the smoke tests; `test_processing.py` asserts they stay readable. |

Everything else the test suite needs (GeoTIFFs, masks, configs) is generated on
the fly in a temporary directory by `tests/conftest.py`, so no large binaries
are needed.

Large artifacts (`.npy`, `.pth`, `.tiff`) are git-ignored here — the ~450 MB of
`pred_mask*` files that used to live in `tests/` were removed when the test
suite was created.
