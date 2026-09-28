import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from PIL import Image

# --- CONFIG ---
predmasks_folder = "results/pred_masks"
tiff_folder = "data/"
output_folder = "results/heatmap"
sigma = 10  # smoothing strength

os.makedirs(output_folder, exist_ok=True)

def load_png_as_array(path):
    img = Image.open(path).convert("L")  # convert to grayscale
    return np.array(img).astype(np.float32)

def normalize(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return arr

def create_heatmap():

    for filename in os.listdir(predmasks_folder):
        if not filename.lower().endswith(".png"):
            continue

        name = os.path.splitext(filename)[0]

        png_path = os.path.join(predmasks_folder, filename)
        tif_path = os.path.join(tiff_folder, name + ".tif")

        if not os.path.exists(tif_path):
            print(f"⚠️ Missing TIFF for {filename}, skipping")
            continue

        print(f"Processing {filename}...")

        # --- Load PNG ---
        data = load_png_as_array(png_path)

        # --- Apply Gaussian smoothing ---
        heatmaps = []
        for class_id in classes:
            heatmaps.append(gaussian_filter(prob_map[class_id], sigma=sigma))
        heatmaps = np.hstack(heatmaps)

        ## --- Normalize (optional but recommended for visualization) ---
        #heatmap = normalize(heatmap)

        # --- Load georeference from TIFF ---
        with rasterio.open(tif_path) as src:
            meta = src.meta.copy()

        # --- Ensure dimensions match ---
        if heatmap.shape != (meta["height"], meta["width"]):
            raise ValueError(f"Shape mismatch for {filename}")

        # --- Update metadata ---
        meta.update({
            "dtype": "float32",
            "count": 1
        })

        # --- Save output ---
        out_path = os.path.join(output_folder, name + "_heatmap.tif")

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(heatmap.astype(np.float32), 1)

    print("✅ Done!")
