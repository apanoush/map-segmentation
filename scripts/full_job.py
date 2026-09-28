
import os
import sys
sys.path.insert(0, ".")
from src.image2shp.processing import single_class_heatmap, single_class_predictions, get_pred_mask, save_as_geotiff, save_as_png, apply_gaussian, preds2colors, load_model, preds2built, get_pred_mask_per_window
from src.image2shp.config import load_config
from typing import Tuple

SIGMA = 25
SKIP_IF_RESULT_EXISTS = True


def handle_paths(output_dir: str, output_name: str, file: str) -> Tuple[str, str]:
    output_path = os.path.join(
        output_dir, output_name
    )
    output_path_png = os.path.join(
        output_path, "png", os.path.splitext(file)[0] + ".png"
    )
    output_path_tiff = os.path.join(
        output_path, "tiff", os.path.splitext(file)[0] + ".tiff"
    )
    os.makedirs(os.path.dirname(output_path_png), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_tiff), exist_ok=True)
    return output_path_png, output_path_tiff

config = load_config()

device = "cpu"
checkpoint_path = "model/best_mIoU_iter_138828.pth"
config_path = "model/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py"

data_dir = "data" #"data/1907-1925-1937 georeferencement bati"
files = os.listdir(data_dir)
output_dir = "results"
print("AHHH", files)

model = load_model(config_path, checkpoint_path, device="cpu")

for file in files:

    print(file)

    try:
        img_path = os.path.join(data_dir, file)

        assert os.path.isfile(img_path)

        #pred_mask = get_pred_mask(model, img_path)
        pred_mask = get_pred_mask_per_window(model, img_path)


        png_output_path, tiff_output_path = handle_paths(output_dir, "segmented_image", file)

        #if (not SKIP_IF_RESULT_EXISTS) and (not os.path.isfile(tiff_output_path)):

        result = preds2colors(pred_mask)
        save_as_geotiff(img_path, result ,tiff_output_path, alpha_layer=False)
        save_as_png(png_output_path, result)

        png_output_path, tiff_output_path = handle_paths(output_dir, "buildings_heatmap", file)

        result = single_class_heatmap(pred_mask, sigma=SIGMA)

        save_as_geotiff(img_path, result ,tiff_output_path, alpha_layer=True)
        save_as_png(png_output_path, result)

        png_output_path, tiff_output_path = handle_paths(output_dir, "buildings_predictions", file)

        result = single_class_predictions(pred_mask)
        save_as_geotiff(img_path, result ,tiff_output_path, alpha_layer=True)
        save_as_png(png_output_path, result)

        png_output_path, tiff_output_path = handle_paths(output_dir, "built_predictions", file)

        result = preds2built(pred_mask, None)
        save_as_geotiff(img_path, result ,tiff_output_path, alpha_layer=True)
        save_as_png(png_output_path, result)

        png_output_path, tiff_output_path = handle_paths(output_dir, "built_heatmap", file)

        result = preds2built(pred_mask, SIGMA)
        save_as_geotiff(img_path, result ,tiff_output_path, alpha_layer=True)
        save_as_png(png_output_path, result)

    except:
        print("Couldn't process file, skipping")
        continue






