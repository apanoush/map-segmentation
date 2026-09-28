"""from <https://github.com/open-mmlab/mmsegmentation/blob/main/demo/inference_demo.ipynb>"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

import sys, os
sys.path.insert(0, ".")
from src.image2shp.colors import preds2colors

try:
    from mmengine.model.utils import revert_sync_batchnorm
    from mmseg.apis import init_model, inference_model, show_result_pyplot

    MMSEG_AVAILABLE = True
except ImportError:
    MMSEG_AVAILABLE = False
    print(
        "Warning: mmsegmentation not installed. Segmentation functions will not work."
    )
    print("Install with: pip install mmsegmentation")

# Try to import rasterio for GeoTIFF reading
try:
    import rasterio
    from rasterio.windows import Window

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not installed. GeoTIFF reading may fail.")


def read_geotiff_as_bgr(tiff_path: str, percentile_clip: tuple = (2, 98)) -> np.ndarray:
    """
    Read a GeoTIFF file and convert to BGR uint8 for mmsegmentation.

    Args:
        tiff_path: Path to GeoTIFF file
        percentile_clip: Tuple of (lower, upper) percentiles for contrast stretching
                         Default (2, 98) clips extreme values

    Returns:
        BGR image as uint8 numpy array with shape (H, W, 3)
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required to read GeoTIFF files")

    with rasterio.open(tiff_path) as src:
        # Check if we have at least 3 bands (RGB)
        if src.count < 3:
            raise ValueError(
                f"TIFF has only {src.count} band(s), need at least 3 for RGB"
            )

        # Read first 3 bands (assumed to be RGB)
        rgb_data = src.read([1, 2, 3], window=Window(0, 0, src.width, src.height))

        # Convert from (bands, height, width) to (height, width, bands)
        rgb_data = np.transpose(rgb_data, (1, 2, 0))

        # Handle 16-bit data by scaling to 8-bit
        if rgb_data.dtype == np.uint16:
            # Use percentile clipping to handle outliers
            lower = np.percentile(rgb_data, percentile_clip[0])
            upper = np.percentile(rgb_data, percentile_clip[1])

            # Clip and scale to 0-255
            rgb_data = np.clip(rgb_data, lower, upper)
            rgb_data = ((rgb_data - lower) / (upper - lower) * 255).astype(np.uint8)
        elif rgb_data.dtype != np.uint8:
            # For other data types (float, etc.), normalize to 8-bit
            rgb_min = rgb_data.min()
            rgb_max = rgb_data.max()
            if rgb_max > rgb_min:
                rgb_data = ((rgb_data - rgb_min) / (rgb_max - rgb_min) * 255).astype(
                    np.uint8
                )
            else:
                rgb_data = rgb_data.astype(np.uint8)

        # Convert RGB to BGR (OpenCV/mmlab default)
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)

        return bgr_data


def segment_image(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_file = (
        "model/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py"
    )
    checkpoint_file = "model/best_mIoU_iter_138828.pth"

    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device=device)

    if not torch.cuda.is_available():
        model = revert_sync_batchnorm(model)
    result = inference_model(model, img_path)

    # show the results
    vis_result = show_result_pyplot(model, img_path, result, show=False)
    plt.imshow(vis_result)

    breakpoint()


def segment_image_to_png(
    img_path,
    output_png_path,
    config_path,
    checkpoint_path,
    building_class_idx=1,
    building_color=(255, 0, 255),
    device=None,
):
    """
    Segment an image using the trained model and save building pixels as colored PNG.

    Args:
        img_path: Path to input image (geotiff or any image format supported by mmseg)
        output_png_path: Path to output PNG file where building pixels will be colored
        config_path: Path to model config file (required)
        checkpoint_path: Path to model checkpoint (required)
        building_class_idx: Index of building class in segmentation output (default: 1)
        building_color: RGB color for building pixels (default: pink)
        device: 'cuda' or 'cpu', auto-detect if None
    """
    if not MMSEG_AVAILABLE:
        raise ImportError("mmsegmentation is not installed. Please install it.")

    if config_path is None:
        raise ValueError("config_path must be provided")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the model
    print(config_path, checkpoint_path)
    model = init_model(str(config_path), str(checkpoint_path), device=device)

    if not torch.cuda.is_available():
        model = revert_sync_batchnorm(model)

    # Check if input is a GeoTIFF and use rasterio for reading
    img_path_obj = Path(img_path)
    img_array = None

    if img_path_obj.suffix.lower() in [".tif", ".tiff"]:
        if RASTERIO_AVAILABLE:
            try:
                print(f"Reading GeoTIFF with rasterio: {img_path}")
                img_array = read_geotiff_as_bgr(img_path)
                print(
                    f"Successfully read TIFF as BGR array with shape {img_array.shape}"
                )
            except Exception as e:
                print(f"Warning: Failed to read TIFF with rasterio: {e}")
                print("Falling back to OpenCV reading...")
        else:
            print(f"Warning: rasterio not available, using OpenCV for TIFF reading")

    # Perform inference
    if img_array is not None:
        # Pass numpy array directly (will use LoadImageFromNDArray)
        result = inference_model(model, img_array)
    else:
        # Use file path (will use LoadImageFromFile)
        result = inference_model(model, img_path)

    """
    # Get predicted segmentation mask
    # result.pred_sem_seg.data is a tensor of shape (1, H, W) with class indices
    pred_mask = result.pred_sem_seg.data.cpu().numpy()  # shape (H, W)

    #torch.save(pred_mask, "tests/pred_mask.pth")
    #np.save("tests/pred_mask_2.npy", result.seg_logits.data.detach().cpu().numpy())

    pred_mask = pred_mask[0]

    # Create RGB image with building pixels colored
    height, width = pred_mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Set building pixels to specified color
    building_mask = pred_mask == 2#building_class_idx
    rgb_image[building_mask] = building_color

    # Save PNG
    output_path = Path(output_png_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_image)
    """

    # the argmax way
    pred_mask2 = result.seg_logits.data.detach().cpu().numpy()

    image_name = os.path.basename(img_path)
    
    np.save(f"results/pred_masks/{os.path.splitext(image_name)[0] + ".npy"})

    colored = preds2colors(pred_mask2)

    cv2.imwrite(str(output_path).replace(".png", "_3.png"), colored)

    print(f"Segmented PNG saved to {output_path}")
    return output_path
