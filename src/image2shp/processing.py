import torch
from typing import Tuple, Union
import cv2
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter

import rasterio

def save_as_png(filepath, image):
    cv2.imwrite(str(filepath), image)


def save_as_geotiff(img_path, img, save_path, alpha_layer = True):

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    # 1. Read original GeoTIFF
    with rasterio.open(img_path) as src:
        profile = src.profile.copy()   # metadata
        #transform = src.transform
        #crs = src.crs

    #nb_layer = 4 if alpha_layer else 3 #img.shape[2]  
    nb_layer = img.shape[2] #4 if alpha_layer else img.shape[2]

    #profile.update(
    #    dtype=rasterio.uint8,
    #    count=nb_layer
    #)
    profile.update(
        dtype=img.dtype,
        count= nb_layer
    )

    #img = np.moveaxis(img, -1, 0)
    
    #with rasterio.open(save_path, "w", **profile) as dst:
    #    dst.write(img)

    with rasterio.open(save_path, "w", **profile) as dst:
        for i in range(nb_layer):
            dst.write(img[:, :, i], i + 1)

def single_class_heatmap(pred_logits, class_id=2, sigma=25, color=(0, 0, 255)):
    # 1. Argmax → class labels
    labels = np.argmax(pred_logits, axis=0)

    # 2. Binary mask for selected class
    mask = (labels == class_id).astype(np.float32)

    # 3. Smooth it
    smooth = gaussian_filter(mask, sigma=sigma)

    return _normalize_and_convert_binary_mask_to_colored_alpha_image(smooth, color)

"""
    # 4. Normalize to [0, 1]
    if smooth.max() > smooth.min():
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())

    # 5. Create colored image
    h, w = smooth.shape
    colored = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

    # RGB color
    colored[..., 0] = color[0]
    colored[..., 1] = color[1]
    colored[..., 2] = color[2]

    # 6. Alpha channel from heatmap
    colored[..., 3] = (smooth * 255).astype(np.uint8)

    return colored
"""

def single_class_predictions(pred_logits, class_id=2, color=(0, 0, 255)):
    # 1. Argmax → class labels
    mask = pred_logits[class_id]

    # 2. Binary mask for selected class
    #mask = (pred_logits == class_id).astype(np.float32)

    # 3. Smooth it
    smooth = mask#gaussian_filter(mask, sigma=sigma)

    return _normalize_and_convert_binary_mask_to_colored_alpha_image(smooth, color)


def _normalize_and_convert_binary_mask_to_colored_alpha_image(binary_mask: np.ndarray, color: Tuple[int, int, int] = (0,0,255)) -> np.ndarray:

    # 4. Normalize to [0, 1]
    if binary_mask.max() > binary_mask.min():
        binary_mask = (binary_mask - binary_mask.min()) / (binary_mask.max() - binary_mask.min())

    # 5. Create colored image
    h, w = binary_mask.shape
    colored = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

    # RGB color
    colored[..., 0] = color[0]
    colored[..., 1] = color[1]
    colored[..., 2] = color[2]

    # 6. Alpha channel from heatmap
    colored[..., 3] = (binary_mask * 255).astype(np.uint8)

    return colored


def softmax(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)  # stability trick
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def apply_gaussian(mask, sigma=10, apply_softmax=True) -> np.ndarray:
    #return gaussian_filter(pred_mask, sigma=sigma)

    if apply_softmax:
        mask = softmax(mask, axis=0)
    smoothed = np.stack([gaussian_filter(c, sigma=sigma) for c in mask])
    return smoothed


def preds2colors(pred_mask):

    image = np.argmax(pred_mask, axis=0)
    #plt.imshow(image)
    #plt.savefig(str(output_path).replace(".png", "_2.png"))


    # Step 2: define 6 class colors (BGR)
    colors = np.array([
        [0, 0, 0],        # class 0 - black -> backgound
        [255, 0, 0],      # class 1 - blue -> contours
        [0, 255, 0],      # class 2 - green -> buildings
        [0, 0, 255],      # class 3 - red -> non-built
        [255, 255, 0],    # class 4 - cyan -> water
        [255, 0, 255],    # class 5 - magenta -> road-network
    ], dtype=np.uint8)

    # Step 3: map labels to colors
    colored = colors[image]
    return colored

def preds2built(pred_logits, heatmap_sigma: Union[int, None] = 25, color:Tuple[int, int, int]=(0,0,255)):

    labels = np.argmax(pred_logits, axis=0)
    # building, countours, road-network
    built_ids = [2, 1, 5]
    mask = np.isin(labels, built_ids).astype(np.float32)

    if heatmap_sigma is not None:
        mask = gaussian_filter(mask, sigma=heatmap_sigma)

    return _normalize_and_convert_binary_mask_to_colored_alpha_image(mask, color)

def load_model(config_path, checkpoint_path, device="cpu"):
    model = init_model(str(config_path), str(checkpoint_path), device=device)

    if not torch.cuda.is_available():
        model = revert_sync_batchnorm(model)

    return model

def get_pred_mask(model, img_path):

    img = tiff.imread(img_path)

    # If single-channel → make it 3-channel
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # Convert uint16 → uint8 with normalization
    img = img.astype(np.float32)

    # Avoid division by zero
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    img = (img * 255).astype(np.uint8)

    result = inference_model(model, img)

    #del img

    pred_mask = result.seg_logits.data.detach().cpu().numpy()

    return pred_mask

def get_pred_mask_per_window(model, img_path, window_size=1024*2):
    """similar to get pred mask but reduce memory usage"""
    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        pred_mask = np.zeros((model.num_classes, H, W), dtype=np.float32)

        for i in range(0, H, window_size):
            for j in range(0, W, window_size):
                window = rasterio.windows.Window(j, i, window_size, window_size)
                img = src.read(window=window)  # shape: (C, h, w)

                img = np.moveaxis(img, 0, -1)  # → (h, w, C)

                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)

                img = img.astype(np.float32)
                if img.max() > img.min():
                    img = (img - img.min()) / (img.max() - img.min())
                img = (img * 255).astype(np.uint8)

                result = inference_model(model, img)
                logits = result.seg_logits.data.cpu().numpy()

                pred_mask[:, i:i+logits.shape[1], j:j+logits.shape[2]] = logits

                del img, result, logits

    return pred_mask
