# Taken from https://github.com/kwea123/nerf_pl/blob/master/utils/visualization.py
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

from models.rendering import to8b


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_


# Used in carla star app init semantic
def combine_static_dynamic(
    H, W, channel, arr_static, arr_dynamic, mask_static, mask_dynamic, debug_str
):
    result = np.zeros((H, W, channel), dtype=np.uint8)

    result[mask_static] = to8b(arr_static.cpu().numpy(), debug_str)
    result[mask_dynamic] = to8b(arr_dynamic.cpu().numpy(), debug_str)

    return result
