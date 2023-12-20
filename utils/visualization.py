# Taken from https://github.com/kwea123/nerf_pl/blob/master/utils/visualization.py
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

from models.rendering__ import to8b


def visualize_depth(depth, H=400, W=400, cmap=cv2.COLORMAP_JET, app_init=False):
    """
    depth: (H*W) or (num_vehicles, H*W)
    """
    if app_init or depth.ndim == 1:
        x = depth.cpu().numpy()
        x = np.nan_to_num(x)  # change nan to 0
        mi = np.min(x)  # get minimum depth
        ma = np.max(x)
        x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
        x = (255 * x).astype(np.uint8)
        x_ = cv2.applyColorMap(x, cmap)
    elif depth.ndim == 2:
        x = depth.cpu().numpy()
        x = np.nan_to_num(x)  # change nan to 0
        mi = np.min(x, axis=1)  # get minimum depth, (num_vehicles, )
        ma = np.max(x, axis=1)
        mi = np.tile(mi[:, None], (1, x.shape[1]))
        ma = np.tile(ma[:, None], (1, x.shape[1]))
        x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1, (num_vehicles, H*W)
        x = (255 * x).astype(np.uint8)
        x_ = []
        for i in range(x.shape[0]):
            x_.append(cv2.applyColorMap(x[i], cmap).reshape((H, W, 3)))
    else:
        raise NotImplementedError()
    return x_


# Used in carla star app init semantic
def combine_static_dynamic(
    H, W, channel, arr_static, arr_dynamic, mask_static, mask_dynamic, debug_str
):
    result = np.zeros((H, W, channel), dtype=np.uint8)

    result[mask_static] = to8b(arr_static.cpu().numpy(), debug_str)
    result[mask_dynamic] = to8b(arr_dynamic.cpu().numpy(), debug_str)

    return result


def to_img(raw_img, H=400, W=400):
    if isinstance(raw_img, torch.Tensor):
        raw_img = raw_img.cpu().detach().numpy()

    if len(raw_img.shape) == 3:
        pass
    elif len(raw_img.shape) == 2:
        if (raw_img.shape[-1]) == 3:
            raw_img = raw_img.reshape(H, W, 3)
        else:
            raise NotImplementedError
    elif len(raw_img.shape) == 1:
        raw_img = raw_img.reshape(H, W, 1)
    else:
        raise NotImplementedError

    img = to8b(raw_img, "to_img")

    return img
