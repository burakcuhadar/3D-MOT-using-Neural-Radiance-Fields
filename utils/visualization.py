# Taken from https://github.com/kwea123/nerf_pl/blob/master/utils/visualization.py
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import math

from PIL import Image
from models.rendering__ import to8b


def visualize_depth(
    depth,
    H=400,
    W=400,
    cmap=cv2.COLORMAP_JET,
    multi_vehicle=False,
    return_normalized=False,
):
    """
    depth: (H*W) or (num_vehicles, H*W)
    """
    assert (
        depth.shape == (H * W,) or depth.shape[1] == H * W
    ), f"wrong depth shape: {depth.shape}"

    if not multi_vehicle or depth.ndim == 1:
        x = depth.cpu().numpy()
        x = np.nan_to_num(x)  # change nan to 0
        mi = np.min(x)  # get minimum depth
        ma = np.max(x)
        normed_depth = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
        x = (255 * normed_depth).astype(np.uint8)
        x_ = cv2.applyColorMap(x, cmap)
    elif depth.ndim == 2:
        x = depth.cpu().numpy()
        x = np.nan_to_num(x)  # change nan to 0
        mi = np.min(x, axis=1)  # get minimum depth, (num_vehicles, )
        print("depth min", mi)
        ma = np.max(x, axis=1)
        print("depth max", ma)
        mi = np.tile(mi[:, None], (1, x.shape[1]))
        ma = np.tile(ma[:, None], (1, x.shape[1]))
        normed_depth = (x - mi) / (
            ma - mi + 1e-8
        )  # normalize to 0~1, (num_vehicles, H*W)
        x = (255 * normed_depth).astype(np.uint8)
        x_ = []
        for i in range(x.shape[0]):
            x_.append(cv2.applyColorMap(x[i], cmap).reshape((H, W, 3)))
        x_ = np.stack(x_, axis=0)
    else:
        raise NotImplementedError()

    if return_normalized:
        return x_, normed_depth
    return x_


def visualize_depth_with_values(
    depth_rgb,  # (num_vehicles, H*W, 3)
    depth_val,  # (num_vehicles, H, W)
    H=400,
    W=400,
):
    depth_rgb = depth_rgb.reshape((-1, H, W, 3))
    depth_val = depth_val.reshape((-1, H, W))
    assert (
        depth_rgb.ndim == 4
    ), f"depth should be (num_vehicles, H, W, 3), instead got {depth_rgb.shape}"
    assert (
        depth_val.ndim == 3
    ), f"depth should be (num_vehicles, H, W), instead got {depth_val.shape}"
    H, W = depth_rgb.shape[1:3]

    depth_rgb_val = depth_rgb.copy()

    for i in range(depth_rgb.shape[0]):
        for x in range(0, W, 20):
            for y in range(0, H, 20):
                depth_rgb_val[i] = cv2.putText(
                    depth_rgb_val[i],
                    # f".{int(math.modf(math.modf(depth_val[i, y, x])[0]*10)[1])}",
                    # f"{depth_val[i, y, x]:.1f}",
                    f"{int(math.modf(math.modf(depth_val[i, y, x])[0]*100)[1])}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (255, 255, 255),
                    1,
                )

    return depth_rgb_val


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
