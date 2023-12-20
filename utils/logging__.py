import wandb
import os
from utils.visualization import to_img
import torch
import cv2
from utils.dataset import invert_transformation
import numpy as np


def log_val_table_app_init(logger, epoch, rgb, rgb_gt, depth, rgb0, depth0, z_std):
    columns = [
        "epoch",
        "rgb",
        "gt rgb",
        "depth",
        "rgb_coarse",
        "depth_coarse",
        "z_std",
    ]
    data = [
        [
            epoch,
            wandb.Image(rgb),
            wandb.Image(rgb_gt),
            wandb.Image(depth),
            wandb.Image(rgb0),
            wandb.Image(depth0),
            wandb.Image(z_std),
        ]
    ]
    logger.log_table(key="val table", columns=columns, data=data)


def log_val_table_app_init_semantic(
    logger, epoch, rgb_car, rgb_noncar, rgb_gt, rgb0_car, rgb0_noncar
):
    columns = [
        "epoch",
        "rgb car",
        "rgb noncar",
        "gt rgb",
        "rgb_coarse car",
        "rgb_coarse noncar",
    ]
    data = [
        [
            epoch,
            wandb.Image(rgb_car),
            wandb.Image(rgb_noncar),
            wandb.Image(rgb_gt),
            wandb.Image(rgb0_car),
            wandb.Image(rgb0_noncar),
        ]
    ]
    logger.log_table(key="val table", columns=columns, data=data)


def log_val_table_online(
    logger,
    epoch,
    rgb,
    rgb_gt,
    rgb_dynamics,
    rgb_static,
    depth,
    depth_dynamics,
    depth_static,
    rgb0,
    rgb_dynamic0s,
    rgb0_static,
    depth0,
    depth_dynamic0s,
    depth_static0,
    z_std,
):
    columns = [
        "epoch",
        "rgb",
        "gt rgb",
        "static rgb",
        "depth",
        "depth static",
        "rgb_coarse",
        "static rgb coarse",
        "depth0",
        "depth static0",
        "z_std",
    ]
    columns += [f"dynamic rgb vehicle{i}" for i in range(rgb_dynamics.shape[0])]
    columns += [f"dynamic depth vehicle{i}" for i in range(rgb_dynamics.shape[0])]
    columns += [f"dynamic rgb coarse vehicle{i}" for i in range(rgb_dynamics.shape[0])]
    columns += [
        f"dynamic depth coarse vehicle{i}" for i in range(rgb_dynamics.shape[0])
    ]

    data = [
        [
            epoch,
            wandb.Image(rgb),
            wandb.Image(rgb_gt),
            wandb.Image(rgb_static),
            wandb.Image(depth),
            wandb.Image(depth_static),
            wandb.Image(rgb0),
            wandb.Image(rgb0_static),
            wandb.Image(depth0),
            wandb.Image(depth_static0),
            wandb.Image(z_std),
        ]
    ]
    data[0] += [wandb.Image(img) for img in rgb_dynamics]
    data[0] += [wandb.Image(img) for img in depth_dynamics]
    data[0] += [wandb.Image(img) for img in rgb_dynamic0s]
    data[0] += [wandb.Image(img) for img in depth_dynamic0s]

    logger.log_table(key="val table", columns=columns, data=data)


def log_test_table_online(
    logger,
    frame,
    rgb,
    rgb_dynamic,
    rgb_static,
    depth,
    depth_dynamic,
    depth_static,
    rgb0,
    rgb0_dynamic,
    rgb0_static,
    depth0,
    depth_dynamic0,
    depth_static0,
    z_std,
):
    columns = [
        "frame",
        "rgb",
        "dynamic rgb",
        "static rgb",
        "depth",
        "depth dynamic",
        "depth static",
        "rgb_coarse",
        "dynamic rgb coarse",
        "static rgb coarse",
        "depth0",
        "depth dynamic0",
        "depth static0",
        "z_std",
    ]
    data = [
        [
            frame,
            wandb.Image(rgb),
            wandb.Image(rgb_dynamic),
            wandb.Image(rgb_static),
            wandb.Image(depth),
            wandb.Image(depth_dynamic),
            wandb.Image(depth_static),
            wandb.Image(rgb0),
            wandb.Image(rgb0_dynamic),
            wandb.Image(rgb0_static),
            wandb.Image(depth0),
            wandb.Image(depth_dynamic0),
            wandb.Image(depth_static0),
            wandb.Image(z_std),
        ]
    ]
    logger.log_table(key="val table", columns=columns, data=data)


def log_2d_iou(iou, predicted_masks, gt_mask, gt_rgb, frame, view):
    num_vehicles = predicted_masks.shape[0]

    columns = ["gt mask"]
    columns += [f"vehicle {i} estimated mask" for i in range(num_vehicles)]
    columns += ["gt rgb"]
    columns += ["2d iou"]
    columns += ["frame"]
    columns += ["view"]

    data = [wandb.Image(to_img(gt_mask))]
    data += [wandb.Image(to_img(mask)) for mask in predicted_masks]
    data += [wandb.Image(gt_rgb)]
    data += [iou]
    data += [frame]
    data += [view]

    table = wandb.Table(columns=columns, data=[data])
    wandb.log({"2D IOU": table})

    """
    to8b(
        batch["target"][frame].reshape(test_H, test_W, 3).cpu().detach().numpy(),
        "target",
    )
    """


# Modified from https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate
    point = np.array([loc[0], loc[1], loc[2], 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)[..., :3]

    # now project 3D->2D using the camera matrix

    point_img = np.ones(3, dtype=np.float32)
    point_img[0] = (K[0][0] * point_camera[0]) / (-point_camera[-1]) + K[0][2]
    point_img[1] = -(K[1][1] * point_camera[1]) / (-point_camera[-1]) + K[1][2]
    point_img[2] = -1

    # point_img = np.dot(K, point_camera)
    # point_img[0] /= -point_img[2]
    # point_img[1] /= -point_img[2]

    return point_img[0:2]


# Some parts are modified from https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
def log_3d_iou(iou_3d, bboxes, gt_bboxes, rgb_gt, K, c2w):
    w2c = invert_transformation(c2w.cpu().numpy())
    num_vehicles = bboxes.shape[0]
    columns = ["rgb"]
    columns += [f"iou vehicle{i}" for i in range(num_vehicles)]

    # Carla bbox edges
    bbox_edges = [
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [0, 4],
        [4, 5],
        [5, 1],
        [5, 7],
        [7, 6],
        [6, 4],
        [6, 2],
        [7, 3],
    ]
    bbox_rgb = np.copy(rgb_gt)

    for i in range(num_vehicles):
        for j, edge in enumerate(bbox_edges):
            # Join the vertices into edges
            p1 = get_image_point(gt_bboxes[i, edge[0]], K, w2c)
            p2 = get_image_point(gt_bboxes[i, edge[1]], K, w2c)
            # Draw the edges into the camera output
            cv2.line(
                bbox_rgb,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                # (0, 0, 255, 255),
                (0, 0, 255),
                1,
            )

            p1 = get_image_point(bboxes[i, edge[0]], K, w2c)
            p2 = get_image_point(bboxes[i, edge[1]], K, w2c)
            # Draw the edges into the camera output
            cv2.line(
                bbox_rgb,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (255, 255, 0),
                1,
            )

    data = [wandb.Image(bbox_rgb)]
    data += [iou_3d[i] for i in range(num_vehicles)]

    table = wandb.Table(columns=columns, data=[data])
    wandb.log({"3D IOU": table})
