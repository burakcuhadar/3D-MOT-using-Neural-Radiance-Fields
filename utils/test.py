from .visualization import visualize_depth
from models.rendering__ import to8b
import torch
from utils.metrics import compute_2d_iou, compute_3d_iou
from utils.dataset import invert_transformation
import numpy as np


def test_step_for_one_frame(
    dataset,
    frame,
    batch,
    batch_idx,
    test_result,
    num_vehicles,
    estimated_relative_pose=None,
    gt_pose=None,
    bbox_local_vertices=None,
    gt_pose0=None,
):
    test_H = dataset.H
    test_W = dataset.W

    result = {}

    result["rgb_gt"] = to8b(
        batch["target"][frame].reshape(test_H, test_W, 3).cpu().detach().numpy(),
        "target",
    )

    result["rgb0"] = to8b(
        torch.reshape(test_result["rgb0"], (test_H, test_W, 3)).cpu().detach().numpy(),
        "rgb0",
    )

    result["depth0"] = visualize_depth(test_result["depth0"]).reshape((test_H, test_W, 3))

    result["rgb"] = to8b(
        torch.reshape(test_result["rgb"], (test_H, test_W, 3)).cpu().detach().numpy(),
        "rgb",
    )

    result["depth"] = visualize_depth(test_result["depth"]).reshape((test_H, test_W, 3))

    # Visualize static and dynamic nerfs separately
    result["rgb_static0"] = to8b(
        torch.reshape(test_result["rgb_static0"], (test_H, test_W, 3))
        .cpu()
        .detach()
        .numpy(),
        "rgb_static0",
    )

    result["rgb_dynamic0s"] = to8b(
        test_result["rgb_dynamic0"]
        .transpose(0, 1)
        .reshape((num_vehicles, test_H, test_W, 3))
        .cpu()
        .detach()
        .numpy(),
        "rgb_dynamic0s",
    )

    result["rgb_static"] = to8b(
        torch.reshape(test_result["rgb_static"], (test_H, test_W, 3)).cpu().detach().numpy(),
        "rgb_static",
    )

    result["rgb_dynamics"] = to8b(
        test_result["rgb_dynamic"]
        .transpose(0, 1)
        .reshape((num_vehicles, test_H, test_W, 3))
        .cpu()
        .detach()
        .numpy(),
        "rgb_dynamics",
    )

    result["depth_static"] = visualize_depth(test_result["depth_static"]).reshape(
        (test_H, test_W, 3)
    )

    result["depth_dynamics"] = visualize_depth(
        test_result["depth_dynamic"].transpose(0, 1), test_H, test_W
    )

    result["depth_static0"] = visualize_depth(test_result["depth_static0"]).reshape(
        (test_H, test_W, 3)
    )

    result["depth_dynamic0s"] = visualize_depth(
        test_result["depth_dynamic0"].transpose(0, 1), test_H, test_W
    )

    result["iou_2d"], result["predicted_masks"] = compute_2d_iou(
        test_result["dynamic_transmittance"],
        batch["semantic_mask"][frame],
    )

    if bbox_local_vertices is not None:
        inv_estimated_relative_pose = invert_transformation(estimated_relative_pose)
        inv_gt_pose = invert_transformation(gt_pose)
        pose = np.einsum(
            "vki,vij->vkj", inv_estimated_relative_pose, invert_transformation(gt_pose0)
        )
        result["iou_3d"], result["bboxes"], result["gt_bboxes"] = compute_3d_iou(
            pose, inv_gt_pose, bbox_local_vertices
        )

    return result
