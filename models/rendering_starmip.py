from typing import Optional, Union, Tuple

import torch

import torch.nn.functional as F
import numpy as np

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from .types__ import StarMipAppInitOutput, StarMipOnlineOutput
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)

from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from models.rendering__ import (
    compute_alpha_entropy,
    compute_dynamic_vs_static_reg,
    compute_ray_reg,
    compute_static_reg,
    compute_dynamic_reg,
)

patch_typeguard()


# Adapted from NerfStudio
def get_weights_alphas_transmittance(deltas, densities):
    """Return weights based on predicted densities

    Args:
        densities: Predicted densities for samples along ray

    Returns:
        Weights for each sample
    """
    if len(densities.shape) == 3:
        delta_density = deltas * densities
    elif len(densities.shape) == 4:
        delta_density = deltas[:, None, :, :] * densities
    else:
        raise ValueError("Invalid shape for densities")

    alphas = 1 - torch.exp(-delta_density)

    transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance = torch.cat(
        [
            torch.zeros((*transmittance.shape[:-2], 1, 1), device=densities.device),
            transmittance,
        ],
        dim=-2,
    )
    transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

    weights = alphas * transmittance  # [..., "num_samples"]
    weights = torch.nan_to_num(weights)

    return weights, alphas, transmittance


@typechecked
def get_starmip_appinit_outputs(
    density_static: TensorType["num_rays", "num_samples", 1],
    rgb_static: TensorType["num_rays", "num_samples", 3],
    deltas: TensorType["num_rays", "num_samples", 1],
    samples: RaySamples,
    depth_renderer: DepthRenderer,
    acc_renderer: AccumulationRenderer,
) -> StarMipAppInitOutput:
    weights, alpha, transmittance = get_weights_alphas_transmittance(
        deltas, density_static
    )

    rgb_map = torch.sum(transmittance * alpha * rgb_static, dim=-2)

    depth_map = depth_renderer(weights, samples)
    acc_map = acc_renderer(weights)

    return {
        "rgb": rgb_map,
        "acc": acc_map,
        "weights": weights,
        "depth": depth_map,
    }


@typechecked
def render_dynamic_depths(
    weights_d: TensorType["num_rays", "num_vehicles", "num_samples", 1],
    samples: RaySamples,
    depth_renderer: DepthRenderer,
) -> TensorType["num_rays", "num_vehicles", 1]:
    num_rays = weights_d.shape[0]
    num_vehicles = weights_d.shape[1]
    depth_dynamic = torch.zeros((num_rays, num_vehicles, 1), device=weights_d.device)

    for i in range(num_vehicles):
        depth_dynamic[:, i, :] = depth_renderer(weights_d[:, i, :, :], samples)

    return depth_dynamic


@typechecked
def get_starmip_online_outputs(
    density_static: TensorType["num_rays", "num_samples", 1],
    rgb_static: TensorType["num_rays", "num_samples", 3],
    density_dynamic: TensorType["num_rays", "num_vehicles", "num_samples", 1],
    rgb_dynamic: TensorType["num_rays", "num_vehicles", "num_samples", 3],
    deltas: TensorType["num_rays", "num_samples", 1],
    samples: RaySamples,  # NOTE: only for depth renderer
    depth_renderer: DepthRenderer,
    acc_renderer: AccumulationRenderer,
) -> StarMipOnlineOutput:
    weights_s, alpha_s, transmittance_s = get_weights_alphas_transmittance(
        deltas, density_static
    )

    weights_d, alpha_d, transmittance_d = get_weights_alphas_transmittance(
        deltas, density_dynamic
    )

    total_density = density_static + density_dynamic.sum(dim=1)
    weights, alpha, transmittance = get_weights_alphas_transmittance(
        deltas, total_density
    )

    rgb_map = torch.sum(
        transmittance
        * (alpha_s * rgb_static + torch.sum(alpha_d * rgb_dynamic, dim=1)),
        dim=-2,
    )

    depth_map = depth_renderer(weights, samples)
    acc_map = acc_renderer(weights)

    # Static outputs
    rgb_map_static = torch.sum(transmittance_s * alpha_s * rgb_static, dim=-2)
    depth_static = depth_renderer(weights_s, samples)

    # Dynamic outputs
    rgb_map_dynamic = torch.sum(transmittance_d * alpha_d * rgb_dynamic, dim=-2)
    depth_dynamic = render_dynamic_depths(weights_d, samples, depth_renderer)

    # Regularization terms
    loss_alpha_entropy = compute_alpha_entropy(alpha_s, alpha_d)
    loss_dynamic_vs_static_reg = compute_dynamic_vs_static_reg(
        density_static, density_dynamic, total_density, alpha_s, alpha_d
    )

    loss_ray_reg = compute_ray_reg(density_dynamic, total_density)
    loss_static_reg = compute_static_reg(transmittance_s, alpha_s)
    loss_dynamic_reg = compute_dynamic_reg(density_dynamic)

    return {
        "rgb": rgb_map,
        "acc": acc_map,
        "weights": weights,
        "depth": depth_map,
        "rgb_static": rgb_map_static,
        "depth_static": depth_static,
        "rgb_dynamic": rgb_map_dynamic,
        "depth_dynamic": depth_dynamic[..., 0],
        "dynamic_transmittance": transmittance_d[:, :, -1, :],
        # Regularization terms
        "loss_alpha_entropy": loss_alpha_entropy,
        "loss_dynamic_vs_static_reg": loss_dynamic_vs_static_reg,
        "loss_ray_reg": loss_ray_reg,
        "loss_static_reg": loss_static_reg,
        "loss_dynamic_reg": loss_dynamic_reg,
    }
