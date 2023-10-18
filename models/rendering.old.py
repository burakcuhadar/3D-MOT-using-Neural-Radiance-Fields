"""
Revert:
    - torch.finfo => 1e-7
    - @typechecked uncomment, TensorType uncomment
"""

from typing import Optional, Union, Tuple
import collections

import torch
import pypose as pp

import torch.nn.functional as F
import numpy as np

from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.volrend import rendering

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from .types import NerfNetworkOutput, StarNetworkOutput, StarRenderOutput

patch_typeguard()


def img2mse(img1, img2):
    return torch.mean((img1 - img2) ** 2)


def mse2psnr(mse):
    return -10.0 * torch.log(mse) / torch.log(torch.tensor([10.0], device=mse.device))


def compute_entropy(alpha_static, alpha_dynamic=None):
    eps = 1e-7
    alpha_static_clamp = alpha_static.clamp(min=eps, max=1 - eps)
    alpha_dynamic_clamp = alpha_dynamic.clamp(min=eps, max=1 - eps)

    entropy = -torch.mean(
        alpha_static * torch.log(alpha_static_clamp)
        + (1 - alpha_static) * torch.log1p(-alpha_static_clamp)
    )
    entropy += -torch.mean(
        alpha_dynamic * torch.log(alpha_dynamic_clamp)
        + (1 - alpha_dynamic) * torch.log1p(-alpha_dynamic_clamp)
    )

    total_alpha = alpha_static + alpha_dynamic
    static_normed_trans = alpha_static / total_alpha.clamp(min=eps)
    static_normed_trans_clamp = static_normed_trans.clamp(min=eps)
    dynamic_normed_trans = alpha_dynamic / total_alpha.clamp(min=eps)
    dynamic_normed_trans_clamp = dynamic_normed_trans.clamp(min=eps)

    entropy += -torch.mean(
        total_alpha
        * (
            static_normed_trans * static_normed_trans_clamp.log()
            + dynamic_normed_trans * dynamic_normed_trans_clamp.log()
        )
    )

    return entropy



# to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
def to8b(img, debug_type):
    with np.errstate(all="raise"):
        try:
            return (255 * np.clip(img, 0, 1)).astype(np.uint8)
        except Exception as _:
            # print('encountered error with input:', x)
            print("encountered to8b error")
            print("is there NaN?:", np.any(np.isnan(img)))
            print("type:", debug_type)
            # print('its clip is:', np.clip(x,0,1))
            # print('255 times clip:', 255*np.clip(x,0,1))
            return np.zeros_like(img, dtype=np.uint8)


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="xy"
    )
    dirs = torch.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
    )

    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape).clone()
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


@typechecked
def sample_pts(
    rays_o: TensorType["num_rays", 3],
    rays_d: TensorType["num_rays", 3],
    near,
    far,
    N_samples: int,
    perturb=0,
    lindisp=False,
    is_train=True,
) -> Tuple[
        TensorType["num_rays", "num_samples", 3], TensorType["num_rays", "num_samples"]
    ]:
    
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=rays_o.device)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    N_rays = rays_o.shape[0]
    z_vals = z_vals.expand([N_rays, N_samples])

    if is_train and perturb > 0.0:
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)
        z_vals = lower + (upper - lower) * t_rand

    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples, 3]

    return pts, z_vals


def render_star_appinit(
    star_network, 
    pts, 
    viewdirs,
    z_vals, 
    rays_o, 
    rays_d, 
    N_importance, 
):
    # Pass through coarse network
    result_coarse = star_network(pts, viewdirs, z_vals, rays_d, is_coarse=True)

    # Hierarchical volume sampling
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    """z_samples = sample_pdf( 
        z_vals[..., :-1, None],
        z_vals[..., 1:, None],
        result_coarse["weights"],
        N_importance,
        det=(not star_network.training),
    )"""
    z_samples = sample_pdf(
        z_vals_mid,
        result_coarse["weights"][..., 1:-1],
        N_importance,
        det=(not star_network.training),
    )
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples + N_importance, 3]

    result_fine = star_network(pts, viewdirs, z_vals, rays_d, is_coarse=False)

    # For visualization
    z_std = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    result = {}
    result |= result_fine
    for k, v in result_coarse.items():
        result[f"{k}0"] = v
    result["z_std"] = z_std

    return result


def render_nerf(
    nerf_coarse,
    nerf_fine,
    pts,
    viewdirs,
    z_vals,
    rays_o,
    rays_d,
    N_importance,
    far_dist,
):
    # Pass through coarse network
    raw_alpha_coarse, raw_rgb_coarse = nerf_coarse(pts, viewdirs, step=None)
    result_coarse = raw2outputs(
        raw_alpha_coarse,
        raw_rgb_coarse,
        z_vals,
        rays_d,
        nerf_coarse.raw_noise_std if nerf_coarse.training else 0,
        nerf_coarse.white_bkgd,
        far_dist=far_dist,
    )

    # Hierarchical volume sampling
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf( #TODO change using
        z_vals_mid,
        result_coarse["weights"][..., 1:-1],
        N_importance,
        det=(not nerf_coarse.training),
    )
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples + N_importance, 3]

    raw_alpha_fine, raw_rgb_fine = nerf_fine(pts, viewdirs, step=None)
    result_fine = raw2outputs(
        raw_alpha_fine,
        raw_rgb_fine,
        z_vals,
        rays_d,
        nerf_fine.raw_noise_std if nerf_fine.training else 0,
        nerf_fine.white_bkgd,
        far_dist=far_dist,
    )

    # For visualization
    z_std = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    result = {}
    result |= result_fine
    for k, v in result_coarse.items():
        result[f"{k}0"] = v
    result["z_std"] = z_std

    return result




@typechecked
def render_star_online(
    star_network,
    pts: TensorType["num_rays", "num_samples", 3],
    viewdirs: TensorType["num_rays", 3],
    z_vals: TensorType["num_rays", "num_samples"],
    rays_o: TensorType["num_rays", 3],
    rays_d: TensorType["num_rays", 3],
    N_importance: int,
    pose: Union[TensorType[4, 4], TensorType[6], TensorType[7], pp.lietensor.lietensor.LieTensor],
    step: Optional[int] = None,
) -> StarRenderOutput:
    if N_importance <= 0:
        raise NotImplementedError

    result_coarse = star_network(
        pts, viewdirs, z_vals, rays_d, pose, is_coarse=True, step=step
    )

    # Hierarchical volume sampling
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid, 
        result_coarse["weights"][..., 1:-1],
        N_importance, 
        det=(not star_network.training),
    )
    """z_samples = sample_pdf(
        z_vals[..., :-1, None],
        z_vals[..., 1:, None],
        result_coarse["weights"],
        N_importance,
        det=(not star_network.training),
    )"""
    z_samples = z_samples.detach()
    
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples + N_importance + 1, 3]

    result_fine = star_network(
        pts, viewdirs, z_vals, rays_d, pose, is_coarse=False, step=step
    )

    result = {}
    result |= result_fine
    for k, v in result_coarse.items():
        result[f"{k}0"] = v

    result["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    return result


'''def raw2alpha(raw, dists, act_fn=F.relu):
    return 1.0 - torch.exp(-act_fn(raw) * dists)'''
def raw2alpha(raw, dists):
    #return 1.0 - torch.exp(-F.relu(raw) * dists)
    #TODO try with softplus instead of relu
    return 1.0 - torch.exp(-F.softplus(raw) * dists)

@typechecked
def raw2outputs(
    raw_alpha: TensorType["num_rays", "num_samples"],
    raw_rgb: TensorType["num_rays", "num_samples", 3],
    z_vals: TensorType["num_rays", "num_samples"],
    rays_d: TensorType["num_rays", 3],
    raw_noise_std: float,
    white_bkgd: bool,
    far_dist: float,
) -> NerfNetworkOutput:
    device = raw_alpha.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # [N_rays,N_samples]
    dists = torch.cat(
        [dists, torch.tensor([far_dist], device=device).expand(dists[..., :1].shape)],
        -1,
    )

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    # if torch.any(dists <= 1e-10):
    #     if dists[dists <= 1e-10].shape[0] > 100:
    #         #print("dists has 0: ", dists[dists <= 1e-10].shape[0])
    #         print(z_vals)
    zero_dist_count = dists[dists <= 1e-10].shape[0]

    rgb = torch.sigmoid(raw_rgb)  # [N_rays, N_samples, 3]
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw_alpha.shape) * raw_noise_std

    alpha = raw2alpha(raw_alpha + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    weights = (
        alpha
        * torch.cumprod(
            torch.cat(
                [torch.ones((alpha.shape[0], 1), device=device), 1.0 - alpha + 1e-10],
                -1,
            ),
            -1,
        )[:, :-1]
    )
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)

    weights_sum = torch.sum(weights, -1)
    weights_sum = torch.where(
        weights_sum >= 0, weights_sum, 1e-7
    )
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / weights_sum
    )
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    result = {
        "rgb": rgb_map,
        "disp": disp_map,
        "acc": acc_map,
        "weights": weights,
        "depth": depth_map,
        "dists": dists,  # used for sigma loss
        "z_vals": z_vals,  # used for sigma loss
        "zero_dist_count": zero_dist_count
    }

    # I think in the paper they dont use regularization for appearance init. But it may still be helpful,
    # do experiments.
    # if ret_entropy:
    #    result["entropy"] = compute_entropy(alpha)

    return result

"""
@typechecked
def raw2outputs_star(
    raw_alpha_static: TensorType["num_rays", "num_samples"],
    raw_rgb_static: TensorType["num_rays", "num_samples", 3],
    raw_alpha_dynamic: TensorType["num_rays", "num_samples"],
    raw_rgb_dynamic: TensorType["num_rays", "num_samples", 3],
    z_vals: TensorType["num_rays", "num_samples"],
    rays_d: TensorType["num_rays", 3],
    raw_noise_std: float,
    white_bkgd: bool,
    far_dist: float,
) -> StarNetworkOutput:
    device = raw_alpha_static.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # [N_rays,N_samples]
    dists = torch.cat(
        [dists, torch.tensor([far_dist], device=device).expand(dists[..., :1].shape)],
        -1,
    )

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    zero_dist_count = dists[dists <= 1e-10].shape[0]
    # if torch.any(dists <= 1e-10):
    #     print("z_vals: ")
    #     print(z_vals)
    #     print("z_vals dtype: ", z_vals.dtype)
    
    rgb_static = torch.sigmoid(raw_rgb_static)  # [N_rays, N_samples, 3]
    rgb_dynamic = torch.sigmoid(raw_rgb_dynamic)  # [N_rays, N_samples, 3]

    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw_alpha_static.shape) * raw_noise_std

    alpha_static = raw2alpha(raw_alpha_static + noise, dists)  # [N_rays, N_samples]
    alpha_dynamic = raw2alpha(raw_alpha_dynamic + noise, dists)  # [N_rays, N_samples]
    alpha_total = raw2alpha(raw_alpha_static + noise + raw_alpha_dynamic + noise, dists)

    T_s = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_static.shape[0], 1), device=device),
                1.0 - alpha_static + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    T_d = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_dynamic.shape[0], 1), device=device),
                1.0 - alpha_dynamic + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    # T = T_s * T_d
    T = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_total.shape[0], 1), device=device),
                1.0 - alpha_total + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    rgb_map = torch.sum(
        T[..., None]
        * (
            alpha_static[..., None] * rgb_static
            + alpha_dynamic[..., None] * rgb_dynamic
        ),
        dim=-2,
    )

    # Only for visualization
    rgb_map_static = torch.sum(
        T_s[..., None] * alpha_static[..., None] * rgb_static, dim=-2
    )
    rgb_map_dynamic = torch.sum(
        T_d[..., None] * alpha_dynamic[..., None] * rgb_dynamic, dim=-2
    )
    dynamic_weights = T_d * alpha_dynamic
    depth_dynamic = torch.sum(dynamic_weights * z_vals, -1)
    static_weights = T_s * alpha_static
    depth_static = torch.sum(static_weights * z_vals, -1)

    # weights = T * (alpha_static + alpha_dynamic)
    weights = T * alpha_total  # [N_rays, N_samples]
    # weights = T_s * T_d * alpha_total

    depth_map = torch.sum(weights * z_vals, -1)

    weights_sum = torch.sum(weights, -1)
    weights_sum = torch.where(
        weights_sum >= 0, weights_sum, 1e-10
    )
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / weights_sum
    )
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    entropy = compute_entropy(alpha_static, alpha_dynamic)

    #TODO remove these
    if torch.any(torch.isnan(rgb_map)):
        print("rgb_map has nan")
    if torch.any(torch.isnan(depth_map)):
        print("depth_map has nan")
    if torch.any(torch.isnan(weights)):
        print("weights has nan")

    return {
        "rgb": rgb_map,
        "disp": disp_map,
        "acc": acc_map,
        "weights": weights,
        "depth": depth_map,
        "rgb_static": rgb_map_static,
        "rgb_dynamic": rgb_map_dynamic,
        "depth_static": depth_static,
        "depth_dynamic": depth_dynamic,
        "entropy": entropy,
        "dists": dists,  # used for sigma loss
        "z_vals": z_vals,  # used for sigma loss
        "zero_dist_count": zero_dist_count
    }
"""

@typechecked
def raw2outputs_star(
    raw_alpha_static: TensorType["num_rays", "num_samples"],
    raw_rgb_static: TensorType["num_rays", "num_samples", 3],
    raw_alpha_dynamic: TensorType["num_rays", "num_samples"],
    raw_rgb_dynamic: TensorType["num_rays", "num_samples", 3],
    z_vals: TensorType["num_rays", "num_samples"],
    rays_d: TensorType["num_rays", 3],
    raw_noise_std: float,
    white_bkgd: bool,
    far_dist: float,
) -> StarNetworkOutput:
    device = raw_alpha_static.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # [N_rays,N_samples]
    dists = torch.cat(
        [dists, torch.tensor([far_dist], device=device).expand(dists[..., :1].shape)],
        -1,
    )

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    zero_dist_count = dists[dists <= 1e-10].shape[0]
    # if torch.any(dists <= 1e-10):
    #     print("z_vals: ")
    #     print(z_vals)
    #     print("z_vals dtype: ", z_vals.dtype)
    
    rgb_static = torch.sigmoid(raw_rgb_static)  # [N_rays, N_samples, 3]
    rgb_dynamic = torch.sigmoid(raw_rgb_dynamic)  # [N_rays, N_samples, 3]

    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw_alpha_static.shape) * raw_noise_std

    alpha_static = raw2alpha(raw_alpha_static + noise, dists)  # [N_rays, N_samples]
    alpha_dynamic = raw2alpha(raw_alpha_dynamic + noise, dists)  # [N_rays, N_samples]
    alpha_total = raw2alpha(raw_alpha_static + noise + raw_alpha_dynamic + noise, dists)

    T_s = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_static.shape[0], 1), device=device),
                1.0 - alpha_static + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    T_d = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_dynamic.shape[0], 1), device=device),
                1.0 - alpha_dynamic + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    # T = T_s * T_d
    T = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_total.shape[0], 1), device=device),
                1.0 - alpha_total + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    rgb_map = torch.sum(
        T[..., None]
        * (
            alpha_static[..., None] * rgb_static
            + alpha_dynamic[..., None] * rgb_dynamic
        ),
        dim=-2,
    )

    # Only for visualization
    rgb_map_static = torch.sum(
        T_s[..., None] * alpha_static[..., None] * rgb_static, dim=-2
    )
    rgb_map_dynamic = torch.sum(
        T_d[..., None] * alpha_dynamic[..., None] * rgb_dynamic, dim=-2
    )
    dynamic_weights = T_d * alpha_dynamic
    depth_dynamic = torch.sum(dynamic_weights * z_vals, -1)
    static_weights = T_s * alpha_static
    depth_static = torch.sum(static_weights * z_vals, -1)

    # weights = T * (alpha_static + alpha_dynamic)
    weights = T * alpha_total  # [N_rays, N_samples]
    # weights = T_s * T_d * alpha_total

    depth_map = torch.sum(weights * z_vals, -1)

    weights_sum = torch.sum(weights, -1)
    weights_sum = torch.where(
        weights_sum >= 0, weights_sum, 1e-10
    )
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / weights_sum
    )
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    entropy = compute_entropy(alpha_static, alpha_dynamic)

    #TODO remove these
    if torch.any(torch.isnan(rgb_map)):
        print("rgb_map has nan")
    if torch.any(torch.isnan(depth_map)):
        print("depth_map has nan")
    if torch.any(torch.isnan(weights)):
        print("weights has nan")

    return {
        "rgb": rgb_map,
        "disp": disp_map,
        "acc": acc_map,
        "weights": weights,
        "depth": depth_map,
        "rgb_static": rgb_map_static,
        "rgb_dynamic": rgb_map_dynamic,
        "depth_static": depth_static,
        "depth_dynamic": depth_dynamic,
        "entropy": entropy,
        "dists": dists,  # used for sigma loss
        "z_vals": z_vals,  # used for sigma loss
        "zero_dist_count": zero_dist_count
    }



@typechecked
def raw2outputs_uorf(
    raw_alpha_static: TensorType["num_rays", "num_samples"],
    raw_rgb_static: TensorType["num_rays", "num_samples", 3],
    raw_alpha_dynamic: TensorType["num_rays", "num_samples"],
    raw_rgb_dynamic: TensorType["num_rays", "num_samples", 3],
    z_vals: TensorType["num_rays", "num_samples"],
    rays_d: TensorType["num_rays", 3],
    raw_noise_std: float,
    white_bkgd: bool,
    far_dist: float,
) -> StarNetworkOutput:
    device = raw_alpha_static.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # [N_rays,N_samples]
    dists = torch.cat(
        [dists, torch.tensor([far_dist], device=device).expand(dists[..., :1].shape)],
        -1,
    )

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    zero_dist_count = dists[dists <= 1e-10].shape[0]
    # if torch.any(dists <= 1e-10):
    #     print("z_vals: ")
    #     print(z_vals)
    #     print("z_vals dtype: ", z_vals.dtype)
    
    rgb_static = torch.sigmoid(raw_rgb_static)  # [N_rays, N_samples, 3]
    rgb_dynamic = torch.sigmoid(raw_rgb_dynamic)  # [N_rays, N_samples, 3]

    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw_alpha_static.shape) * raw_noise_std

    #alpha_static = raw2alpha(raw_alpha_static + noise, dists)  # [N_rays, N_samples]
    #alpha_dynamic = raw2alpha(raw_alpha_dynamic + noise, dists)  # [N_rays, N_samples]
    #alpha_total = raw2alpha(raw_alpha_static + noise + raw_alpha_dynamic + noise, dists)
    
    weights_total = raw_alpha_static + raw_alpha_dynamic
    weights_total = torch.where(weights_total > 0, weights_total, 1e-7)
    weights_static = raw_alpha_static / weights_total
    weights_dynamic = raw_alpha_dynamic / weights_total

    density_composition = weights_static * raw_alpha_static + weights_dynamic * raw_alpha_dynamic
    color_composition = weights_static[..., None] * raw_rgb_static + weights_dynamic[..., None] * raw_rgb_dynamic

    alpha_total = raw2alpha(density_composition + noise, dists)
    alpha_static = raw2alpha(weights_static * raw_alpha_static + noise, dists)
    alpha_dynamic = raw2alpha(weights_dynamic * raw_alpha_dynamic + noise, dists)
    
    T_s = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_static.shape[0], 1), device=device),
                1.0 - alpha_static + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    T_d = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_dynamic.shape[0], 1), device=device),
                1.0 - alpha_dynamic + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    T = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_total.shape[0], 1), device=device),
                1.0 - alpha_total + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    

    rgb_map = torch.sum(
        T[..., None]
        * alpha_total[..., None]
        * color_composition,
        dim=-2,
    )

    # Only for visualization
    rgb_map_static = torch.sum(
        T_s[..., None] * alpha_static[..., None] * weights_static[..., None] * rgb_static, dim=-2
    )
    rgb_map_dynamic = torch.sum(
        T_d[..., None] * alpha_dynamic[..., None] * weights_dynamic[..., None] * rgb_dynamic, dim=-2
    )

    dynamic_weights = T_d * alpha_dynamic * weights_dynamic
    depth_dynamic = torch.sum(dynamic_weights * z_vals, -1)
    static_weights = T_s * alpha_static * weights_static
    depth_static = torch.sum(static_weights * z_vals, -1)

    # weights = T * (alpha_static + alpha_dynamic)
    weights = T * alpha_total  # [N_rays, N_samples]
    # weights = T_s * T_d * alpha_total

    depth_map = torch.sum(weights * z_vals, -1)

    weights_sum = torch.sum(weights, -1)
    weights_sum = torch.where(
        weights_sum >= 0, weights_sum, 1e-10
    )
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / weights_sum
    )
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    entropy = compute_entropy(alpha_static, alpha_dynamic)

    #TODO remove these
    if torch.any(torch.isnan(rgb_map)):
        print("rgb_map has nan")
    if torch.any(torch.isnan(depth_map)):
        print("depth_map has nan")
    if torch.any(torch.isnan(weights)):
        print("weights has nan")

    return {
        "rgb": rgb_map,
        "disp": disp_map,
        "acc": acc_map,
        "weights": weights,
        "depth": depth_map,
        "rgb_static": rgb_map_static,
        "rgb_dynamic": rgb_map_dynamic,
        "depth_static": depth_static,
        "depth_dynamic": depth_dynamic,
        "entropy": entropy,
        "dists": dists,  # used for sigma loss
        "z_vals": z_vals,  # used for sigma loss
        "zero_dist_count": zero_dist_count
    }


'''
 def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        ray_samples: Optional[RaySamples] = None,
        weights: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle: Rays to generate samples for
            ray_samples: Existing ray samples
            weights: Weights for each bin
            num_samples: Number of samples per ray
            eps: Small value to prevent numerical issues.

        Returns:
            Positions and deltas for samples along a ray
        """

        if ray_samples is None or ray_bundle is None:
            raise ValueError("ray_samples and ray_bundle must be provided")
        assert weights is not None, "weights must be provided"

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_bins = num_samples + 1

        weights = weights[..., 0] + self.histogram_padding

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if self.train_stratified and self.training:
            # Stratified samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
            u = u + rand
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        assert (
            ray_samples.spacing_starts is not None and ray_samples.spacing_ends is not None
        ), "ray_sample spacing_starts and spacing_ends must be provided"
        assert ray_samples.spacing_to_euclidean_fn is not None, "ray_samples.spacing_to_euclidean_fn must be provided"
        existing_bins = torch.cat(
            [
                ray_samples.spacing_starts[..., 0],
                ray_samples.spacing_ends[..., -1:, 0],
            ],
            dim=-1,
        )

        inds = torch.searchsorted(cdf, u, side="right")
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        if self.include_original:
            bins, _ = torch.sort(torch.cat([existing_bins, bins], -1), -1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
        )

        return ray_samples
'''


def sample_pdf(bins, weights, N_samples, det=False):
    device = weights.device
    eps=torch.finfo(torch.float32).eps

    # Get pdf
    weights = weights + eps  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, side="right")
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# Adapted from nerfstudio
"""
def sample_pdf(spacing_starts, spacing_ends, weights, N_samples, det=False):
    #print("spacing_starts:", spacing_starts.shape)
    #print("spacing_ends:", spacing_ends.shape)
    #print("weights:", weights.shape)
    weights = weights[:, :, None] # to make compatible with nerfstudio code

    histogram_padding = 0.01
    eps = 1e-5
    weights = weights[..., 0] + histogram_padding  
    num_bins = N_samples + 1

    # Add small offset to rays with zero weight to prevent NaNs
    weights_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.relu(eps - weights_sum)
    weights = weights + padding / weights.shape[-1]
    weights_sum += padding

    pdf = weights / weights_sum
    cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if not det:
        # Stratified samples between 0 and 1
        u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
        u = u.expand(size=(*cdf.shape[:-1], num_bins))
        rand = torch.rand((*cdf.shape[:-1], N_samples + 1), device=cdf.device) / num_bins
        u = u + rand
    else:
        # Uniform samples between 0 and 1
        u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
        u = u + 1.0 / (2 * num_bins)
        u = u.expand(size=(*cdf.shape[:-1], num_bins))
    u = u.contiguous()

    existing_bins = torch.cat(
        [
            spacing_starts[..., 0],
            spacing_ends[..., -1:, 0],
        ],
        dim=-1,
    )
    inds = torch.searchsorted(cdf, u, side="right")
    below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
    above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
    cdf_g0 = torch.gather(cdf, -1, below)
    bins_g0 = torch.gather(existing_bins, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g1 = torch.gather(existing_bins, -1, above)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    
    return samples
"""

# Adapted from https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
'''
def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples
'''

###############
# For Nerfacc #
###############

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )




