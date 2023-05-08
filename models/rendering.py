import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import imageio
from tqdm import tqdm
from tqdm.contrib import tenumerate

# from utils.io import device

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = (
    lambda x: -10.0 * torch.log(x) / torch.log(torch.tensor([10.0], device=x.device))
)


# to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
def to8b(x, debug_type):
    with np.errstate(all="raise"):
        try:
            return (255 * np.clip(x, 0, 1)).astype(np.uint8)
        except Exception as e:
            # print('encountered error with input:', x)
            print("encountered to8b error")
            print("is there NaN?:", np.any(np.isnan(x)))
            print("type:", debug_type)
            # print('its clip is:', np.clip(x,0,1))
            # print('255 times clip:', 255*np.clip(x,0,1))
            return np.zeros_like(x, dtype=np.uint8)


# Ray helpers
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


def sample_pts(
    rays_o, rays_d, near, far, N_samples, perturb=0, lindisp=False, is_train=True
):
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
    star_network, pts, viewdirs, z_vals, rays_o, rays_d, N_importance
):
    # Pass through coarse network
    result_coarse = star_network(pts, viewdirs, z_vals, rays_d, is_coarse=True)

    # Hierarchical volume sampling
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
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
    result.update(result_fine)
    for k, v in result_fine.items():
        result[k + "0"] = v
    result["z_std"] = z_std

    return result


def render_star_online(
    star_network, pts, viewdirs, z_vals, rays_o, rays_d, N_importance, pose, step
):
    # TODO can pose be None
    # TODO do we need to input object_pose to network?

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
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples + N_importance, 3]

    result_fine = star_network(
        pts, viewdirs, z_vals, rays_d, pose, is_coarse=False, step=step
    )

    result = {}
    result.update(result_fine)
    for k, v in result_coarse.items():
        result[k + "0"] = v

    result["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    return result


def raw2outputs(
    raw_alpha,
    raw_rgb,
    z_vals,
    rays_d,
    raw_noise_std=0,
    white_bkgd=False,
    ret_entropy=False,
):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw_alpha: [num_rays, num_samples along ray]. Raw volume density predicted by the model.
        raw_rgb: [num_rays, num_samples along ray, 3]. Raw rgb predicted by the model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    device = raw_alpha.device
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # [N_rays,N_samples]
    dists = torch.cat(
        [dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)], -1
    )

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

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
    weights_sum = torch.where(weights_sum != 0, weights_sum, 1e-10)
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
    }

    # TODO I think in the paper they dont use regularization for appearance init. But it may still be helpful,
    # do experiments.
    if ret_entropy:
        result["entropy"] = compute_entropy(alpha)

    return result


def raw2outputs_star(
    raw_alpha_static,
    raw_rgb_static,
    raw_alpha_dynamic,
    raw_rgb_dynamic,
    z_vals,
    rays_d,
    raw_noise_std=0,
    white_bkgd=False,
):
    # if torch.isnan(z_vals).any().cpu().detach().item():
    #     print('z_vals nan!')
    device = raw_alpha_static.device
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # [N_rays,N_samples]
    dists = torch.cat(
        [dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)], -1
    )

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # if torch.isnan(dists).any().cpu().detach().item():
    #     print('dists nan!')

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

    """if torch.isnan(rgb_map).any().cpu().detach().item():
        print('rgb_map nan!')
    if torch.isnan(rgb_static).any().cpu().detach().item():
        print('rgb_static nan!')
    if torch.isnan(rgb_dynamic).any().cpu().detach().item():
        print('rgb_dynamic nan!')
    if torch.isnan(alpha_static).any().cpu().detach().item():
        print('alpha_static nan!')
    if torch.isnan(alpha_dynamic).any().cpu().detach().item():
        print('alpha_dynamic nan!')
    if torch.isnan(T_s).any().cpu().detach().item():
        print('T_s nan!')
    if torch.isnan(T_d).any().cpu().detach().item():
        print('T_d nan!')
    if torch.isnan(T).any().cpu().detach().item():
        print('T nan!')"""

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

    # if torch.isnan(weights).any().cpu().detach().item():
    #     print('weights NaN!')
    # if not torch.sum(weights, -1).all().cpu().detach().item():
    #    print('weights sum zero!')

    depth_map = torch.sum(weights * z_vals, -1)
    # if torch.isnan(depth_map).any().cpu().detach().item():
    #     print('depth map NaN!')
    weights_sum = torch.sum(weights, -1)
    weights_sum = torch.where(weights_sum != 0, weights_sum, 1e-10)
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / weights_sum
    )
    # if torch.isnan(disp_map).any().cpu().detach().item():
    #     print('disp map NaN!')
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    entropy = compute_entropy(alpha_static, alpha_dynamic)

    result = {
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
    }

    return result


def compute_entropy(alpha_static, alpha_dynamic=None):
    eps = torch.finfo(alpha_static.dtype).eps
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


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
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
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
