from typing import Optional, Union, Tuple

import torch

import torch.nn.functional as F
import numpy as np

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from utils import constants

from .types__ import NerfNetworkOutput, StarNetworkOutput, StarRenderOutput

patch_typeguard()


def img2mse(img1, img2):
    return torch.mean((img1 - img2) ** 2)


def mse2psnr(mse):
    return -10.0 * torch.log(mse) / torch.log(torch.tensor([10.0], device=mse.device))


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
    N_samples,
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
    z_samples = sample_pdf(
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
    pose: Union[TensorType["num_vehicles", 4, 4], TensorType["num_vehicles", 7]],
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
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    """z_samples = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples.detach()"""
    # z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples + N_importance, 3]

    result_fine = star_network(
        pts, viewdirs, z_vals, rays_d, pose, is_coarse=False, step=step
    )

    result = {}
    # result |= result_fine
    for k, v in result_fine.items():
        result[k] = v
    for k, v in result_coarse.items():
        result[f"{k}0"] = v

    result["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    return result


def raw2alpha(raw, dists, act_fn=F.relu):
    # return 1.0 - torch.exp(-act_fn(raw) * dists)
    return 1.0 - torch.exp(-F.softplus(raw) * dists)


def sigma2alpha(sigma, dists):
    # return 1.0 - torch.exp(-act_fn(raw) * dists)
    return 1.0 - torch.exp(-sigma * dists)


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
    # TODO #zero_dist_count = dists[dists <= 1e-10].shape[0]

    rgb = torch.sigmoid(raw_rgb)  # [N_rays, N_samples, 3]
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw_alpha.shape) * raw_noise_std

    alpha = raw2alpha(raw_alpha + noise, dists)  # [N_rays, N_samples]

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
    weights_sum = torch.where(weights_sum >= 0, weights_sum, 1e-7)
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
        # TODO #"zero_dist_count": zero_dist_count
    }

    # I think in the paper they dont use regularization for appearance init. But it may still be helpful,
    # do experiments.
    # if ret_entropy:
    #    result["entropy"] = compute_entropy(alpha)

    return result


@typechecked
def raw2outputs_star(
    raw_alpha_static: TensorType["num_rays", "num_samples"],
    raw_rgb_static: TensorType["num_rays", "num_samples", 3],
    raw_alpha_dynamic: TensorType["num_rays", "num_vehicles", "num_samples"],
    raw_rgb_dynamic: TensorType["num_rays", "num_vehicles", "num_samples", 3],
    z_vals: TensorType["num_rays", "num_samples"],
    rays_d: TensorType["num_rays", 3],
    raw_noise_std=0,
    white_bkgd=False,
    far_dist=1e10,
) -> StarNetworkOutput:
    device = raw_alpha_static.device

    raw_alpha_sum = raw_alpha_static + raw_alpha_dynamic.sum(dim=1)
    raw_alpha_sum = raw_alpha_sum.clamp(min=constants.EPS)

    sigma_s = F.softplus(raw_alpha_static)
    sigma_d = F.softplus(raw_alpha_dynamic)
    sigma_sum = sigma_s + sigma_d.sum(dim=1)
    # sigma_sum = torch.where(sigma_sum > 0, sigma_sum, constants.EPS)
    # sigma_sum += constants.EPS # to prevent division by 0 #TODO also try clamp
    sigma_sum = sigma_sum.clamp(min=constants.EPS)

    rgb_static = torch.sigmoid(raw_rgb_static)  # [N_rays, N_samples, 3]
    rgb_dynamic = torch.sigmoid(raw_rgb_dynamic)  # [N_rays, num_vehicles, N_samples, 3]

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # [N_rays,N_samples]
    dists_s = torch.cat(
        [dists, torch.tensor([far_dist], device=device).expand(dists[..., :1].shape)],
        -1,
    )
    dists_d = torch.cat(
        [
            dists,
            torch.tensor([constants.EPS], device=device).expand(dists[..., :1].shape),
        ],
        -1,
    )

    dists_s = dists_s * torch.norm(rays_d[..., None, :], dim=-1)
    dists_d = dists_d * torch.norm(rays_d[..., None, :], dim=-1)

    alpha_static = sigma2alpha(sigma_s, dists_s)  # [N_rays, N_samples]
    alpha_dynamic = sigma2alpha(
        sigma_d, dists_d[:, None, :]
    )  # [N_rays, num_vehicles, N_samples]

    # alpha_total = raw2alpha(
    #     raw_alpha_static + noise + raw_alpha_dynamic.sum(dim=1) + noise, dists
    # )
    alpha_total = 1.0 - torch.exp(-(sigma_s * dists_s + sigma_d.sum(dim=1) * dists_d))

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
                torch.ones(
                    (alpha_dynamic.shape[0], alpha_dynamic.shape[1], 1), device=device
                ),
                1.0 - alpha_dynamic + 1e-10,
            ],
            -1,
        ),
        -1,
    )[
        ..., :-1
    ]  # N_rays, num_vehicles, N_samples
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
    # T = T_s * torch.prod(T_d, dim=1) #TODO which one?

    # weights_d = alpha_dynamic * T[:, None, :] #d2nerf
    weights_d = T_d * alpha_dynamic
    # weights_s = alpha_static * T #d2nerf
    weights_s = T_s * alpha_static
    weights = (weights_d.sum(dim=1) + weights_s).clamp(min=constants.EPS)
    acc = weights.sum(dim=-1)

    # d2nerf
    # rgb_map = (
    #     (weights_d[..., None] * rgb_dynamic).sum(dim=1)
    #     + weights_s[..., None] * rgb_static
    # ).sum(dim=-2)

    # weighted_rgb = (
    #     sigma_s[..., None] * rgb_static
    #     + torch.sum(sigma_d[..., None] * rgb_dynamic, dim=1)
    # ) / sigma_sum[..., None]

    weighted_rgb = torch.sigmoid(
        (
            raw_alpha_static[..., None] * raw_rgb_static
            + torch.sum(raw_alpha_dynamic[..., None] * raw_rgb_dynamic, dim=1)
        )
        / raw_alpha_sum[..., None]
    )

    # weighted_rgb = torch.sigmoid(
    #     (
    #         alpha_static[..., None] * rgb_static
    #         + torch.sum(alpha_dynamic[..., None] * rgb_dynamic, dim=1)
    #     )
    #     / (alpha_total[..., None] + 1e-10)
    # )

    if torch.any(torch.isnan(weighted_rgb)):
        print("weighted rgb has nan")
        if torch.any(torch.isnan(sigma_sum)):
            print("sigma sum has nan")
        if torch.any(sigma_sum <= 0):
            print("sigma sum has 0")
        if torch.any(torch.isnan(sigma_s)):
            print("sigma s has nan")
        if torch.any(torch.isnan(sigma_d)):
            print("sigma d has nan")
        if torch.any(torch.isnan(rgb_static)):
            print("rgb static has nan")
        if torch.any(torch.isnan(rgb_dynamic)):
            print("rgb dynamic has nan")

        raise ValueError

    rgb_map = torch.sum(
        T[..., None] * alpha_total[..., None] * weighted_rgb,
        dim=-2,
    )

    depth_map = (weights * z_vals).sum(dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc[..., None])

    acc_map = (weights_d.sum(dim=1) + weights_s)[..., :-1].sum(dim=-1)

    ######## Only for visualization
    rgb_map_static = torch.sum(weights_s[..., None] * rgb_static, dim=-2)

    # [N_rays, num_vehicles, N_samples, 3]
    rgb_map_dynamic = torch.sum(
        weights_d[..., None] * rgb_dynamic, dim=-2
    )  # N_rays, num_vehicles, 3
    # rgb_map_dynamic = torch.sum(
    #     weights_d[..., None]
    #     * torch.sigmoid(sigma_d[..., None] * raw_rgb_dynamic / sigma_sum[:, None, :, None]),
    #     dim=-2,
    # )

    depth_static = torch.sum(weights_s * z_vals, -1)
    depth_dynamic = torch.sum(
        weights_d * z_vals[:, None, :], -1
    )  # N_rays, num_vehicles

    weights_sum = torch.sum(weights, -1)
    weights_sum = torch.where(
        weights_sum > 0, weights_sum, torch.finfo(torch.float32).eps
    )
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / weights_sum
    )

    ###### star way
    # rgb_map = torch.sum(
    #     T[..., None]
    #     * (
    #         alpha_static[..., None] * rgb_static
    #         + torch.sum(alpha_dynamic[..., None] * rgb_dynamic, dim=1)
    #     ),
    #     dim=-2,
    # )

    """ ###### normalized way
    # TODO also try with weighting rgb_static and rgb_dynamic
    weighted_rgb = torch.sigmoid((
        sigma_s[..., None] * raw_rgb_static
        + torch.sum(sigma_d[..., None] * raw_rgb_dynamic, dim=1)
    ) / sigma_sum[..., None])

    rgb_map = torch.sum(
        T[..., None] * alpha_total[..., None] * weighted_rgb,
        dim=-2,
    )

    # Only for visualization
    rgb_map_static = torch.sum(
        T_s[..., None] * alpha_static[..., None] * rgb_static, dim=-2
    )

    rgb_map_dynamic = torch.sum(
        T_d[..., None] * alpha_dynamic[..., None] * rgb_dynamic, dim=-2
    )  # num_vehicles, N_rays, 3
    dynamic_weights = T_d * alpha_dynamic  # N_rays, num_vehicles, N_samples
    depth_dynamic = torch.sum(
        dynamic_weights * z_vals[:, None, :], -1
    )  # N_rays, num_vehicles

    static_weights = T_s * alpha_static
    depth_static = torch.sum(static_weights * z_vals, -1)

    # weights = T * (alpha_static + alpha_dynamic)
    weights = T * alpha_total  # [N_rays, N_samples]
    # weights = T_s * T_d * alpha_total

    depth_map = torch.sum(weights * z_vals, -1)

    weights_sum = torch.sum(weights, -1)
    weights_sum = torch.where(
        weights_sum >= 0, weights_sum, torch.finfo(torch.float32).eps
    )
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / weights_sum
    )
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])
    """

    # entropy = compute_entropy(alpha_static, alpha_dynamic)
    loss_alpha_entropy = compute_alpha_entropy(alpha_static, alpha_dynamic)
    loss_dynamic_vs_static_reg = compute_dynamic_vs_static_reg(
        sigma_s, sigma_d, sigma_sum, alpha_static, alpha_dynamic
    )
    loss_ray_reg = compute_ray_reg(sigma_d, sigma_sum)
    loss_static_reg = compute_static_reg(sigma_s, alpha_static)

    # print("rgb map shape:", rgb_map.shape)
    # print("disp map shape:", disp_map.shape)
    # print("acc map shape:", acc_map.shape)
    # print("weights shape:", weights.shape)
    # print("depth map shape:", depth_map.shape)
    # print("rgb map static shape:", rgb_map_static.shape)
    # print("rgb map dynamic shape:", rgb_map_dynamic.shape)
    # print("depth static shape:", depth_static.shape)
    # print("depth dynamic shape:", depth_dynamic.shape)
    # # print("entropy shape:", entropy.shape)
    # print("dists shape:", dists_s.shape)

    if torch.any(torch.isnan(rgb_map)):
        print("rgb map has nan")
        raise ValueError
    if torch.any(torch.isnan(disp_map)):
        print("disp map has nan")
        raise ValueError
    if torch.any(torch.isnan(acc_map)):
        print("acc map has nan")
        raise ValueError
    if torch.any(torch.isnan(weights)):
        print("weights has nan")
        raise ValueError
    if torch.any(torch.isnan(depth_map)):
        print("depth map has nan")
        raise ValueError
    if torch.any(torch.isnan(rgb_map_static)):
        print("rgb map static has nan")
        raise ValueError
    if torch.any(torch.isnan(rgb_map_dynamic)):
        print("rgb map dynamic has nan")
        raise ValueError
    if torch.any(torch.isnan(depth_static)):
        print("depth static has nan")
        raise ValueError
    if torch.any(torch.isnan(depth_dynamic)):
        print("depth dynamic has nan")
        raise ValueError

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
        # "entropy": entropy,
        "dists": dists_s,  # used for sigma loss #TODO remove
        "z_vals": z_vals,  # used for sigma loss #TODO remove
        # Regularization terms
        "loss_alpha_entropy": loss_alpha_entropy,
        "loss_dynamic_vs_static_reg": loss_dynamic_vs_static_reg,
        "loss_ray_reg": loss_ray_reg,
        "loss_static_reg": loss_static_reg,
    }


def compute_entropy(alpha_static, alpha_dynamic):
    alpha_static_clamp = alpha_static.clamp(min=constants.EPS, max=1 - constants.EPS)
    alpha_dynamic_clamp = alpha_dynamic.clamp(min=constants.EPS, max=1 - constants.EPS)

    entropy = -torch.mean(
        alpha_static * torch.log(alpha_static_clamp)
        + (1 - alpha_static) * torch.log1p(-alpha_static_clamp)
    )
    # [N_rays, num_vehicles, N_samples]
    entropy += -torch.mean(
        alpha_dynamic * torch.log(alpha_dynamic_clamp)
        + (1 - alpha_dynamic) * torch.log1p(-alpha_dynamic_clamp),
        (0, 2),
    ).sum()

    total_alpha = alpha_static + alpha_dynamic.sum(dim=1)  # N_rays, N_samples
    static_normed_trans = alpha_static / total_alpha.clamp(min=constants.EPS)
    static_normed_trans_clamp = static_normed_trans.clamp(min=constants.EPS)
    dynamic_normed_trans = (
        alpha_dynamic / total_alpha.clamp(min=constants.EPS)[:, None, :]
    )
    dynamic_normed_trans_clamp = dynamic_normed_trans.clamp(min=constants.EPS)

    entropy += -torch.mean(
        total_alpha
        * (
            static_normed_trans * static_normed_trans_clamp.log()
            + torch.sum(dynamic_normed_trans * dynamic_normed_trans_clamp.log(), dim=1)
        )
    )

    return entropy


def compute_alpha_entropy(alpha_s, alpha_d):
    # Computes H(alpha_s) H(alpha_d) from Star paper

    num_vehicles = alpha_d.shape[1]

    alpha_s_clamp = alpha_s.clamp(min=constants.EPS, max=1 - constants.EPS)
    alpha_d_clamp = alpha_d.clamp(min=constants.EPS, max=1 - constants.EPS)

    entropy = -torch.mean(
        alpha_s * torch.log(alpha_s_clamp) + (1 - alpha_s) * torch.log1p(-alpha_s_clamp)
    ) / (num_vehicles + 1)

    # [N_rays, num_vehicles, N_samples]
    entropy += -torch.mean(
        alpha_d * torch.log(alpha_d_clamp)
        + (1 - alpha_d) * torch.log1p(-alpha_d_clamp),
        (0, 2),
    ).sum() / (num_vehicles + 1)

    return entropy


def compute_dynamic_vs_static_reg(sigma_s, sigma_d, total_sigma, alpha_s, alpha_d):
    num_vehicles = sigma_d.shape[1]

    # Dynamic vs Static Regularization from Star
    total_alpha = alpha_s + alpha_d.sum(dim=1)  # N_rays, N_samples
    # total_alpha = total_alpha.clamp(max=1 - constants.EPS)
    static_normed = alpha_s / total_alpha.clamp(min=constants.EPS)
    static_normed = static_normed.clamp(min=constants.EPS)
    dynamic_normed = alpha_d / total_alpha.clamp(min=constants.EPS)[:, None, :]
    dynamic_normed = dynamic_normed.clamp(min=constants.EPS)

    loss = -torch.mean(
        total_alpha
        * (
            static_normed * static_normed.log()
            + torch.sum(dynamic_normed * dynamic_normed.log(), dim=1)
        )
    )

    ########### Dynamic vs Static Regularization from D2Nerf
    # # normed_sigma_s = sigma_s / total_sigma.clamp(min=eps)
    # # normed_sigma_s = F.sigmoid(normed_sigma_s).clamp(min=constants.EPS, max=1 - constants.EPS)
    # # loss = -torch.mean(
    # #     normed_sigma_s * torch.log(normed_sigma_s)
    # #     + (1 - normed_sigma_s) * torch.log1p(-normed_sigma_s)
    # # ) / (num_vehicles + 1)

    # normed_sigma_d = sigma_d / total_sigma.clamp(min=eps)[:, None, :]
    # normed_sigma_d = F.sigmoid(normed_sigma_d)
    # skewness = 1.0
    # normed_sigma_d = torch.clamp(normed_sigma_d**skewness, min=constants.EPS, max=1 - constants.EPS)
    # rev_normed_sigma_d = torch.clamp(1-normed_sigma_d, min=eps)

    # [N_rays, num_vehicles, N_samples]
    # """loss = (
    #     -torch.mean(
    #         normed_sigma_d * torch.log(normed_sigma_d)
    #         + (1 - normed_sigma_d) * torch.log1p(-normed_sigma_d),
    #         (0, 2),
    #     ).sum()
    #     / num_vehicles
    # )"""
    # loss = (
    #     -torch.mean(
    #         normed_sigma_d * torch.log(normed_sigma_d)
    #         + rev_normed_sigma_d * torch.log(rev_normed_sigma_d),
    #         (0, 2),
    #     ).sum()
    #     / num_vehicles
    # )

    return loss


def compute_ray_reg(sigma_d, total_sigma):
    # Ray regularization from D2Nerf

    num_vehicles = sigma_d.shape[1]

    normed_sigma_d = sigma_d / total_sigma.clamp(min=constants.EPS)[:, None, :]
    normed_sigma_d = F.sigmoid(normed_sigma_d)

    loss = (
        torch.mean(torch.max(normed_sigma_d, dim=-1)[0] ** 2.0, dim=0).sum()
        / num_vehicles
    )

    return loss


def compute_static_reg(sigma_s, alpha_s):
    alpha_static_clamp = alpha_s.clamp(min=constants.EPS, max=1 - constants.EPS)

    mask_thresold = 0.1
    sigma_s_sum = torch.sum(sigma_s, dim=-1, keepdims=True)
    mask = torch.where(sigma_s_sum < mask_thresold, 0.0, 1.0)
    p = alpha_static_clamp / torch.sum(alpha_static_clamp, dim=-1, keepdims=True)
    loss = torch.mean(mask * -torch.mean(p * torch.log(p), dim=-1, keepdims=True))

    return loss


# Hierarchical sampling (section 5.2)
'''
def sample_pdf(bins, weights, N_samples, det=False):
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    """
    weights_sum = torch.sum(weights, -1, keepdim=True)
    weights_sum = torch.where(
        weights_sum >= 0, weights_sum, torch.finfo(torch.float32).eps
    )
    pdf = weights / weights_sum
    """

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
'''


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
