from typing import Optional, Union, Tuple

import torch

import torch.nn.functional as F
import numpy as np

from utils import constants

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

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
    # z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    # z_samples = sample_pdf(
    #     z_vals_mid,
    #     result_coarse["weights"][..., 1:-1],
    #     N_importance,
    #     det=(not star_network.training),
    # )
    # z_samples = z_samples.detach()
    # z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    z_samples = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

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
    # #zero_dist_count = dists[dists <= 1e-10].shape[0]

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
        # "zero_dist_count": zero_dist_count
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

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # [N_rays,N_samples]
    dists = torch.cat(
        [dists, torch.tensor([far_dist], device=device).expand(dists[..., :1].shape)],
        -1,
    )
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb_static = torch.sigmoid(raw_rgb_static)  # [N_rays, N_samples, 3]
    rgb_dynamic = torch.sigmoid(raw_rgb_dynamic)  # [N_rays, num_vehicles, N_samples, 3]

    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw_alpha_static.shape) * raw_noise_std

    alpha_static = raw2alpha(raw_alpha_static + noise, dists)  # [N_rays, N_samples]
    alpha_dynamic = raw2alpha(
        raw_alpha_dynamic + noise, dists[:, None, :]
    )  # [N_rays, num_vehicles, N_samples]
    alpha_total = raw2alpha(
        raw_alpha_static + noise + raw_alpha_dynamic.sum(dim=1) + noise, dists
    )

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
    # T = T_s * torch.prod(T_d, dim=1) # which one?

    rgb_map = torch.sum(
        T[..., None]
        * (
            alpha_static[..., None] * rgb_static
            + torch.sum(alpha_dynamic[..., None] * rgb_dynamic, dim=1)
        ),
        dim=-2,
    )

    # raw_alpha_sum = raw_alpha_static + raw_alpha_dynamic.sum(dim=1)
    # raw_alpha_sum = torch.where(raw_alpha_sum > 0, raw_alpha_sum, 1e-10)

    # weighted_rgb = torch.sigmoid(
    #     (
    #         raw_alpha_static[..., None] * raw_rgb_static
    #         + torch.sum(raw_alpha_dynamic[..., None] * raw_rgb_dynamic, dim=1)
    #     )
    #     / raw_alpha_sum[..., None]
    # )

    # rgb_map = torch.sum(
    #     T[..., None] * alpha_total[..., None] * weighted_rgb,
    #     dim=-2,
    # )

    # Only for visualization
    rgb_map_static = torch.sum(
        T_s[..., None] * alpha_static[..., None] * rgb_static, dim=-2
    )

    rgb_map_dynamic = torch.sum(
        T_d[..., None] * alpha_dynamic[..., None] * rgb_dynamic, dim=-2
    )  # N_rays, num_vehicles, 3
    dynamic_weights = T_d * alpha_dynamic  # N_rays, num_vehicles, N_samples
    depth_dynamic = torch.sum(
        dynamic_weights * z_vals[:, None, :], -1
    )  # N_rays, num_vehicles
    # rgb_map_dynamic = torch.where(
    #     depth_dynamic[..., None] < 0.69,  # todo: get constant from args.
    #     rgb_map_dynamic,
    #     torch.zeros_like(rgb_map_dynamic),
    # )

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
    # acc_map_dynamic = torch.sum(dynamic_weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    sigma_s = F.softplus(raw_alpha_static)
    sigma_d = F.softplus(raw_alpha_dynamic)
    sigma_sum = sigma_s + sigma_d.sum(dim=1)

    # entropy = compute_entropy(alpha_static, alpha_dynamic)
    loss_alpha_entropy = compute_alpha_entropy(alpha_static, alpha_dynamic)
    loss_dynamic_vs_static_reg = compute_dynamic_vs_static_reg(
        sigma_s, sigma_d, sigma_sum, alpha_static, alpha_dynamic
    )

    loss_ray_reg = compute_ray_reg(sigma_d, sigma_sum)
    loss_static_reg = compute_static_reg(T_s, alpha_static)
    loss_dynamic_reg = compute_dynamic_reg(T_d, alpha_dynamic)

    # print("max sigma sum", torch.sum(sigma_d, dim=-1).max())
    # print("min sigma sum", torch.sum(sigma_d, dim=-1).min())
    # print("max acc dyna sum", acc_map_dynamic.max())
    # print("min acc dyna sum", acc_map_dynamic.min())
    # print("max alpha sum", alpha_dynamic.sum(dim=-1).max())
    # print("min alpha sum", alpha_dynamic.sum(dim=-1).min())
    # print("min T_d", T_d.min())
    # print("max T_d", T_d.max())
    # print("min T_d[-1]", T_d[:, -1].min())
    # print("max T_d[-1]", T_d[:, -1].max())
    # print("min T_s", T_s.min())
    # print("max T_s", T_s.max())
    # print("min T_s[-1]", T_s[:, -1].min())
    # print("max T_s[-1]", T_s[:, -1].max())
    # print("min T", T.min())
    # print("max T", T.max())
    # print("min T[-1]", T[:, -1].min())
    # print("max T[-1]", T[:, -1].max())

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
        "dynamic_transmittance": T_d[:, :, -1],
        # "entropy": entropy,
        # Regularization terms
        "loss_alpha_entropy": loss_alpha_entropy,
        "loss_dynamic_vs_static_reg": loss_dynamic_vs_static_reg,
        "loss_ray_reg": loss_ray_reg,
        "loss_static_reg": loss_static_reg,
        "loss_dynamic_reg": loss_dynamic_reg,
    }


def compute_entropy(alpha_static, alpha_dynamic):
    eps = torch.finfo(alpha_static.dtype).eps
    alpha_static_clamp = alpha_static.clamp(min=eps, max=1 - eps)
    alpha_dynamic_clamp = alpha_dynamic.clamp(min=eps, max=1 - eps)

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
    static_normed_trans = alpha_static / total_alpha.clamp(min=eps)
    static_normed_trans_clamp = static_normed_trans.clamp(min=eps)
    dynamic_normed_trans = alpha_dynamic / total_alpha.clamp(min=eps)[:, None, :]
    dynamic_normed_trans_clamp = dynamic_normed_trans.clamp(min=eps)

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
    num_vehicles = alpha_d.shape[1]

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
    # normed_sigma_s = sigma_s / total_sigma.clamp(min=eps)
    # normed_sigma_s = F.sigmoid(normed_sigma_s).clamp(min=constants.EPS, max=1 - constants.EPS)
    # loss = -torch.mean(
    #     normed_sigma_s * torch.log(normed_sigma_s)
    #     + (1 - normed_sigma_s) * torch.log1p(-normed_sigma_s)
    # ) / (num_vehicles + 1)

    # normed_sigma_d = sigma_d / total_sigma.clamp(min=constants.EPS)[:, None, :]
    # normed_sigma_d = F.sigmoid(normed_sigma_d)
    # skewness = 1.75
    # normed_sigma_d = torch.clamp(
    #     normed_sigma_d**skewness, min=constants.EPS, max=1 - constants.EPS
    # )
    # rev_normed_sigma_d = torch.clamp(1 - normed_sigma_d, min=constants.EPS)

    # # [N_rays, num_vehicles, N_samples]
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


def compute_static_reg(T_s, alpha_s):
    alpha_static_clamp = alpha_s.clamp(min=constants.EPS, max=1 - constants.EPS)

    mask_thresold = 0.1
    # sigma_s_sum = torch.sum(sigma_s, dim=-1, keepdims=True)
    # print("sigma_s_sum max min", sigma_s_sum.max(), sigma_s_sum.min())
    # mask = torch.where(sigma_s_sum < mask_thresold, 0.0, 1.0)
    # print("mask nonzero", torch.count_nonzero(mask))
    # alpha_s_sum = torch.sum(alpha_s, dim=-1, keepdims=True)
    # print("alpha_s_sum max min", alpha_s_sum.max(), alpha_s_sum.min())
    # mask = torch.where(alpha_s_sum < mask_thresold, 0.0, 1.0)
    # print("mask nonzero", torch.count_nonzero(mask))

    # mask = torch.where(T_s[:, -1] < mask_thresold, 0.0, 1.0)[:, None]
    # p = alpha_static_clamp / torch.sum(alpha_static_clamp, dim=-1, keepdims=True)
    # loss = torch.mean(mask * -torch.mean(p * torch.log(p), dim=-1, keepdims=True))

    p = alpha_static_clamp / torch.sum(alpha_static_clamp, dim=-1, keepdims=True)
    loss = torch.mean(-torch.mean(p * torch.log(p), dim=-1, keepdims=True))

    return loss


"""emer-nerf way"""
# def compute_dynamic_reg(sigma_d):
#     return sigma_d.mean()


# "num_rays", "num_vehicles", "num_samples"
def compute_dynamic_reg(T_d, alpha_d):
    alpha_dynamic_clamp = alpha_d.clamp(min=constants.EPS, max=1 - constants.EPS)

    mask_thresold = 0.1
    # sigma_d_sum = torch.sum(sigma_d, dim=-1, keepdims=True)  # n_rays, n_vehicles, 1
    # print("sigma_d_sum max min", sigma_d_sum.max(), sigma_d_sum.min())
    # mask = torch.where(sigma_d_sum < mask_thresold, 0.0, 1.0)  # n_rays, n_vehicles, 1
    # print("mask nonzero", torch.count_nonzero(mask))
    mask = torch.where(T_d[:, :, -1] < mask_thresold, 0.0, 1.0)[:, :, None]

    p = alpha_dynamic_clamp / torch.sum(
        alpha_dynamic_clamp, dim=-1, keepdims=True
    )  # n_rays, n_vehicles, n_samples
    loss = torch.mean(mask * -torch.mean(p * torch.log(p), dim=-1, keepdims=True))

    return loss


# old not used anymore
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

"""
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
"""


# Modified from nerf studio
@typechecked
def sample_pdf(
    bins,
    weights: TensorType["num_rays", "num_samples"],
    num_samples,
    det=False,
    single_jitter=False,
):
    num_bins = num_samples + 1
    histogram_padding = 1e-5
    eps = 1e-5

    weights = weights + histogram_padding

    # Add small offset to rays with zero weight to prevent NaNs
    weights_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.relu(eps - weights_sum)
    weights = weights + padding / weights.shape[-1]
    weights_sum += padding

    pdf = weights / weights_sum
    cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if det:
        # Stratified samples between 0 and 1
        u = torch.linspace(
            0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device
        )
        u = u.expand(size=(*cdf.shape[:-1], num_bins))
        if single_jitter:
            rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
        else:
            rand = (
                torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device)
                / num_bins
            )
        u = u + rand
    else:
        # Uniform samples between 0 and 1
        u = torch.linspace(
            0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device
        )
        u = u + 1.0 / (2 * num_bins)
        u = u.expand(size=(*cdf.shape[:-1], num_bins))
    u = u.contiguous()

    # existing_bins = torch.cat(
    #     [
    #         ray_samples.spacing_starts[..., 0],
    #         ray_samples.spacing_ends[..., -1:, 0],
    #     ],
    #     dim=-1,
    # )

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, 0, bins.shape[-1] - 1)
    above = torch.clamp(inds, 0, bins.shape[-1] - 1)
    cdf_g0 = torch.gather(cdf, -1, below)
    bins_g0 = torch.gather(bins, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g1 = torch.gather(bins, -1, above)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)

    return samples
