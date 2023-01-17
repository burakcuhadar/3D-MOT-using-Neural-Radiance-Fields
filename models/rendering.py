import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import imageio
from tqdm import tqdm
from tqdm.contrib import tenumerate
from utils.io import device

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy') 
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d



def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d



def render(model_coarse, model_fine, pts, viewdirs, z_vals, rays_o, rays_d, retraw=True, N_importance=0): 
    
    rgb_map, disp_map, acc_map, weights, depth_map = model_coarse(pts, viewdirs, z_vals, rays_d)

    # Hierarchical volume sampling
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(not model_coarse.training))
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        rgb_map, disp_map, acc_map, weights, depth_map = model_fine(pts, viewdirs, z_vals, rays_d)
        
    extras = {}
    """TODO: needed? if retraw:
        extras['raw'] = raw"""
    if N_importance > 0:
        extras['rgb0'] = rgb_map_0
        extras['disp0'] = disp_map_0
        extras['acc0'] = acc_map_0
        extras['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    return rgb_map, disp_map, acc_map, extras
    



def render_star(star_model, pts, viewdirs, z_vals, rays_o, rays_d, frames=None, retraw=True, N_importance=0, 
    appearance_init=False, object_pose=None, step=None): 
    
    if appearance_init:
        rgb_map, disp_map, acc_map, weights, depth_map = star_model(pts, viewdirs, z_vals, rays_d, frames, 
            is_coarse=True)
    else:
        if frames is not None:
            rgb_map, disp_map, acc_map, weights, depth_map, entropy, rgb_map_static, rgb_map_dynamic, transformed_pts = \
                star_model(pts, viewdirs, z_vals, rays_d, frames, is_coarse=True, object_pose=object_pose, step=step)
        else:
            rgb_map, disp_map, acc_map, weights, depth_map, entropy, rgb_map_static, rgb_map_dynamic = star_model(pts, 
                viewdirs, z_vals, rays_d, frames, is_coarse=True, object_pose=object_pose)

    # Hierarchical volume sampling
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        if not appearance_init:
            rgb_map_static0, rgb_map_dynamic0, entropy0 = rgb_map_static, rgb_map_dynamic, entropy
            if frames is not None:
                transformed_pts0 = transformed_pts

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(not star_model.training))
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        if appearance_init:
            rgb_map, disp_map, acc_map, weights, depth_map = star_model(pts, viewdirs, z_vals, rays_d, frames, 
                is_coarse=False)
        else:
            if frames is not None:
                rgb_map, disp_map, acc_map, weights, depth_map, entropy, rgb_map_static, rgb_map_dynamic, transformed_pts = \
                    star_model(pts, viewdirs, z_vals, rays_d, frames, is_coarse=False, object_pose=object_pose, step=step)
            else:
                rgb_map, disp_map, acc_map, weights, depth_map, entropy, rgb_map_static, rgb_map_dynamic = star_model(pts, 
                    viewdirs, z_vals, rays_d, frames, is_coarse=False, object_pose=object_pose)
            
    extras = {}
    if not appearance_init:
        extras['rgb_map_dynamic'] = rgb_map_dynamic
        extras['rgb_map_static'] = rgb_map_static
        if frames is not None:
            extras['transformed_pts'] = transformed_pts
    if N_importance > 0:
        extras['rgb0'] = rgb_map_0
        extras['disp0'] = disp_map_0
        extras['acc0'] = acc_map_0
        extras['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        if not appearance_init:
            extras['rgb_map_static0'] = rgb_map_static0
            extras['rgb_map_dynamic0'] = rgb_map_dynamic0
            extras['entropy0'] = entropy0
            if frames is not None:
                extras['transformed_pts0'] = transformed_pts0

    if appearance_init:
        return rgb_map, disp_map, acc_map, extras
    else:
        return rgb_map, disp_map, acc_map, extras, entropy
    



def raw2outputs(raw_alpha, raw_rgb, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, ret_entropy=False):
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
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[...,:1].shape)], -1) #[N_rays,N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw_rgb)  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_alpha.shape) * raw_noise_std

    alpha = raw2alpha(raw_alpha + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    if ret_entropy: # TODO I think in the paper they dont use regularization for appearance init. But it may still be helpful, do experiments.
        entropy = compute_entropy(alpha)
        return rgb_map, disp_map, acc_map, weights, depth_map, entropy 

    return rgb_map, disp_map, acc_map, weights, depth_map


def raw2outputs_star(raw_alpha_static, raw_rgb_static, raw_alpha_dynamic, raw_rgb_dynamic, z_vals, rays_d, 
    raw_noise_std=0, white_bkgd=False, ret_entropy=False):
    
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[...,:1].shape)], -1) #[N_rays,N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb_static = torch.sigmoid(raw_rgb_static)  # [N_rays, N_samples, 3]
    rgb_dynamic = torch.sigmoid(raw_rgb_dynamic)  # [N_rays, N_samples, 3]
    
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_alpha_static.shape) * raw_noise_std

    alpha_static = raw2alpha(raw_alpha_static + noise, dists)  # [N_rays, N_samples]
    alpha_dynamic = raw2alpha(raw_alpha_dynamic + noise, dists)  # [N_rays, N_samples]
    alpha_total = raw2alpha(raw_alpha_static + noise + raw_alpha_dynamic + noise, dists)

    T_s = torch.cumprod(torch.cat([torch.ones((alpha_static.shape[0], 1), device=device), 1.-alpha_static + 1e-10], -1), -1)[:, :-1]
    T_d = torch.cumprod(torch.cat([torch.ones((alpha_dynamic.shape[0], 1), device=device), 1.-alpha_dynamic + 1e-10], -1), -1)[:, :-1]
    #T = T_s * T_d 
    T = torch.cumprod(torch.cat([torch.ones((alpha_total.shape[0], 1), device=device), 1.-alpha_total + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(T[...,None] * (alpha_static[...,None] * rgb_static + alpha_dynamic[...,None] * rgb_dynamic), dim=-2)
    
    
    # Only for visualization
    rgb_map_static = torch.sum(T_s[...,None] * alpha_static[...,None] * rgb_static, dim=-2)
    rgb_map_dynamic = torch.sum(T_d[...,None] * alpha_dynamic[...,None] * rgb_dynamic, dim=-2)

    #weights = T * (alpha_static + alpha_dynamic)
    #weights = T * alpha_total # [N_rays, N_samples]
    weights = T_s * T_d * alpha_total

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    if ret_entropy:
        entropy = compute_entropy(alpha_static, alpha_dynamic)
        return rgb_map, disp_map, acc_map, weights, depth_map, entropy, rgb_map_static, rgb_map_dynamic

    return rgb_map, disp_map, acc_map, weights, depth_map, rgb_map_static, rgb_map_dynamic


def compute_entropy(alpha_static, alpha_dynamic=None):
    
    eps = torch.finfo(alpha_static.dtype).eps
    alpha_static_clamp = alpha_static.clamp(min=eps, max=1-eps)
    alpha_dynamic_clamp = alpha_dynamic.clamp(min=eps, max=1-eps)
    
    entropy = -torch.mean(alpha_static * torch.log(alpha_static_clamp) + (1 - alpha_static) * torch.log1p(-alpha_static_clamp))
    entropy += -torch.mean(alpha_dynamic * torch.log(alpha_dynamic_clamp) + (1 - alpha_dynamic) * torch.log1p(-alpha_dynamic_clamp))
    

    total_alpha = alpha_static + alpha_dynamic
    static_normed_trans = alpha_static / total_alpha.clamp(min=eps)
    static_normed_trans_clamp = static_normed_trans.clamp(min=eps)
    dynamic_normed_trans = alpha_dynamic / total_alpha.clamp(min=eps)
    dynamic_normed_trans_clamp = dynamic_normed_trans.clamp(min=eps)

    #TODO entropy += ...
    entropy += -torch.mean(total_alpha * (static_normed_trans * static_normed_trans_clamp.log() + \
        dynamic_normed_trans * dynamic_normed_trans_clamp.log()))
    
    return entropy
    

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples



def render_path(render_dataloader, model_coarse, model_fine, N_importance, device, render_factor=0, savedir=None):
    rgbs = []
    disps = []
    
    H = render_dataloader.dataset.H
    W = render_dataloader.dataset.W
    
    # Keep track of loss and psnr, these are computed when gt images are provided 
    test_loss = 0
    test_psnr = 0
    
    with torch.no_grad():
        #for i, batch in enumerate(render_dataloader):
        for i, batch in tenumerate(render_dataloader):
        
            render_dataloader.dataset.move_batch_to_device(batch, device)
            pts, viewdirs, z_vals, rays_o, rays_d, target = batch
            
            rgb, disp, acc, extras = render(model_coarse, model_fine, pts, viewdirs, z_vals, rays_o, rays_d, 
                retraw=True, N_importance=N_importance)
            
            if target is not None and render_factor==0:
                loss = img2mse(rgb,target)
                psnr = mse2psnr(loss)
                test_loss += loss.item()
                test_psnr += psnr.item()

            rgbs.append(to8b(rgb.cpu().detach().numpy().reshape((H, W, 3))))
            disps.append(disp.cpu().detach().numpy().reshape((H, W, 1)))

            if savedir is not None:
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgbs[-1])

    if target is not None and render_factor==0:
        test_loss /= len(render_dataloader)
        test_psnr /= len(render_dataloader)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    disps = to8b(disps)

    return rgbs, disps, test_loss, test_psnr

# step is required for barf encoding
def render_path_star(render_dataloader, star_model, N_importance, device, savedir=None, step=None):
    rgbs = []
    disps = []
    rgb_statics0 = []
    rgb_dynamics0 = []
    rgb_statics = []
    rgb_dynamics = []

    H = render_dataloader.dataset.H
    W = render_dataloader.dataset.W
    
    with torch.no_grad():
        for i, batch in tenumerate(render_dataloader):
            render_dataloader.dataset.move_batch_to_device(batch, device)
            pts, viewdirs, z_vals, rays_o, rays_d, object_pose = batch

            rgb, disp, acc, extras, _ = render_star(star_model, pts, viewdirs, z_vals, rays_o, rays_d, 
                retraw=True, N_importance=N_importance, object_pose=object_pose, step=step)
            
            rgbs.append(to8b(rgb.cpu().detach().numpy().reshape((H, W, 3))))
            disps.append(disp.cpu().detach().numpy().reshape((H, W, 1)))
            rgb_statics0.append(to8b(extras['rgb_map_static0'].cpu().detach().numpy().reshape((H, W, 3))))
            rgb_dynamics0.append(to8b(extras['rgb_map_dynamic0'].cpu().detach().numpy().reshape((H, W, 3))))
            rgb_statics.append(to8b(extras['rgb_map_static'].cpu().detach().numpy().reshape((H, W, 3))))
            rgb_dynamics.append(to8b(extras['rgb_map_dynamic'].cpu().detach().numpy().reshape((H, W, 3))))

            if savedir is not None:
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgbs[-1])
                filename = os.path.join(savedir, '{:03d}_static.png'.format(i))
                imageio.imwrite(filename, rgb_statics[-1])
                filename = os.path.join(savedir, '{:03d}_dynamic.png'.format(i))
                imageio.imwrite(filename, rgb_dynamics[-1])
    
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    disps = to8b(disps)
    rgb_statics0 = np.stack(rgb_statics0, 0)
    rgb_statics = np.stack(rgb_statics, 0)
    rgb_dynamics0 = np.stack(rgb_dynamics0, 0)
    rgb_dynamics = np.stack(rgb_dynamics, 0)

    return rgbs, disps, rgb_statics0, rgb_dynamics0, rgb_statics, rgb_dynamics
    

