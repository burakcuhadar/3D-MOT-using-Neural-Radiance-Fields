from collections import defaultdict

import torch
import torch.nn as nn

from lietorch import SO3, SE3

# from pytorch3d.transforms import se3_exp_map

from models.nerf import VanillaNeRFRadianceField
from models.rendering import raw2outputs, raw2outputs_star


# For type checking of pytorch tensors at runtime
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from .types import StarNetworkOutput, NerfNetworkOutput
from typing import Union, Optional

patch_typeguard()


class STaR(nn.Module):
    def __init__(self, args):
        super(STaR, self).__init__()

        self.chunk = args.chunk
        self.far_dist = args.far_dist

        self.N_importance = args.N_importance

        """self.static_coarse_nerf = NeRF(D=args.netdepth, W=args.netwidth, args=args)
        self.static_fine_nerf = NeRF(
            D=args.netdepth_fine, W=args.netwidth_fine, args=args
        )

        self.dynamic_coarse_nerf = NeRF(D=args.netdepth, W=args.netwidth, args=args)
        self.dynamic_fine_nerf = NeRF(
            D=args.netdepth_fine, W=args.netwidth_fine, args=args
        )"""
        
        self.static_nerf = VanillaNeRFRadianceField(net_depth=args.netdepth, net_width=args.netwidth)
        self.dynamic_nerf = VanillaNeRFRadianceField(net_depth=args.netdepth, net_width=args.netwidth)
    
    @typechecked
    def forward(
        self,
        pts: TensorType["num_rays", "num_samples", 3],
        viewdirs: TensorType["num_rays", 3],
        z_vals: TensorType["num_rays", "num_samples"],
        rays_d: TensorType["num_rays", 3],
        pose: Optional[Union[TensorType[4, 4], TensorType[6]]] = None,
        is_coarse=True,
        object_pose=None,
        step=None,
    ) -> Union[NerfNetworkOutput, StarNetworkOutput]:
        # result = defaultdict(list)
        result = {}

        for i in range(0, pts.shape[0], self.chunk):
            end_i = min(pts.shape[0], i + self.chunk)
            pts_chunk = pts[i:end_i, ...]
            viewdirs_chunk = viewdirs[i:end_i, ...]
            z_vals_chunk = z_vals[i:end_i, ...]
            rays_d_chunk = rays_d[i:end_i, ...]

            chunk_result = self.forward_chunk(
                pts_chunk,
                viewdirs_chunk,
                z_vals_chunk,
                rays_d_chunk,
                pose,
                is_coarse,
                object_pose=object_pose,
                step=step,
            )

            for k, v in chunk_result.items():
                if i == 0:
                    result[k] = [v]
                else:
                    result[k] += [v]

        for k, v in result.items():
            if len(result[k][0].shape) == 0:
                result[k] = sum(v)
            else:
                result[k] = torch.cat(v, 0)

        return result


    @typechecked
    def forward_chunk(
        self,
        pts: TensorType["num_rays", "num_samples", 3],
        viewdirs: TensorType["num_rays", 3],
        z_vals: TensorType["num_rays", "num_samples"],
        rays_d: TensorType["num_rays", 3],
        pose: Optional[Union[TensorType[4, 4], TensorType[6]]] = None,
        is_coarse=True,
        object_pose=None,
        step=None,
    ) -> Union[NerfNetworkOutput, StarNetworkOutput]:
        N_rays = pts.shape[0]
        N_samples = pts.shape[1]

        if is_coarse:
            static_model = self.static_coarse_nerf
            dynamic_model = self.dynamic_coarse_nerf
        else:
            static_model = self.static_fine_nerf
            dynamic_model = self.dynamic_fine_nerf

        raw_alpha_static, raw_rgb_static = static_model(pts, viewdirs, step=None)

        # During appearance initialization only static part is trained
        if pose is None and object_pose is None:
            return raw2outputs(
                raw_alpha_static,
                raw_rgb_static,
                z_vals,
                rays_d,
                static_model.raw_noise_std if self.training else 0,
                static_model.white_bkgd,
                far_dist=self.far_dist,
            )

        if object_pose is not None or len(pose.shape) not in [2, 1]:
            raise NotImplementedError
        elif len(pose.shape) == 2:
            pose_matrix = pose
        else:
            rot = pose[3:]
            pose_matrix = torch.eye(4, device=pts.device, dtype=torch.float32)
            pose_matrix[:3, :3] = SO3.exp(rot).matrix()[:3, :3]
            pose_matrix[:3, 3] = pose[:3]

        pts_homog = torch.cat(
            [pts, torch.ones((N_rays, N_samples, 1), device=pts.device)], dim=-1
        )  # [N_rays, N_samples, 4]
        pts_homog_flat = pts_homog.reshape((-1, 4))  # [N_rays*N_samples, 4]

        pts_dynamic_homog_flat = torch.einsum("ij,nj->ni", pose_matrix, pts_homog_flat)
        pts_dynamic_homog = pts_dynamic_homog_flat.reshape((N_rays, N_samples, 4))
        pts_dynamic = pts_dynamic_homog[..., :3]  # [N_rays, N_samples, 3]

        viewdirs_dynamic = torch.einsum("ij,nj->ni", pose_matrix[:3, :3], viewdirs)

        raw_alpha_dynamic, raw_rgb_dynamic = dynamic_model(
            pts_dynamic, viewdirs_dynamic, step=step
        )

        return raw2outputs_star(
            raw_alpha_static,
            raw_rgb_static,
            raw_alpha_dynamic,
            raw_rgb_dynamic,
            z_vals,
            rays_d,
            # From the paper: "we add small Gaussian noise to the density outputs during
            # appearance initialization but turn it off during online training."
            0,
            static_model.white_bkgd,
            far_dist=self.far_dist,
        )
