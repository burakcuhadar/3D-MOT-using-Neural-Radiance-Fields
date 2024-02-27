from collections import defaultdict

import torch
import torch.nn as nn

from lietorch import SO3, SE3
import pypose as pp

# from pytorch3d.transforms import se3_exp_map

from models.nerf import NeRF
from models.rendering__ import raw2outputs, raw2outputs_star


# For type checking of pytorch tensors at runtime
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from .types__ import StarNetworkOutput, NerfNetworkOutput
from typing import Union, Optional

patch_typeguard()


class STaR(nn.Module):
    def __init__(self, args):
        super(STaR, self).__init__()

        self.num_vehicles = args.num_vehicles

        self.chunk = args.chunk
        self.far_dist = args.far_dist

        self.N_importance = args.N_importance

        self.static_coarse_nerf = NeRF(D=args.netdepth, W=args.netwidth, args=args)
        if args.N_importance > 0:
            self.static_fine_nerf = NeRF(
                D=args.netdepth_fine, W=args.netwidth_fine, args=args
            )

        self.dynamic_coarse_nerfs = nn.ModuleList(
            [
                NeRF(D=args.netdepth // 2, W=args.netwidth, args=args)
                for _ in range(self.num_vehicles)
            ]
        )
        if args.N_importance > 0:
            self.dynamic_fine_nerfs = nn.ModuleList(
                [
                    NeRF(
                        D=args.netdepth_fine // 2,
                        W=args.netwidth_fine,
                        args=args,
                    )
                    for _ in range(self.num_vehicles)
                ]
            )

    def get_nerf_params(self):
        return (
            list(self.static_coarse_nerf.parameters())
            + list(self.static_fine_nerf.parameters())
            + list(self.dynamic_coarse_nerfs.parameters())
            + list(self.dynamic_fine_nerfs.parameters())
        )

    @typechecked
    def forward(
        self,
        pts: TensorType["num_rays", "num_samples", 3],
        viewdirs: TensorType["num_rays", 3],
        z_vals: TensorType["num_rays", "num_samples"],
        rays_d: TensorType["num_rays", 3],
        pose: Optional[
            Union[TensorType["num_vehicles", 4, 4], TensorType["num_vehicles", 7]]
        ] = None,
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
            if result[k][0] is None:
                result[k] = None
            elif len(result[k][0].shape) == 0:
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
        pose: Optional[
            Union[TensorType["num_vehicles", 4, 4], TensorType["num_vehicles", 7]]
        ] = None,
        is_coarse=True,
        object_pose=None,
        step=None,
    ) -> Union[NerfNetworkOutput, StarNetworkOutput]:
        N_rays = pts.shape[0]
        N_samples = pts.shape[1]

        if is_coarse:
            static_model = self.static_coarse_nerf
            dynamic_models = self.dynamic_coarse_nerfs
        else:
            if self.N_importance <= 0:
                raise ValueError("N_importance should be positive")
            static_model = self.static_fine_nerf
            dynamic_models = self.dynamic_fine_nerfs

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

        if object_pose is not None or len(pose.shape) not in [3, 2]:
            raise NotImplementedError
        elif len(pose.shape) == 3:
            pose_matrix = pose

            pts_homog = torch.cat(
                [pts, torch.ones((N_rays, N_samples, 1), device=pts.device)], dim=-1
            )  # [N_rays, N_samples, 4]
            pts_homog_flat = pts_homog.reshape((-1, 4))  # [N_rays*N_samples, 4]

            pts_dynamic_homog_flat = torch.einsum(
                "vij,nj->vni", pose_matrix, pts_homog_flat
            )
            pts_dynamic_homog = pts_dynamic_homog_flat.reshape(
                (self.num_vehicles, N_rays, N_samples, 4)
            )
            pts_dynamic = pts_dynamic_homog[
                ..., :3
            ]  # [num_vehicles, N_rays, N_samples, 3]

            viewdirs_dynamic = torch.einsum(
                "vij,nj->vni", pose_matrix[:, :3, :3], viewdirs
            )

        else:
            # pose_matrix = torch.eye(4, device=pts.device, dtype=torch.float32).repeat([self.num_vehicles, 1, 1])
            # rot = pose[:, 3:]
            # pose_matrix[:, :3, :3] = SO3.exp(rot).matrix()[:, :3, :3]
            # pose_matrix[:, :3, 3] = pose[:, :3]
            pts_dynamic = []
            viewdirs_dynamic = []
            pts_flat = pts.reshape((-1, 3))

            for i in range(self.num_vehicles):
                pts_dynamic_flat = pp.SE3(pose[i]).Act(pts_flat)
                pts_dynamic.append(
                    pts_dynamic_flat.reshape((N_rays, N_samples, 3)).unsqueeze(0)
                )
                viewdirs_dynamic.append(pp.SO3(pose[i, 3:]).Act(viewdirs).unsqueeze(0))

            pts_dynamic = torch.cat(pts_dynamic, dim=0)
            viewdirs_dynamic = torch.cat(viewdirs_dynamic, dim=0)

        raw_alpha_dynamic = torch.zeros(
            (N_rays, self.num_vehicles, N_samples), device=pts.device
        )
        raw_rgb_dynamic = torch.zeros(
            (N_rays, self.num_vehicles, N_samples, 3), device=pts.device
        )
        for i, dynamic_model in enumerate(dynamic_models):
            raw_alpha_dynamic[:, i], raw_rgb_dynamic[:, i] = dynamic_model(
                pts_dynamic[i], viewdirs_dynamic[i], step=step
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
            test=(not self.training),
        )
