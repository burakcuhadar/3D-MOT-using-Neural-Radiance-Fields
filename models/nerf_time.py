import torch
import torch.nn as nn

from models.nerf import NeRF
from models.rendering__ import raw2outputs

# For type checking of pytorch tensors at runtime
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from .types__ import StarNetworkOutput, NerfNetworkOutput
from typing import Union, Optional

patch_typeguard()


class NerfTime(nn.Module):
    def __init__(self, args):
        super(NerfTime, self).__init__()

        self.chunk = args.chunk
        self.far_dist = args.far_dist

        self.num_frames = args.num_frames

        self.N_importance = args.N_importance

        self.coarse_nerf = NeRF(
            D=args.netdepth, W=args.netwidth, args=args, has_time=True
        )
        self.fine_nerf = NeRF(
            D=args.netdepth_fine, W=args.netwidth_fine, args=args, has_time=True
        )

    def get_nerf_params(self):
        return list(self.coarse_nerf.parameters()) + list(self.fine_nerf.parameters())

    @typechecked
    def forward(
        self,
        pts: TensorType["num_rays", "num_samples", 3],
        viewdirs: TensorType["num_rays", 3],
        z_vals: TensorType["num_rays", "num_samples"],
        rays_d: TensorType["num_rays", 3],
        frame,
        is_coarse=True,
    ) -> NerfNetworkOutput:
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
                frame,
                is_coarse,
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
        frame,
        is_coarse=True,
    ) -> NerfNetworkOutput:
        N_rays = pts.shape[0]
        N_samples = pts.shape[1]

        nerf_model = self.coarse_nerf if is_coarse else self.fine_nerf

        # Normalize time
        time = frame / (self.num_frames - 1)

        raw_alpha, raw_rgb = nerf_model(pts, viewdirs, step=None, time=time)

        return raw2outputs(
            raw_alpha,
            raw_rgb,
            z_vals,
            rays_d,
            nerf_model.raw_noise_std if self.training else 0,
            nerf_model.white_bkgd,
            far_dist=self.far_dist,
        )
