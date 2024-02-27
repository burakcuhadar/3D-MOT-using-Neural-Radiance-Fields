from collections import defaultdict
from random import uniform

import torch
import torch.nn as nn

from lietorch import SO3, SE3
import pypose as pp

# from pytorch3d.transforms import se3_exp_map

from models.nerf import NeRF
from models.rendering_starmip import (
    get_starmip_online_outputs,
    get_starmip_appinit_outputs,
)
from models.mipnerf import MipNerfModel
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.utils.colors import COLORS_DICT

# For type checking of pytorch tensors at runtime
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from .types__ import (
    StarMipAppInitCombinedOutput,
    StarMipOnlineCombinedOutput,
)
from typing import Union, Optional

patch_typeguard()


class STaR(nn.Module):
    def __init__(self, args):
        super(STaR, self).__init__()

        self.num_vehicles = args.num_vehicles

        self.chunk = args.chunk
        self.far_dist = args.far_dist

        self.N_importance = args.N_importance

        self.static_nerf_config = VanillaModelConfig(
            _target=MipNerfModel,
            num_coarse_samples=args.N_samples,
            num_importance_samples=args.N_importance,
            # collider_params=collider_params,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            eval_num_rays_per_chunk=1024,
        )
        self.static_nerf = MipNerfModel(self.static_nerf_config)

        self.dynamic_nerf_config = VanillaModelConfig(
            _target=MipNerfModel,
            num_coarse_samples=args.N_samples,
            num_importance_samples=args.N_importance,
            # collider_params=collider_params,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            eval_num_rays_per_chunk=1024,
        )
        self.dynamic_nerfs = nn.ModuleList(
            [MipNerfModel(self.dynamic_nerf_config) for _ in range(self.num_vehicles)]
        )

        # samplers
        self.sampler_uniform = UniformSampler(
            num_samples=self.static_nerf_config.num_coarse_samples
        )
        self.sampler_pdf = PDFSampler(
            num_samples=self.static_nerf_config.num_importance_samples,
            include_original=False,
        )

        self.collider = NearFarCollider(
            near_plane=args.scale_factor * args.near,
            far_plane=args.scale_factor * args.far,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=COLORS_DICT["black"])
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

    def get_nerf_params(self):
        return list(self.static_nerf.parameters()) + list(
            self.dynamic_nerfs.parameters()
        )

    @typechecked
    def forward(
        self,
        origins: TensorType["num_rays", 3],
        viewdirs: TensorType["num_rays", 3],
        pose: Optional[
            Union[TensorType["num_vehicles", 4, 4], TensorType["num_vehicles", 7]]
        ] = None,
    ) -> Union[StarMipAppInitCombinedOutput, StarMipOnlineCombinedOutput]:
        # result = defaultdict(list)
        result = {}

        for i in range(0, origins.shape[0], self.chunk):
            end_i = min(origins.shape[0], i + self.chunk)
            origins_chunk = origins[i:end_i, ...]
            viewdirs_chunk = viewdirs[i:end_i, ...]

            chunk_result = self.forward_chunk(
                origins_chunk,
                viewdirs_chunk,
                pose,
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
        origins: TensorType["num_rays", 3],
        viewdirs: TensorType["num_rays", 3],
        pose: Optional[
            Union[TensorType["num_vehicles", 4, 4], TensorType["num_vehicles", 7]]
        ] = None,
    ) -> Union[StarMipAppInitCombinedOutput, StarMipOnlineCombinedOutput]:
        if pose is not None:
            return self.__forward_online(origins, viewdirs, pose)
        else:
            return self.__forward_app_init(origins, viewdirs)

    """
    def __transform_origins_viewdirs(self, pose, origins, viewdirs):
        N_rays = origins.shape[0]

        if len(pose.shape) not in [2, 3]:
            raise NotImplementedError
        elif len(pose.shape) == 3:
            pose_matrix = pose

            origins_homog = torch.cat(
                [origins, torch.ones((N_rays, 1), device=origins.device)],
                dim=-1,
            )  # [N_rays, 4]

            origins_dynamic_homog = torch.einsum(
                "vij,nj->vni", pose_matrix, origins_homog
            )

            origins_dynamic = origins_dynamic_homog[
                ..., :3
            ]  # [num_vehicles, N_rays, 3]

            viewdirs_dynamic = torch.einsum(
                "vij,nj->vni", pose_matrix[:, :3, :3], viewdirs
            )

        else:
            origins_dynamic = []
            viewdirs_dynamic = []
            # origins_flat = origins.reshape((-1, 3))

            for i in range(self.num_vehicles):
                origins_dynamic.append(pp.SE3(pose[i]).Act(origins).unsqueeze(0))
                viewdirs_dynamic.append(pp.SO3(pose[i, 3:]).Act(viewdirs).unsqueeze(0))

            origins_dynamic = torch.cat(origins_dynamic, dim=0)
            viewdirs_dynamic = torch.cat(viewdirs_dynamic, dim=0)

        return origins_dynamic, viewdirs_dynamic
    """

    def __transform_frustums(self, samples: RaySamples, pose) -> list[RaySamples]:
        if len(pose.shape) not in [2, 3]:
            raise NotImplementedError
        elif len(pose.shape) == 3:
            raise NotImplementedError
        else:
            dynamic_ray_samples = []
            # origins_dynamic = []
            # viewdirs_dynamic = []
            # origins_flat = origins.reshape((-1, 3))

            for i in range(self.num_vehicles):
                origins_dynamic = (
                    pp.SE3(pose[i]).Act(samples.frustums.origins).unsqueeze(0)
                )
                directions_dynamic = (
                    pp.SO3(pose[i, 3:]).Act(samples.frustums.directions).unsqueeze(0)
                )

                dynamic_frustums = Frustums(
                    origins=origins_dynamic,
                    directions=directions_dynamic,
                    starts=samples.frustums.starts,
                    ends=samples.frustums.ends,
                    offsets=samples.frustums.offsets,
                    pixel_area=samples.frustums.pixel_area,
                )

                dynamic_ray_samples.append(
                    RaySamples(
                        frustums=dynamic_frustums,
                        camera_indices=samples.camera_indices,
                        deltas=samples.deltas,
                        spacing_starts=samples.spacing_starts,
                        spacing_ends=samples.spacing_ends,
                        spacing_to_euclidean_fn=samples.spacing_to_euclidean_fn,
                        metadata=samples.metadata,
                        times=samples.times,
                    )
                )

            # origins_dynamic = torch.cat(origins_dynamic, dim=0)
            # viewdirs_dynamic = torch.cat(viewdirs_dynamic, dim=0)
            return dynamic_ray_samples

    def __forward_dynamic(self, dynamic_samples: list[RaySamples]):
        N_rays = dynamic_samples[0].spacing_starts.shape[1]
        N_samples = dynamic_samples[0].spacing_starts.shape[2]
        device = dynamic_samples[0].spacing_starts.device

        density_dynamic = torch.zeros(
            (N_rays, self.num_vehicles, N_samples, 1), device=device
        )
        rgb_dynamic = torch.zeros(
            (N_rays, self.num_vehicles, N_samples, 3), device=device
        )

        for i, dynamic_model in enumerate(self.dynamic_nerfs):
            (
                density_dynamic[:, i, :, :],
                rgb_dynamic[:, i, :, :],
            ) = dynamic_model.get_outputs(dynamic_samples[i])

        return density_dynamic, rgb_dynamic

    def __combine_result_dicts(self, result_coarse, result_fine):
        result = {}
        result |= result_fine
        for k, v in result_coarse.items():
            result[f"{k}0"] = v

        return result

    def __forward_app_init(self, origins, viewdirs):
        ray_bundle = RayBundle(
            origins=origins,
            directions=viewdirs,
            pixel_area=torch.ones_like(origins[..., 0:1]),
        )
        ray_bundle = self.collider.set_nears_and_fars(ray_bundle)
        ray_bundle = self.collider(ray_bundle)

        uniform_samples = self.sampler_uniform(ray_bundle)

        density_static_coarse, rgb_static_coarse = self.static_nerf.get_outputs(
            uniform_samples
        )

        result_coarse = get_starmip_appinit_outputs(
            density_static_coarse,
            rgb_static_coarse,
            uniform_samples.deltas,
            uniform_samples,
            self.renderer_depth,
            self.renderer_accumulation,
        )

        pdf_samples = self.sampler_pdf(
            ray_bundle, uniform_samples, result_coarse["weights"]
        )

        density_static, rgb_static = self.static_nerf.get_outputs(pdf_samples)

        result_fine = get_starmip_appinit_outputs(
            density_static,
            rgb_static,
            pdf_samples.deltas,
            pdf_samples,
            self.renderer_depth,
            self.renderer_accumulation,
        )

        return self.__combine_result_dicts(result_coarse, result_fine)

    def __forward_online(self, origins, viewdirs, pose):
        ray_bundle = RayBundle(
            origins=origins,
            directions=viewdirs,
            pixel_area=torch.ones_like(origins[..., 0:1]),
        )
        ray_bundle = self.collider.set_nears_and_fars(ray_bundle)
        ray_bundle = self.collider(ray_bundle)

        uniform_samples = self.sampler_uniform(ray_bundle)

        density_static_coarse, rgb_static_coarse = self.static_nerf.get_outputs(
            uniform_samples
        )

        dynamic_uniform_samples = self.__transform_frustums(uniform_samples, pose)

        density_dynamic_coarse, rgb_dynamic_coarse = self.__forward_dynamic(
            dynamic_uniform_samples
        )

        result_coarse = get_starmip_online_outputs(
            density_static_coarse,
            rgb_static_coarse,
            density_dynamic_coarse,
            rgb_dynamic_coarse,
            uniform_samples.deltas,
            uniform_samples,
            self.renderer_depth,
            self.renderer_accumulation,
        )

        pdf_samples = self.sampler_pdf(
            ray_bundle, uniform_samples, result_coarse["weights"]
        )

        density_static, rgb_static = self.static_nerf.get_outputs(pdf_samples)

        dynamic_pdf_samples = self.__transform_frustums(pdf_samples, pose)

        density_dynamic, rgb_dynamic = self.__forward_dynamic(dynamic_pdf_samples)

        result_fine = get_starmip_online_outputs(
            density_static,
            rgb_static,
            density_dynamic,
            rgb_dynamic,
            pdf_samples.deltas,
            pdf_samples,
            self.renderer_depth,
            self.renderer_accumulation,
        )

        return self.__combine_result_dicts(result_coarse, result_fine)
