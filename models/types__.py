from typing import TypedDict, Optional, Union
from torchtyping import TensorType


class NerfNetworkOutput(TypedDict):
    rgb: TensorType["num_rays", 3]
    disp: TensorType["num_rays"]
    acc: TensorType["num_rays"]
    weights: TensorType["num_rays", "num_samples"]
    depth: TensorType["num_rays"]
    dists: TensorType["num_rays", "num_samples"]
    z_vals: TensorType["num_rays", "num_samples"]


class StarNetworkOutput(TypedDict):
    # num_samples_ is different from num_samples because num_samples_ = N_samples, num_samples = N_samples + N_importance
    rgb: TensorType["num_rays", 3]
    disp: TensorType["num_rays"]
    acc: TensorType["num_rays"]
    dynamic_transmittance: TensorType["num_rays", "num_vehicles"]
    weights: TensorType["num_rays", "num_samples_"]
    depth: TensorType["num_rays"]
    rgb_static: TensorType["num_rays", 3]
    depth_static: TensorType["num_rays"]
    rgb_dynamic: TensorType["num_rays", "num_vehicles", 3]
    rgb_dynamic_all: Optional[TensorType["num_rays", 3]] = None
    depth_dynamic: TensorType["num_rays", "num_vehicles"]
    # entropy: TensorType[()]
    loss_alpha_entropy: TensorType[()]
    loss_dynamic_vs_static_reg: TensorType[()]
    loss_ray_reg: TensorType[()]
    loss_static_reg: TensorType[()]
    loss_dynamic_reg: TensorType[()]


class StarMipOnlineOutput(TypedDict):
    # num_samples_ is different from num_samples because num_samples_ = N_samples, num_samples = N_samples + N_importance
    rgb: TensorType["num_rays", 3]
    acc: TensorType["num_rays", 1]
    weights: TensorType["num_rays", "num_samples", 1]
    depth: TensorType["num_rays", 1]

    rgb_static: TensorType["num_rays", 3]
    depth_static: TensorType["num_rays", 1]
    rgb_dynamic: TensorType["num_rays", "num_vehicles", 3]
    depth_dynamic: TensorType["num_rays", "num_vehicles"]
    dynamic_transmittance: TensorType["num_rays", "num_vehicles", 1]

    loss_alpha_entropy: TensorType[()]
    loss_dynamic_vs_static_reg: TensorType[()]
    loss_ray_reg: TensorType[()]
    loss_static_reg: TensorType[()]
    loss_dynamic_reg: TensorType[()]


class StarMipOnlineCombinedOutput(StarMipOnlineOutput):
    rgb0: TensorType["num_rays", 3]
    acc0: TensorType["num_rays", 1]
    weights0: TensorType["num_rays", "num_samples_", 1]
    depth0: TensorType["num_rays", 1]

    rgb_static0: TensorType["num_rays", 3]
    depth_static0: TensorType["num_rays", 1]
    rgb_dynamic0: TensorType["num_rays", "num_vehicles", 3]
    depth_dynamic0: TensorType["num_rays", "num_vehicles"]
    dynamic_transmittance0: TensorType["num_rays", "num_vehicles", 1]

    loss_alpha_entropy0: TensorType[()]
    loss_dynamic_vs_static_reg0: TensorType[()]
    loss_ray_reg0: TensorType[()]
    loss_static_reg0: TensorType[()]
    loss_dynamic_reg0: TensorType[()]


class StarMipAppInitOutput(TypedDict):
    # num_samples_ is different from num_samples because num_samples_ = N_samples, num_samples = N_samples + N_importance
    rgb: TensorType["num_rays", 3]
    acc: TensorType["num_rays", 1]
    weights: TensorType["num_rays", "num_samples", 1]
    depth: TensorType["num_rays", 1]


class StarMipAppInitCombinedOutput(StarMipAppInitOutput):
    rgb0: TensorType["num_rays", 3]
    acc0: TensorType["num_rays", 1]
    weights0: TensorType["num_rays", "num_samples_", 1]
    depth0: TensorType["num_rays", 1]

class StarCoarseNetworkOutput(TypedDict):
    rgb0: TensorType["num_rays", 3]
    disp0: TensorType["num_rays"]
    acc0: TensorType["num_rays"]
    dynamic_transmittance0: TensorType["num_rays", "num_vehicles"]
    weights0: TensorType["num_rays", "num_samples"]
    depth0: TensorType["num_rays"]
    rgb_static0: TensorType["num_rays", 3]
    depth_static0: TensorType["num_rays"]
    rgb_dynamic0: TensorType["num_rays", "num_vehicles", 3]
    rgb_dynamic_all0: Optional[TensorType["num_rays", 3]] = None
    depth_dynamic0: TensorType["num_rays", "num_vehicles"]
    # z_std: TensorType["num_rays"]
    loss_alpha_entropy0: TensorType[()]
    loss_dynamic_vs_static_reg0: TensorType[()]
    loss_ray_reg0: TensorType[()]
    loss_static_reg0: TensorType[()]
    loss_dynamic_reg0: TensorType[()]

class CoarseWithFineRenderOutput(StarNetworkOutput, StarCoarseNetworkOutput):
    z_std: TensorType["num_rays"]
    
StarRenderOutput = Union[CoarseWithFineRenderOutput, StarCoarseNetworkOutput]