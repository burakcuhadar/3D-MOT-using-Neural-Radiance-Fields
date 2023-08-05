from typing import TypedDict
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
    weights: TensorType["num_rays", "num_samples_"]
    depth: TensorType["num_rays"]
    rgb_static: TensorType["num_rays", 3]
    rgb_dynamic: TensorType["num_rays", 3]
    depth_static: TensorType["num_rays"]
    depth_dynamic: TensorType["num_rays"]
    dists: TensorType["num_rays", "num_samples_"]
    entropy: TensorType[()]
    z_vals: TensorType["num_rays", "num_samples_"]


class StarRenderOutput(StarNetworkOutput):
    rgb0: TensorType["num_rays", 3]
    disp0: TensorType["num_rays"]
    acc0: TensorType["num_rays"]
    weights0: TensorType["num_rays", "num_samples"]
    depth0: TensorType["num_rays"]
    rgb_static0: TensorType["num_rays", 3]
    rgb_dynamic0: TensorType["num_rays", 3]
    depth_static0: TensorType["num_rays"]
    depth_dynamic0: TensorType["num_rays"]
    dists0: TensorType["num_rays", "num_samples"]
    entropy0: TensorType[()]
    z_vals0: TensorType["num_rays", "num_samples"]
    z_std: TensorType["num_rays"]
