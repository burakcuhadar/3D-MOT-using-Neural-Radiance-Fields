# Modified from Nerf studio
"""
Implementation of mip-NeRF.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import (
    MSELoss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.base_model import Model
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.field_components.field_heads import RGBFieldHead


class MipNerfModel(Model):
    """mip-NeRF model

    Args:
        config: MipNerf configuration to instantiate model
    """

    config: VanillaModelConfig

    def __init__(
        self,
        config: VanillaModelConfig,
        **kwargs,
    ) -> None:
        self.field = None
        # assert (
        #     config.collider_params is not None
        # ), "MipNeRF model requires bounding box collider parameters."
        # TODO none scene_box, -1 num_train working?
        super().__init__(config=config, scene_box=None, num_train_data=-1, **kwargs)
        # assert (
        #     self.config.collider_params is not None
        # ), "mip-NeRF requires collider parameters to be set."

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=24,  # TODO was 16
            min_freq_exp=0.0,
            max_freq_exp=24.0,  # TODO was 16
            include_input=True,
        )
        direction_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=4,
            min_freq_exp=0.0,
            max_freq_exp=4.0,
            include_input=True,
        )

        self.field = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            use_integrated_encoding=True,
            field_heads=(RGBFieldHead(),),
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, samples: RaySamples):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        field_outputs_coarse = self.field.forward(samples)
        # if self.config.use_gradient_scaling:
        #     field_outputs_coarse = scale_gradients_by_distance_squared(
        #         field_outputs_coarse, samples
        #     )

        densities = field_outputs_coarse[FieldHeadNames.DENSITY]
        rgb = field_outputs_coarse[FieldHeadNames.RGB]

        return densities, rgb
