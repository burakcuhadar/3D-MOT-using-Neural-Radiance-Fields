import torch
import os
from glob import glob
import numpy as np
import imageio

from models.rendering import get_rays, get_rays_np
from utils.dataset import load_intrinsics, natural_keys, from_ue4_to_nerf
from .carla_star_app_init import StarAppInitDataset

class StarAppInitDatasetNerfacc(StarAppInitDataset):
    def __init__(self, args, split):
        super().__init__(args, split)

    def update_num_rays(self, num_rays):
        self.N_rand = num_rays