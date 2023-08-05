import torch
import os
from glob import glob
import numpy as np
import imageio

from datasets.carla_star_app_init import StarAppInitDataset
from torch.utils.data import Dataset
from models.rendering import get_rays, get_rays_np
from utils.dataset import load_intrinsics, natural_keys, from_ue4_to_nerf


# TODO delete it
class StarAppInitSemanticDataset(StarAppInitDataset):
    def __init__(self, args, split):
        self.validate_split(split)
        self.split = split
        self.N_samples = args.N_samples
        self.N_rand = args.N_rand

        imgs, poses = self.load_imgs_poses(args)
        H, W, focal = load_intrinsics(args)

        self.H = int(H)
        self.W = int(W)
        self.focal = focal

        self.imgs = imgs
        self.poses = poses

        self.near = args.near
        self.far = args.far

        if args.scale_factor > 0:
            self.near *= args.scale_factor
            self.far *= args.scale_factor
            self.poses[:, :3, 3] *= args.scale_factor

        print("near", self.near)
        print("far", self.far)

        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

        self.K = K

        if self.split == "train":
            # [N,ro+rd,H,W,3]
            rays = np.stack(
                [get_rays_np(self.H, self.W, self.K, p) for p in self.poses[:, :3, :4]],
                0,
            )
            rays_o = rays[:, 0, :, :, :]  # [N,H,W,3]
            rays_d = rays[:, 1, :, :, :]  # [N,H,W,3]

            rays_o = np.reshape(rays_o, [-1, 3])  # [N*H*W,3]
            rays_d = np.reshape(rays_d, [-1, 3])  # [N*H*W,3]
            target_rgbs = np.reshape(self.imgs, [-1, self.imgs.shape[-1]])  # [N*H*W,3]

            rays_o = rays_o.astype(np.float32)
            rays_d = rays_d.astype(np.float32)

            car_mask = self.semantic_imgs.reshape([target_rgbs.shape[0]]) == 10
            noncar_mask = self.semantic_imgs.reshape([target_rgbs.shape[0]]) != 10

            self.rays_o_car = rays_o[car_mask]
            self.rays_d_car = rays_d[car_mask]
            self.target_rgbs_car = target_rgbs[car_mask]

            self.rays_o_noncar = rays_o[noncar_mask]
            self.rays_d_noncar = rays_d[noncar_mask]
            self.target_rgbs_noncar = target_rgbs[noncar_mask]

    def __getitem__(self, idx):
        if self.split == "train":
            car_indices = np.random.choice(len(self.rays_o_car), self.N_rand // 2)
            rays_o_car = self.rays_o_car[car_indices, ...]
            rays_d_car = self.rays_d_car[car_indices, ...]
            target_car = self.target_rgbs_car[car_indices, ...]

            noncar_indices = np.random.choice(len(self.rays_o_noncar), self.N_rand // 2)
            rays_o_noncar = self.rays_o_noncar[noncar_indices, ...]
            rays_d_noncar = self.rays_d_noncar[noncar_indices, ...]
            target_noncar = self.target_rgbs_noncar[noncar_indices, ...]

        elif self.split == "val":
            idx = np.random.randint(low=0, high=self.imgs.shape[0])
            # Load all of the rays of one image
            target = self.imgs[idx]
            target = torch.Tensor(target)
            pose = self.poses[idx, :3, :4]
            rays_o, rays_d = get_rays(
                self.H, self.W, self.K, torch.Tensor(pose)
            )  # (H, W, 3), (H, W, 3)
            rays_o = torch.reshape(rays_o, [-1, 3])  # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1, 3])  # (H*W, 3)
            target = torch.reshape(target, [-1, 3])  # (H*W, 3)

            # Pass the whole rays as we dont have semantic mask for val views
            # TODO modify after dataset is corrected
            rays_o_car = rays_o
            rays_d_car = rays_d
            target_car = target
            rays_o_noncar = rays_o
            rays_d_noncar = rays_d
            target_noncar = target

        return {
            "rays_o_car": rays_o_car,
            "rays_d_car": rays_d_car,
            "target_car": target_car,
            "rays_o_noncar": rays_o_noncar,
            "rays_d_noncar": rays_d_noncar,
            "target_noncar": target_noncar,
        }
