import numpy as np
import torch

from datasets.carla_star_online import StarOnlineDataset
from models.rendering import get_rays


class StarOnlineSemanticDataset(StarOnlineDataset):
    def __getitem__(self, idx):
        if self.split == "train":
            frame = np.random.randint(low=0, high=self.num_frames)
            frames = np.array([frame])[:, None]  # 1,1

            car_sample_num = int(self.N_rand * self.car_sample_ratio)
            car_indices = np.random.choice(self.rays_o_car.shape[1], car_sample_num)
            rays_o_car = self.rays_o_car[frame, car_indices, ...]
            rays_d_car = self.rays_d_car[frame, car_indices, ...]
            target_car = self.target_rgbs_car[frame, car_indices, ...]

            noncar_indices = np.random.choice(
                self.rays_o_noncar.shape[1], self.N_rand - car_sample_num
            )
            rays_o_noncar = self.rays_o_noncar[frame, noncar_indices, ...]
            rays_d_noncar = self.rays_d_noncar[frame, noncar_indices, ...]
            target_noncar = self.target_rgbs_noncar[frame, noncar_indices, ...]

        elif self.split == "val":
            frame = np.random.randint(low=0, high=self.num_frames)
            frames = np.array([frame])[:, None]  # 1,1

            view = np.random.randint(low=0, high=self.view_num)
            pose = self.poses[view, :3, :4]

            rays_o, rays_d = get_rays(
                self.H, self.W, self.K, torch.Tensor(pose)
            )  # (H, W, 3), (H, W, 3)

            target = self.imgs[view, frame, ...]
            target = torch.Tensor(target)

            semantic_img = self.semantic_imgs[view, frame, ...]  # H*W
            semantic_img = torch.Tensor(semantic_img)

            car_mask = semantic_img == 10
            noncar_mask = semantic_img != 10

            rays_o_car = rays_o[car_mask]
            rays_d_car = rays_d[car_mask]
            target_car = target[car_mask]

            rays_o_noncar = rays_o[noncar_mask]
            rays_d_noncar = rays_d[noncar_mask]
            target_noncar = target[noncar_mask]

        return {
            "rays_o_car": rays_o_car,
            "rays_d_car": rays_d_car,
            "target_car": target_car,
            "rays_o_noncar": rays_o_noncar,
            "rays_d_noncar": rays_d_noncar,
            "target_noncar": target_noncar,
        }
