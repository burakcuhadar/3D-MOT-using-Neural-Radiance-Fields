import torch
import os
from glob import glob
import numpy as np
import pypose as pp
import imageio
import logging
import time
import random
from scipy.spatial.transform import Rotation


from lietorch import SE3

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from models.rendering import get_rays, get_rays_np
from utils.dataset import (
    load_intrinsics,
    natural_keys,
    from_ue4_to_nerf,
    invert_transformation,
    pose_translational,
    pose_spherical,
    se3_log_map,
    pose_rotational,
    to_quaternion,
    to_euler
)


class StarOnlineDataset(Dataset):
    def __init__(self, args, split, num_frames, current_frame, start_frame=0):
        print(
            f"creating {split} dataset with start_frame={start_frame} num_frames={num_frames}"
        )
        self.validate_split(split)
        self.split = split
        self.has_depth_data = args.has_depth_data
        self.num_frames = num_frames
        self.current_frame = current_frame
        self.start_frame = start_frame
        self.N_samples = args.N_samples
        self.N_rand = args.N_rand
        self.car_sample_ratio = args.car_sample_ratio

        self.gt_relative_poses = torch.from_numpy(self.load_gt_relative_poses(args)).float()
        #with torch.no_grad():
        #    self.gt_relative_poses_quat = SE3.exp(self.gt_relative_poses).vec()
        self.gt_vehicle_poses = self.get_gt_vehicle_poses(args)

        if split == "test":
            # self.object_poses = np.stack(
            #    [pose_translational(t) for t in np.arange(0.0, 1.0, 0.25)], axis=0
            # )

            self.object_poses = np.stack(
                [pose_rotational(deg) for deg in np.arange(0.0, 360.0, 20.0)], axis=0
            )

            if args.scale_factor > 0:
                self.object_poses[:, :3, 3] *= args.scale_factor

            # self.object_poses = self.get_gt_vehicle_poses(args)
            poses = pose_spherical(0, 25.0)[None, ...]
            imgs = None
            semantic_imgs = None
            depth_imgs = None
        else:
            imgs, poses, semantic_imgs, depth_imgs = self.load_imgs_poses(args)

        H, W, focal = load_intrinsics(args)

        self.H = int(H)
        self.W = int(W)
        self.focal = focal

        self.imgs = imgs
        self.semantic_imgs = semantic_imgs
        self.poses = poses
        self.depth_imgs = depth_imgs

        self.near = args.near
        self.far = args.far

        if args.scale_factor > 0:
            self.near *= args.scale_factor
            self.far *= args.scale_factor
            self.poses[:, :3, 3] *= args.scale_factor

            if args.has_depth_data and self.depth_imgs is not None:
                self.depth_imgs *= args.scale_factor

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

            rays = np.expand_dims(rays, axis=1).repeat(
                num_frames, axis=1
            )  # [N,frame_num,2,H,W,3]
            rays_o = rays[:, :, 0, :, :, :]  # [N,frame_num,H,W,3]
            rays_d = rays[:, :, 1, :, :, :]  # [N,frame_num,H,W,3]

            # frame_num = rays_o.shape[1]
            rays_o = np.swapaxes(rays_o, 0, 1)  # [frame_num, N,H,W,3]
            rays_d = np.swapaxes(rays_d, 0, 1)  # [frame_num, N,H,W,3]
            rays_o = np.reshape(rays_o, [num_frames, -1, 3])  # [frame_num, N*H*W, 3]
            rays_d = np.reshape(rays_d, [num_frames, -1, 3])  # [frame_num, N*H*W, 3]

            imgs = np.swapaxes(self.imgs, 0, 1)
            target_rgbs = np.reshape(
                imgs, [num_frames, -1, 3]
            )  # [num_frames, N*H*W, 3]

            rays_o = rays_o.astype(np.float32)
            rays_d = rays_d.astype(np.float32)

            self.rays_o = rays_o
            self.rays_d = rays_d
            self.target_rgbs = target_rgbs

            print("rays_o", self.rays_o.shape)
            print("rays_d", self.rays_d.shape)
            print("target_rgbs", self.target_rgbs.shape)

            if args.has_depth_data:
                self.target_depths = np.swapaxes(
                    self.depth_imgs, 0, 1
                )  # [frame_num, view num, H, W]
                self.target_depths = np.reshape(
                    self.target_depths, [num_frames, -1]
                )  # [num_frames, N*H*W]

            if self.semantic_imgs is not None:
                semantic_rays = np.swapaxes(
                    self.semantic_imgs, 0, 1
                )  # [frame_num, N, H, W]
                semantic_rays = np.reshape(
                    semantic_rays, [num_frames, -1]
                )  # [frame_num, N*H*W]
                self.semantic_rays = semantic_rays

                print("semantic rays shape", semantic_rays.shape)

            """
            car_mask = semantic_rays == 10
            noncar_mask = semantic_rays != 10

            print("car mask shape", car_mask.shape)
            print("noncar mask shape", noncar_mask.shape)

            self.rays_o_car = rays_o[car_mask]
            self.rays_d_car = rays_d[car_mask]
            self.target_rgbs_car = target_rgbs[car_mask]

            self.rays_o_noncar = rays_o[noncar_mask]
            self.rays_d_noncar = rays_d[noncar_mask]
            self.target_rgbs_noncar = target_rgbs[noncar_mask]

            print("rays_o car", self.rays_o_car.shape)
            print("rays_d car", self.rays_d_car.shape)
            print("target_rgbs car", self.target_rgbs_car.shape)
            print("rays_o noncar", self.rays_o_noncar.shape)
            print("rays_d noncar", self.rays_d_noncar.shape)
            print("target_rgbs noncar", self.target_rgbs_noncar.shape)
            """

    def load_imgs_poses(self, args):
        # How many images we have for one frame: rgb, semantic, (depth)
        # img_num_for_one_frame = 3 if args.has_depth_data else 2

        extrinsics = np.load(
            os.path.join(args.datadir, "extrinsics.npy"), allow_pickle=True
        ).item()

        cameras = sorted(glob(f"{args.datadir}/camera*/"), key=natural_keys)

        imgs = []
        poses = []
        semantic_imgs = []

        if args.has_depth_data:
            depth_imgs = []

        for i, cam in enumerate(cameras):
            if self.split == "train":
                if i >= 50:
                    continue
            elif self.split == "val":
                if i < 50:
                    continue
                # Currently, I skip the problematic val view
                # if i == len(cameras) - 1:
                #   continue
            elif self.split == "test":
                if i <= 55:
                    continue
            

            logging.info(f"{cam} goes to {self.split}")

            imgpaths = []
            semantic_imgpaths = []
            if args.has_depth_data:
                depth_imgs_cam = []

            for path in sorted(glob(f"{cam}*.png"), key=natural_keys):
                if path.endswith("_semantic.png"):
                    semantic_imgpaths.append(path)
                elif path.endswith("_depth.png"):
                    depth_img = imageio.imread(path).astype(np.uint8)
                    normalized = (
                        depth_img[:, :, 0]
                        + depth_img[:, :, 1] * 256.0
                        + depth_img[:, :, 2] * 256.0 * 256.0
                    ) / (256.0 * 256.0 * 256.0 - 1.0)
                    in_meters = 1000 * normalized
                    depth_imgs_cam.append(in_meters.astype(np.float32))
                else:
                    imgpaths.append(path)
            semantic_imgs.append(
                [imageio.imread(imgpath) for imgpath in semantic_imgpaths]
            )

            imgs.append([imageio.imread(imgpath) for imgpath in imgpaths])
            poses.append(from_ue4_to_nerf(extrinsics[i]))

            if args.has_depth_data:
                depth_imgs.append(depth_imgs_cam)

        imgs = (np.array(imgs) / 255.0).astype(np.float32)[
            ..., :3
        ]  # [view_num, frame_num, H, W, 3]
        poses = np.array(poses).astype(np.float32)  # [view_num, 4, 4]
        self.view_num = len(poses)

        if len(semantic_imgs[0]) > 0:
            semantic_imgs = np.array(semantic_imgs).astype(np.uint8)[
                ..., 0
            ]  # [view_num, frame_num, H, W]
        else:
            semantic_imgs = None

        if args.has_depth_data:
            depth_imgs = np.array(depth_imgs)  # [view num, frame_num, H, W]
            print("depth imgs shape", depth_imgs.shape)
        else:
            depth_imgs = None

        return imgs, poses, semantic_imgs, depth_imgs

    def __len__(self):
        if self.split == "train":
            return 1000
        elif self.split == "val":
            return 1
        elif self.split == "test":
            return len(self.object_poses)
        else:
            raise ValueError("invalid dataset split")

    def __getitem__(self, idx):
        target_depth = None

        if self.split == "train":
            if self.car_sample_ratio == 0:
                """No semantic rays"""
                frame = np.random.randint(low=self.start_frame, high=self.current_frame)
                """frame = random.choice(
                    [0]
                    + list(
                        range(
                            self.start_frame if self.start_frame != 0 else 1,
                            self.current_frame,
                        )
                    )
                )"""
                frames = np.array([frame])[:, None]  # 1,1

                indices = np.random.choice(self.rays_o.shape[1], self.N_rand)
                rays_o = self.rays_o[frame, indices, ...]
                rays_d = self.rays_d[frame, indices, ...]
                target = self.target_rgbs[frame, indices, ...]

                if self.has_depth_data:
                    target_depth = self.target_depths[frame, indices, ...]
            else:
                """Semantic rays"""
                frame = np.random.randint(low=self.start_frame, high=self.current_frame)
                """frame = random.choice(
                    [0]
                    + list(
                        range(
                            self.start_frame if self.start_frame != 0 else 1,
                            self.current_frame,
                        )
                    )
                )"""
                frames = np.array([frame])[:, None]  # 1,1

                car_sample_num = int(self.N_rand * self.car_sample_ratio)
                noncar_sample_num = self.N_rand - car_sample_num

                car_mask = self.semantic_rays[frame] == 10
                noncar_mask = self.semantic_rays[frame] != 10

                rays_o_car = self.rays_o[frame, car_mask, ...]
                rays_d_car = self.rays_d[frame, car_mask, ...]
                target_car = self.target_rgbs[frame, car_mask, ...]
                rays_o_noncar = self.rays_o[frame, noncar_mask, ...]
                rays_d_noncar = self.rays_d[frame, noncar_mask, ...]
                target_noncar = self.target_rgbs[frame, noncar_mask, ...]

                car_indices = np.random.choice(rays_o_car.shape[0], car_sample_num)
                noncar_indices = np.random.choice(
                    rays_o_noncar.shape[0], noncar_sample_num
                )

                rays_o_car = rays_o_car[car_indices]
                rays_d_car = rays_d_car[car_indices]
                target_car = target_car[car_indices]
                rays_o_noncar = rays_o_noncar[noncar_indices]
                rays_d_noncar = rays_d_noncar[noncar_indices]
                target_noncar = target_noncar[noncar_indices]

                rays_o = np.concatenate([rays_o_car, rays_o_noncar], axis=0)
                rays_d = np.concatenate([rays_d_car, rays_d_noncar], axis=0)
                target = np.concatenate([target_car, target_noncar], axis=0)

                p = np.random.permutation(len(rays_o))
                rays_o = rays_o[p]
                rays_d = rays_d[p]
                target = target[p]

        elif self.split == "val":
            logging.info(f"choosing btw {self.start_frame} and {self.current_frame}")
            frame = np.random.randint(low=self.start_frame, high=self.current_frame)
            frames = np.array([frame])[:, None]  # 1,1

            view = np.random.randint(low=0, high=self.view_num)

            pose = self.poses[view, :3, :4]
            rays_o, rays_d = get_rays(
                self.H, self.W, self.K, torch.Tensor(pose)
            )  # (H, W, 3), (H, W, 3)
            target = self.imgs[view, frame, ...]
            target = torch.Tensor(target)

            rays_o = torch.reshape(rays_o, [-1, 3])  # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1, 3])  # (H*W, 3)
            target = torch.reshape(target, [-1, 3])  # (H*W, 3)

        elif self.split == "test":
            pose = self.poses[0, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))
            rays_o = torch.reshape(rays_o, [-1, 3])  # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1, 3])  # (H*W, 3)
            target = None
            frames = None
            object_pose = self.object_poses[idx, ...]

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "target": target,
            "frames": frames,
            "target_depth": target_depth,
            "gt_relative_poses": self.gt_relative_poses,  # used for logging/debugging
            "gt_vehicle_poses": self.gt_vehicle_poses,
            "object_pose": object_pose if self.split == "test" else None,
        }

    # used for debugging
    def find_bounds(self):
        farthest = self.rays_o + self.far * self.rays_d
        norms = np.linalg.norm(farthest, axis=-1)
        print("norms shape", norms.shape)
        farthest_ind = np.argmax(norms)
        print("farthest norm", norms[farthest_ind])
        print("farthes coords", farthest[farthest_ind])
        print("largest x", np.max(farthest[:, 0]))
        print("largest y", np.max(farthest[:, 1]))
        print("largest z", np.max(farthest[:, 2]))
        print("smallest x", np.min(farthest[:, 0]))
        print("smallest y", np.min(farthest[:, 1]))
        print("smallest z", np.min(farthest[:, 2]))

    @staticmethod
    def validate_split(split):
        splits = ["train", "val", "test"]
        assert split in splits, "Dataset split should be one of " + ", ".join(splits)

    # Used for debugging
    @torch.no_grad()
    def get_gt_vehicle_poses(self, args):
        pose_files = sorted(glob(args.datadir + "/poses/*.npy"), key=natural_keys)
        poses = []

        for f in pose_files:
            posei = from_ue4_to_nerf(np.load(f))
            if args.scale_factor > 0:
                posei[:3, 3] *= args.scale_factor
            posei_inv = np.eye(4, dtype=np.float32)
            posei_inv[:3, :3] = posei[:3, :3].T
            posei_inv[:3, -1] = -posei[:3, :3].T @ posei[:3, -1]
            poses.append(posei_inv.astype(np.float32))

        poses = np.stack(poses, axis=0)
        poses = torch.from_numpy(poses)
        return poses

    # Used for trans/rot error logging
    def load_gt_relative_poses(self, args):
        pose_files = sorted(glob(args.datadir + "/poses/*.npy"), key=natural_keys)

        poses_matrices = []

        pose0 = None
        for i, f in enumerate(pose_files):
            pose = from_ue4_to_nerf(np.load(f))
            if args.scale_factor > 0:
                pose[:3, 3] *= args.scale_factor
            if i == 0:
                # pose0_inv = invert_transformation(pose)
                pose0 = pose.astype(np.float32)
                poses_matrices.append(np.eye(4, dtype=np.float32))
            else:
                pose_inv = invert_transformation(pose)
                # posei_0 = pose_inv @ pose0.numpy()
                posei_0 = pose0 @ pose_inv  
                poses_matrices.append(posei_0)

        poses_matrices = np.stack(poses_matrices, axis=0)
        with torch.no_grad():
            self.gt_relative_poses_matrices = torch.from_numpy(poses_matrices)

        poses_log = se3_log_map(poses_matrices)  # num_frames, 6

        return poses_log

    @torch.no_grad()
    def get_noisy_gt_relative_poses(self):
        print("gt relative poses\n", self.gt_relative_poses)
        
        """
        gt_matrix = SE3.exp(self.gt_relative_poses).matrix().numpy()
        gt_rot = gt_matrix[:, :3, :3]
        gt_rot_euler = Rotation.from_matrix(gt_rot).as_euler("xyz")
        print("gt_rot_euler\n", gt_rot_euler)
        gt_trans = gt_matrix[:, :3, 3]
        """

        gt_rot_euler = pp.SE3(self.gt_relative_poses).rotation().euler().numpy()
        print("gt_rot_euler\n", gt_rot_euler)
        gt_trans = pp.SE3(self.gt_relative_poses).translation().numpy()
        
        rot_noise = np.random.randn(self.gt_relative_poses.shape[0] - 1) * np.pi / 16 - np.pi / 32
        trans_noise = np.random.randn(self.gt_relative_poses.shape[0] - 1, 3) / 100.0
        
        noisy_rot = np.zeros((self.gt_relative_poses.shape[0], 3))
        noisy_rot += gt_rot_euler
        noisy_rot[1:, 1] += rot_noise # we only add noise to y-axis rotation
        
        noisy_trans = np.zeros((self.gt_relative_poses.shape[0], 3))
        noisy_trans += gt_trans
        noisy_trans[1:, ...] += trans_noise
        
        noisy_pose_matrix = np.eye(4, dtype=np.float32)[None, ...].repeat(self.gt_relative_poses.shape[0], axis=0)
        noisy_pose_matrix[:, :3, :3] = Rotation.from_euler("xyz", noisy_rot).as_matrix()
        noisy_pose_matrix[:, :3, 3] = noisy_trans

        noisy_poses_log = se3_log_map(noisy_pose_matrix)
        noisy_poses_log = torch.from_numpy(noisy_poses_log)
        
        return noisy_poses_log
