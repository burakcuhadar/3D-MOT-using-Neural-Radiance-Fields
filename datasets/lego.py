import torch
import os
from glob import glob
import numpy as np
import imageio
import cv2
import json


from torch.utils.data import Dataset
from models.rendering import get_rays, get_rays_np
from utils.dataset import load_intrinsics, natural_keys, from_ue4_to_nerf

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

class LegoDataset(Dataset):
    def __init__(self, args, split):
        self.validate_split(split)
        self.split = split
        #self.has_depth_data = args.has_depth_data
        self.N_samples = args.N_samples
        self.N_rand = args.N_rand
        self.args = args

        imgs, poses, render_poses, H, W, focal, i_split = self.load_blender_data(args.datadir)

        self.H = int(H)
        self.W = int(W)
        self.focal = focal

        self.imgs = imgs
        #self.semantic_imgs = semantic_imgs
        self.poses = poses
        #self.depth_imgs = depth_imgs

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

            self.rays_o = rays_o
            self.rays_d = rays_d
            self.target_rgbs = target_rgbs

            print("rays_o", self.rays_o.shape)
            print("rays_d", self.rays_d.shape)
            print("target_rgbs", self.target_rgbs.shape)

    def load_blender_data(self, basedir, half_res=False, testskip=1):
        splits = ['train', 'val', 'test']
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            if s=='train' or testskip==0:
                skip = 1
            else:
                skip = testskip
                
            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
        
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
        
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        
        if half_res:
            H = H//2
            W = W//2
            focal = focal/2.

            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
            # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        if self.args.white_bkgd:
            imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
        else:
            imgs = imgs[...,:3]
            
        return imgs, poses, render_poses, H, W, focal, i_split

    def __len__(self):
        if self.split == "train":
            # return len(self.rays_o)
            return 1000
        elif self.split == "val":
            return 1
        else:
            raise ValueError("invalid dataset split")

    def __getitem__(self, idx):
        if self.split == "train":
            indices = np.random.choice(len(self.rays_o), self.N_rand)
            rays_o = self.rays_o[indices, ...]
            rays_d = self.rays_d[indices, ...]
            target = self.target_rgbs[indices, ...]

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

            
        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "target": target,
        }

    @staticmethod
    def validate_split(split):
        splits = ["train", "val", "test"]
        assert split in splits, "Dataset split should be one of " + ", ".join(splits)
