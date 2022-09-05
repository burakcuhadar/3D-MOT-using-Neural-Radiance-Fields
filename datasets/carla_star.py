import os
import numpy as np
from glob import glob
import imageio 
import torch
import re

from datasets.base_star import BaseStarDataset


# Translation in UE4
trans_t = lambda t : np.array([
    [1,0,0,t],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]], dtype=np.float32)
trans_z = lambda z : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,z],
    [0,0,0,1]], dtype=np.float32)
# Rotation around z-axis in UE4
rot_theta = lambda th : np.array([
    [np.cos(th),np.sin(th),0,0],
    [-np.sin(th),np.cos(th),0,0],
    [0,0,1,0],
    [0,0,0,1]], dtype=np.float32)
# Rotation around y-axis in UE4
rot_phi = lambda phi : np.array([
    [np.cos(phi),0,-np.sin(phi),0],
    [0,1,0,0],
    [np.sin(phi),0,np.cos(phi),0],
    [0,0,0,1]], dtype=np.float32)


def pose_spherical(theta, radius):
    c2w = trans_z(6.)
    c2w = rot_phi(-25. / 180. * np.pi) @ c2w
    c2w = rot_theta(-np.pi) @ c2w
    c2w = trans_t(radius) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    
    c2w = from_ue4_to_nerf(c2w)
    return c2w


def from_ue4_to_nerf(pose):
    change_ue4_to_nerf = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0]
    ], dtype=np.float32)

    # inverse of the above
    change_nerf_to_ue4 = np.array([
        [0, 0, -1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    new_pose = np.eye(pose.shape[0], pose.shape[1])

    # Rotation
    new_pose[:3,:3] = change_ue4_to_nerf @ pose[:3,:3] @ change_nerf_to_ue4
    # Translation
    new_pose[:3,-1] = change_ue4_to_nerf @ pose[:3,-1] 

    return new_pose



class CarlaStarDataset(BaseStarDataset):
    def __init__(self, args, split, num_frames):
        super().validate_split(split)
        self.split = split
        self.num_frames = num_frames

        if split == 'render_video': # TODO
            # Render video using evenly spaced views
            poses = np.stack([pose_spherical(angle, 20.0) for angle in np.linspace(-180,180,40+1)[:-1]], axis=0)
            imgs = None
        else:
            imgs, poses, frames = self.load_imgs_poses(args, split)

        
        H, W, focal = self.load_intrinsics(args)
        self.H = int(H)
        self.W = int(W)
        self.focal = focal
        
        self.imgs = imgs
        self.poses = poses
        self.frames = frames
        self.near = 6. 
        self.far = 100.
        print('near', self.near)
        print('far', self.far)
        
        if args.scale_factor > 0:
            self.near *= args.scale_factor
            self.far *= args.scale_factor
            self.poses[:,:3,3] *= args.scale_factor

        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        self.K = K

        super().__init__(args)

    def load_intrinsics(self, args):
        intrinsics = np.load(os.path.join(args.datadir, 'intrinsics.npy'), allow_pickle=True).item()
        H = intrinsics['h']
        W = intrinsics['w']
        fov = intrinsics['fov']
        focal = W / (2 * np.tan(fov * np.pi / 360))

        return H, W, focal
    
    def get_gt_vehicle_poses(self, args):
        pose_files = sorted(glob(args.datadir + '/poses/*.npy'), key=natural_keys)
        poses = []
        pose0 = None
        for i, f in enumerate(pose_files):
            if i == 0:
                pose0 = from_ue4_to_nerf(np.load(f))
                poses.append(np.eye(4, dtype=np.float32))
            else:
                posei = from_ue4_to_nerf(np.load(f))
                posei_inv = np.eye(4, dtype=np.float32)
                posei_inv[:3,:3] = posei[:3,:3].T
                posei_inv[:3,-1] = -posei[:3,:3].T @ posei[:3,-1]
                pose = pose0 @ posei_inv
                poses.append(pose.astype(np.float32))
        poses = np.stack(poses, axis=0)
        poses = torch.from_numpy(poses)
        return poses

    def load_imgs_poses(self, args, split):
        extrinsics = np.load(os.path.join(args.datadir, 'extrinsics.npy'), allow_pickle=True).item()

        cameras = sorted(glob(args.datadir + '/camera*/'), key=natural_keys)

        imgs = []
        poses = []
        
        for i, cam in enumerate(cameras):
            if self.split == 'train_appearance' or self.split == 'train_online':
                if i >= 50:
                    continue
            elif self.split == 'val_appearance' or self.split == 'val_online':
                if i < 50:
                    continue
            print(cam, 'goes to', self.split)
            #print(cam, ' extrinsics:', extrinsics[i])

            imgpaths = sorted(glob(cam + '*.png'), key=natural_keys)[:self.num_frames]
            # if self.split == 'train_appearance' or self.split == 'val_appearance':  
            #     # The first frame is used for appearance init
            #     imgpaths = imgpaths[:1]
            imgs.append([imageio.imread(imgpath) for imgpath in imgpaths])
            poses.append(from_ue4_to_nerf(extrinsics[i]))

        imgs = (np.array(imgs) / 255.).astype(np.float32)[...,:3] # [view_num, frame_num, H, W, 3]
        poses = np.array(poses).astype(np.float32) # [view_num, 4, 4]
        frames = None

        if self.split == 'train_appearance' or self.split == 'val_appearance':
            imgs = np.squeeze(imgs, axis=1)
        else:
            frames = np.arange(imgs.shape[1])[None,:].repeat(imgs.shape[0], axis=0) # [view_num, frame_num]


        return imgs, poses, frames
        


def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

