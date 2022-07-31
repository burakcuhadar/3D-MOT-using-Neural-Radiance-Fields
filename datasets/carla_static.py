import os
import numpy as np
from glob import glob
import imageio 
import torch
import re

from datasets.base import BaseDataset


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



class CarlaStaticDataset(BaseDataset):
    def __init__(self, args, split='train'):
        super().validate_split(split)
        self.split = split

        if split == 'render_video':
            # Render video using evenly spaced views
            poses = np.stack([pose_spherical(angle, 20.0) for angle in np.linspace(-180,180,40+1)[:-1]], axis=0)
            imgs = None
        else:
            imgs, poses = self.load_imgs_poses(args, split)

        H, W, focal = self.load_intrinsics(args)
        self.H = int(H)
        self.W = int(W)
        self.focal = focal
        
        self.imgs = imgs
        self.poses = poses
        self.near = 6. 
        self.far = 42.
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


    def load_imgs_poses(self, args, split):
        extrinsics = np.load(os.path.join(args.datadir, 'extrinsics.npy'), allow_pickle=True).item()

        cameras = sorted(glob(args.datadir + '/camera*/'), key=natural_keys)
        if self.split == 'train':
            cameras = cameras[:50]
        elif self.split == 'val':
            cameras = cameras[50:]

        imgs = []
        poses = []
        
        for i, cam in enumerate(cameras):
            print(cam, 'goes to', self.split)
            print(cam, ' extrinsics:', extrinsics[i])

            imgpath = glob(cam + '*.png')[0] # Only one image per dir
            imgs.append(imageio.imread(imgpath))
            poses.append(from_ue4_to_nerf(extrinsics[i]))

        imgs = (np.array(imgs) / 255.).astype(np.float32)[...,:3]
        poses = np.array(poses).astype(np.float32)

        return imgs, poses
        


def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

