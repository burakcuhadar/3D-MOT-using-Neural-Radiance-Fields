import os
import numpy as np
from numpy.random import default_rng
from glob import glob
import imageio 
import torch
import re
from pytorch3d.transforms import se3_log_map
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


def pose_translational(t):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0], 
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def invert_transformation(t):
    t_inv = np.eye(4, dtype=np.float32)
    t_inv[:3,:3] = t[:3,:3].T
    t_inv[:3,-1] = -t[:3,:3].T @ t[:3,-1]
    return t_inv


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
    def __init__(self, args, split, num_frames=None):
        super().validate_split(split)
        self.split = split
        self.num_frames = num_frames

        if split == 'render_video':
            # Render video by moving the car
            self.object_poses = np.stack([pose_translational(t) for t in np.arange(-2., 8., 0.25)], axis=0) 
            #self.object_poses = self.get_gt_vehicle_poses(args)
            poses = pose_spherical(-90., 15.)[None, ...]
            imgs = None
            frames = None
        else:
            imgs, poses, frames = self.load_imgs_poses(args, split)

        self.gt_vehicle_poses = self.get_gt_vehicle_poses(args)
        self.gt_relative_poses = self.load_gt_relative_poses(args)
        H, W, focal = self.load_intrinsics(args)
        self.H = int(H)
        self.W = int(W)
        self.focal = focal
        
        self.imgs = imgs
        self.poses = poses
        self.frames = frames
        self.near = 3. 
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
        '''pose0 = None
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
                # how to scale?
                poses.append(pose.astype(np.float32))
        '''
        for f in pose_files:
            posei = from_ue4_to_nerf(np.load(f))
            if args.scale_factor > 0:
                posei[:3,3] *= args.scale_factor
            posei_inv = np.eye(4, dtype=np.float32)
            posei_inv[:3,:3] = posei[:3,:3].T
            posei_inv[:3,-1] = -posei[:3,:3].T @ posei[:3,-1]    
            poses.append(posei_inv.astype(np.float32))

        poses = np.stack(poses, axis=0)
        poses = torch.from_numpy(poses)
        return poses
    

    def load_gt_relative_poses(self, args):
        poses = []
        pose0 = None
        for i,pose in enumerate(self.gt_vehicle_poses):
            if args.scale_factor > 0:
                pose[:3,3] *= args.scale_factor
            if i == 0:
                #pose0_inv = invert_transformation(pose)
                pose0 = pose
                poses.append(np.eye(4, dtype=np.float32))
            else:
                pose_inv = invert_transformation(pose)
                #posei_0 = pose_inv @ pose0.numpy()
                posei_0 = pose0.numpy() @ pose_inv
                # for pytorch3d 4x4 format
                posei_0_ = np.eye(4, dtype=np.float32)
                posei_0_[:3,:3] = posei_0[:3,:3]
                posei_0_[3,:3] = posei_0[:3,3]
                poses.append(posei_0_)

        poses = np.stack(poses, axis=0)
        poses = torch.from_numpy(poses) # num_frames, 4, 4
        with torch.no_grad():
            self.gt_relative_poses_matrices = poses.clone()
        poses = se3_log_map(poses) # num_frames, 6
        
        return poses

    def get_noisy_gt_relative_poses(self):        
        print('gt relative poses', self.gt_relative_poses)
        noise = torch.randn((self.gt_relative_poses.shape[0]-1, 6), dtype=torch.float32) / 100.
        noisy_poses = torch.zeros_like(self.gt_relative_poses)
        noisy_poses += self.gt_relative_poses
        noisy_poses[1:,:] += noise
        return noisy_poses


    def load_imgs_poses(self, args, split):
        extrinsics = np.load(os.path.join(args.datadir, 'extrinsics.npy'), allow_pickle=True).item()

        cameras = sorted(glob(args.datadir + '/camera*/'), key=natural_keys)

        imgs = []
        poses = []
        
        for i, cam in enumerate(cameras):
            if self.split == 'train_appearance' or self.split == 'train_online' or self.split == 'train_render':
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

