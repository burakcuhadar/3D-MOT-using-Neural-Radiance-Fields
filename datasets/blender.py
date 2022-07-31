import os
import torch
from torch.utils.data import Dataset
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
from datasets.base import BaseDataset

from models.rendering import get_rays_np, get_rays, ndc_rays

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


class BlenderDataset(BaseDataset):
    def __init__(self, args, split='train'):
        super().validate_split(split)
        self.split = split

        if split == 'render_video':
            if args.render_test:
                # Render video using test set views
                imgs, poses, H, W, focal = self.load_imgs_poses(args, 'test')
                imgs, H, W, focal = self.preprocess_imgs(imgs, H, W, focal)
            else:
                # Render video using evenly spaced views instead of test set views
                poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
                H, W, focal = self.load_hwf(args)
                imgs = None
        else:
            imgs, poses, H, W, focal = self.load_imgs_poses(args, split)
            imgs, H, W, focal = self.preprocess_imgs(args, imgs, H, W, focal)
        
        self.H = int(H)
        self.W = int(W)
        self.focal = focal
        
        self.imgs = imgs
        self.poses = poses
        self.near = 2.
        self.far = 6.
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        self.K = K


        ### Blender data is loaded
        super().__init__(args)


    def load_imgs_poses(self, args, split):
        ### Blender specific part
        with open(os.path.join(args.datadir, 'transforms_{}.json'.format(split)), 'r') as fp:
            meta = json.load(fp)

        # Read images and poses
        imgs = []
        poses = []
        if split=='train' or args.testskip==0:
            skip = 1
        else:
            skip = args.testskip
        for frame in meta['frames'][::skip]:
            fname = os.path.join(args.datadir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

        return imgs, poses, H, W, focal

    def load_hwf(self, args):
        # Load h, w, f from test meta data
        with open(os.path.join(args.datadir, 'transforms_test.json'), 'r') as fp:
            meta = json.load(fp)
        
        frame = meta['frames'][0]
        fname = os.path.join(args.datadir, frame['file_path'] + '.png')
        img = imageio.imread(fname)
        H, W = img.shape[:2]

        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

        return H, W, focal

    
    def preprocess_imgs(self, args, imgs, H, W, focal):
        if args.half_res:
            H = H//2
            W = W//2
            focal = focal/2.

            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
            # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    
        if args.white_bkgd:
            imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
        else:
            imgs = imgs[...,:3]

        return imgs, H, W, focal






