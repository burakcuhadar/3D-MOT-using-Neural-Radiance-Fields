import torch
import os
from glob import glob
import numpy as np
import imageio 



from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from models.rendering import get_rays, get_rays_np
from utils.dataset import load_intrinsics, natural_keys, from_ue4_to_nerf


class StarAppInitDataset(Dataset):
    def __init__(self, args, split):
        self.validate_split(split)
        self.split = split
        self.N_samples = args.N_samples
        self.N_rand = args.N_rand
        self.use_batching = not args.no_batching #TODO eliminate the need for it

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
            self.poses[:,:3,3] *= args.scale_factor

        print('near', self.near)
        print('far', self.far)

        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

        self.K = K


        if self.split == 'train':
            # [N,ro+rd,H,W,3]
            rays = np.stack([get_rays_np(self.H, self.W, self.K, p) for p in self.poses[:,:3,:4]], 0) 
            rays_o = rays[:,0,:,:,:] # [N,H,W,3]
            rays_d = rays[:,1,:,:,:] # [N,H,W,3]

            rays_o = np.reshape(rays_o, [-1, 3]) # [N*H*W,3]
            rays_d = np.reshape(rays_d, [-1, 3]) # [N*H*W,3]
            target_rgbs = np.reshape(self.imgs, [-1, self.imgs.shape[-1]]) # [N*H*W,3]

            rays_o = rays_o.astype(np.float32)
            rays_d = rays_d.astype(np.float32)

            self.rays_o = rays_o
            self.rays_d = rays_d
            self.target_rgbs = target_rgbs

            print('rays_o', self.rays_o.shape) 
            print('rays_d', self.rays_d.shape) 
            print('target_rgbs', self.target_rgbs.shape)
    
    def load_imgs_poses(self, args):
        extrinsics = np.load(os.path.join(args.datadir, 'extrinsics.npy'), allow_pickle=True).item()

        cameras = sorted(glob(args.datadir + '/camera*/'), key=natural_keys)

        imgs = []
        poses = []

        for i, cam in enumerate(cameras):
            if self.split == 'train':
                if i >= 50:
                    continue
            elif self.split == 'val':
                if i < 50:
                    continue
                # Currently, I skip the problematic val view
                if i == len(cameras)-1:
                    continue
            print(cam, 'goes to', self.split)

            if self.split == 'train':
                imgpaths = []
                # :3 since there are also semantic and depth images
                for path in sorted(glob(cam + '*.png'), key=natural_keys)[:3]: 
                    if path.endswith('_semantic.png'):
                        pass
                    elif path.endswith('_depth.png'):
                        pass
                    else:
                        imgpaths.append(path)
            elif self.split == 'val':
                imgpaths = sorted(glob(cam + '*.png'), key=natural_keys)[:1]

            imgs.append([imageio.imread(imgpath) for imgpath in imgpaths])
            poses.append(from_ue4_to_nerf(extrinsics[i]))

        imgs = (np.array(imgs) / 255.).astype(np.float32)[...,:3] # [view_num, frame_num, H, W, 3]
        imgs = np.squeeze(imgs, axis=1)
        poses = np.array(poses).astype(np.float32) # [view_num, 4, 4]
        
        return imgs, poses

    def __len__(self):
        if self.split == 'train':
            #return len(self.rays_o)
            return 1000
        elif self.split == 'val':
            return 1
        else:
            raise ValueError("invalid dataset split")


    def __getitem__(self, idx):
        if self.split == 'train':
            indices = np.random.choice(len(self.rays_o), self.N_rand)
            rays_o = self.rays_o[indices, ...]
            rays_d = self.rays_d[indices, ...]
            target = self.target_rgbs[indices, ...]
            
            if not self.use_batching:
                raise NotImplementedError
    
        elif self.split == 'val':
            # Load all of the rays of one image
            target = self.imgs[idx] 
            target = torch.Tensor(target) 
            pose = self.poses[idx, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            rays_o = torch.reshape(rays_o, [-1,3]) # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1,3]) # (H*W, 3)
            target = torch.reshape(target, [-1,3]) # (H*W, 3)
        
        return {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'target': target
        }


    # used for debugging
    def find_bounds(self):
        farthest = self.rays_o + self.far * self.rays_d
        norms = np.linalg.norm(farthest, axis=-1)
        print('norms shape', norms.shape)
        farthest_ind = np.argmax(norms)
        print('farthest norm', norms[farthest_ind])
        print('farthes coords', farthest[farthest_ind])
        print('largest x', np.max(farthest[:,0]))
        print('largest y', np.max(farthest[:,1]))
        print('largest z', np.max(farthest[:,2]))
        print('smallest x', np.min(farthest[:,0]))
        print('smallest y', np.min(farthest[:,1]))
        print('smallest z', np.min(farthest[:,2]))

    
    @staticmethod
    def validate_split(split):
        splits = ['train', 'val', 'test'] 
        assert split in splits, "Dataset split should be one of " + ", ".join(splits)

        