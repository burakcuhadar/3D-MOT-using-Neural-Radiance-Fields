import torch
import os
from glob import glob
import numpy as np
import imageio 


from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from models.rendering import get_rays, get_rays_np
from utils.dataset import load_intrinsics, natural_keys, from_ue4_to_nerf


class StarOnlineDataset(Dataset):
    def __init__(self, args, split, num_frames):
        self.validate_split(split)
        self.split = split
        self.num_frames = num_frames
        self.N_samples = args.N_samples
        self.N_rand = args.N_rand
        self.use_batching = not args.no_batching #TODO eliminate the need for it

        imgs, poses, semantic_imgs = self.load_imgs_poses(args)
        H, W, focal = load_intrinsics(args)

        self.H = int(H)
        self.W = int(W)
        self.focal = focal

        self.imgs = imgs
        self.semantic_imgs = semantic_imgs
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
            frame_num = num_frames 
            rays = np.expand_dims(rays, axis=1).repeat(frame_num, axis=1) # [N,frame_num,2,H,W,3] 
            rays_o = rays[:,:,0,:,:,:] # [N,frame_num,H,W,3]
            rays_d = rays[:,:,1,:,:,:] # [N,frame_num,H,W,3]

            N = rays_o.shape[0]
            #frame_num = rays_o.shape[1]
            rays_o = np.swapaxes(rays_o, 0, 1) # [frame_num, N,H,W,3]
            rays_d = np.swapaxes(rays_d, 0, 1) # [frame_num, N,H,W,3]
            rays_o = np.reshape(rays_o, [frame_num, -1, 3]) # [frame_num, N*H*W, 3]
            rays_d = np.reshape(rays_d, [frame_num, -1, 3]) # [frame_num, N*H*W, 3]
            
            imgs = np.swapaxes(self.imgs, 0, 1)
            target_rgbs = np.reshape(imgs, [frame_num, -1, 3]) # [num_frames, N*H*W, 3]

            rays_o = rays_o.astype(np.float32)
            rays_d = rays_d.astype(np.float32)

            self.rays_o = rays_o
            self.rays_d = rays_d
            self.target_rgbs = target_rgbs

            print('rays_o', self.rays_o.shape) 
            print('rays_d', self.rays_d.shape) 
            print('target_rgbs', self.target_rgbs.shape)
            
            semantic_rays = np.swapaxes(self.semantic_imgs, 0, 1) # [frame_num, N, H, W]
            semantic_rays = np.reshape(semantic_rays, [frame_num, -1]) # [frame_num, N*H*W]
            self.semantic_rays = semantic_rays
    
    def load_imgs_poses(self, args):
        extrinsics = np.load(os.path.join(args.datadir, 'extrinsics.npy'), allow_pickle=True).item()

        cameras = sorted(glob(args.datadir + '/camera*/'), key=natural_keys)

        imgs = []
        poses = []
        semantic_imgs = []

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
                semantic_imgpaths = []
                # *3 since there are also semantic and depth images
                for path in sorted(glob(cam + '*.png'), key=natural_keys)[:3*self.num_frames]: 
                    if path.endswith('_semantic.png'):
                        semantic_imgpaths.append(path)
                    elif path.endswith('_depth.png'):
                        pass
                    else:
                        imgpaths.append(path)
                semantic_imgs.append([imageio.imread(imgpath) for imgpath in semantic_imgpaths])
            elif self.split == 'val':
                imgpaths = sorted(glob(cam + '*.png'), key=natural_keys)[:self.num_frames]

            imgs.append([imageio.imread(imgpath) for imgpath in imgpaths])
            poses.append(from_ue4_to_nerf(extrinsics[i]))
            

        imgs = (np.array(imgs) / 255.).astype(np.float32)[...,:3] # [view_num, frame_num, H, W, 3]
        poses = np.array(poses).astype(np.float32) # [view_num, 4, 4]
        self.view_num = len(poses)

        if self.split == 'train':
            semantic_imgs = np.array(semantic_imgs).astype(np.uint8)[...,0] # [view_num, frame_num, H, W]

        return imgs, poses, semantic_imgs

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
            frame = np.random.randint(low=0, high=self.num_frames)
            frames = np.array([frame])[:, None] # 1,1

            indices = np.random.choice(self.rays_o.shape[1], self.N_rand)
            rays_o = self.rays_o[frame, indices, ...]
            rays_d = self.rays_d[frame, indices, ...]
            target = self.target_rgbs[frame, indices, ...]        

            if not self.use_batching:
                raise NotImplementedError
    
        elif self.split == 'val':
            frame = np.random.randint(low=0, high=self.num_frames)
            frames = np.array([frame])[:, None] # 1,1

            view = np.random.randint(low=0, high=self.view_num)
            
            pose = self.poses[view, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            target = self.imgs[view, frame, ...]
            target = torch.Tensor(target)

            rays_o = torch.reshape(rays_o, [-1,3]) # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1,3]) # (H*W, 3)
            target = torch.reshape(target, [-1,3]) # (H*W, 3)
            
        return {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'target': target,
            'frames': frames
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

    #Used for debugging
    def get_gt_vehicle_poses(self, args):
        pose_files = sorted(glob(args.datadir + '/poses/*.npy'), key=natural_keys)
        poses = []
        
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
    