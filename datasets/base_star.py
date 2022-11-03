import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from models.rendering import get_rays, ndc_rays, get_rays_np


class BaseStarDataset(Dataset):
    def __init__(self, args):
        self.render_test = args.render_test
        self.chunk = args.chunk
        self.perturb = args.perturb if self.split == 'train' else 0
        self.ndc = not (args.dataset_type != 'llff' or args.no_ndc)
        self.lindisp = args.lindisp
        self.N_samples = args.N_samples
        self.N_rand = args.N_rand
        self.use_viewdirs = args.use_viewdirs
        self.use_batching = not args.no_batching 
        

        if self.split in ['test','render_video'] and args.render_factor!=0:
            # Render downsampled for speed
            self.H = self.H//args.render_factor
            self.W = self.W//args.render_factor
            self.focal = self.focal/args.render_factor


        if self.split == 'train_appearance' and self.use_batching:
            rays = np.stack([get_rays_np(self.H, self.W, self.K, p) for p in self.poses[:,:3,:4]], 0) # [N,ro+rd,H,W,3]
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

        elif self.split == 'train_online' and self.use_batching:
            rays = np.stack([get_rays_np(self.H, self.W, self.K, p) for p in self.poses[:,:3,:4]], 0) # [N,ro+rd,H,W,3]
            frame_num = self.frames.shape[1]
            rays = np.expand_dims(rays, axis=1).repeat(frame_num, axis=1) # [N,frame_num,2,H,W,3] 
            rays_o = rays[:,:,0,:,:,:] # [N,frame_num,H,W,3]
            rays_d = rays[:,:,1,:,:,:] # [N,frame_num,H,W,3]

            rays_o = np.reshape(rays_o, [-1, 3]) # [N*frame_num*H*W,3]
            rays_d = np.reshape(rays_d, [-1, 3]) # [N*frame_num*H*W,3]
            target_rgbs = np.reshape(self.imgs, [-1, self.imgs.shape[-1]]) # [N*num_frames*H*W,3]

            rays_o = rays_o.astype(np.float32)
            rays_d = rays_d.astype(np.float32)

            self.frames = self.frames[...,None,None].repeat(self.H, axis=-2).repeat(self.W, axis=-1).reshape([-1,1]) # [N*num_frames*H*W,1]

            self.rays_o = rays_o
            self.rays_d = rays_d
            self.target_rgbs = target_rgbs

            print('rays_o', self.rays_o.shape) 
            print('rays_d', self.rays_d.shape) 
            print('target_rgbs', self.target_rgbs.shape)


    def __len__(self):
        if self.split == 'train_appearance':
            if self.use_batching:
                return len(self.rays_o)
            else:
                return len(self.imgs)
        elif self.split == 'val_appearance':
            return len(self.imgs)
        elif self.split == 'train_online':
            if self.use_batching:
                return len(self.rays_o)
            else:
                return self.imgs.shape[0] * self.num_frames
        elif self.split == 'val_online':
            return self.imgs.shape[0] * self.num_frames
        elif self.split == 'test': # TODO
            return len(self.imgs)
        elif self.split == 'render_video':
            return len(self.object_poses)
        elif self.split == 'train_render':
            return self.imgs.shape[0] * self.num_frames
        else:
            raise ValueError("invalid dataset split")


    def __getitem__(self, idx):
        if self.split == 'train_appearance':
            frames = None
            if self.use_batching:
                rays_o = self.rays_o[idx, ...]
                rays_d = self.rays_d[idx, ...]
                target = self.target_rgbs[idx, ...]
            else:
                # Random from one image
                target = self.imgs[idx]
                target = torch.Tensor(target) 
                pose = self.poses[idx, :3, :4]
                rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                coords = torch.stack(torch.meshgrid(
                    torch.linspace(0, self.H-1, self.H), 
                    torch.linspace(0, self.W-1, self.W), indexing='ij'), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = torch.randperm(coords.shape[0])[:self.N_rand]
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                target = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)                    

            """
            NOTE: precrop is not implemented! 
            see: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf.py#L738
            """
        elif self.split == 'val_appearance' or self.split == 'test': # TODO test?
            frames = None
            # Load all of the rays of one image
            target = self.imgs[idx] 
            target = torch.Tensor(target) 
            pose = self.poses[idx, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            rays_o = torch.reshape(rays_o, [-1,3]) # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1,3]) # (H*W, 3)
            target = torch.reshape(target, [-1,3]) # (H*W, 3)
        elif self.split == 'train_online':
            if self.use_batching:
                rays_o = self.rays_o[idx, ...]
                rays_d = self.rays_d[idx, ...]
                target = self.target_rgbs[idx, ...]
                frames = self.frames[idx, ...]
            else:
                view_idx = idx // self.num_frames
                frame_idx = idx % self.num_frames
                target = self.imgs[view_idx, frame_idx, ...]
                target = torch.Tensor(target) 
                pose = self.poses[view_idx, :3, :4]
                rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                coords = torch.stack(torch.meshgrid(
                    torch.linspace(0, self.H-1, self.H), 
                    torch.linspace(0, self.W-1, self.W), indexing='ij'), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = torch.randperm(coords.shape[0])[:self.N_rand]
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                target = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)                    
                frames = self.frames[view_idx, frame_idx, ...][None,...].repeat(self.N_rand, axis=0)[..., None] # (N_rand, 1)
                #print('frames shape', frames.shape)
        elif self.split == 'val_online':
            target = self.imgs.reshape((-1, self.H, self.W, 3))[idx, ...] # (N*num_frames, H, W, 3) => (H, W, 3)
            target = torch.Tensor(target) 
            pose = self.poses[idx // self.num_frames, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            rays_o = torch.reshape(rays_o, [-1,3]) # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1,3]) # (H*W, 3)
            target = torch.reshape(target, [-1,3]) # (H*W, 3)
            frames = self.frames.reshape((-1,1))[idx,...].repeat(self.H * self.W, axis=0)[..., None] # (H*W, 1)
            #print('frames shape', frames.shape)
        elif self.split == 'train_render':
            target = self.imgs.reshape((-1, self.H, self.W, 3))[idx, ...] # (N*num_frames, H, W, 3) => (H, W, 3)
            target = torch.Tensor(target) 
            pose = self.poses[idx // self.num_frames, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            rays_o = torch.reshape(rays_o, [-1,3]) # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1,3]) # (H*W, 3)
            target = torch.reshape(target, [-1,3]) # (H*W, 3)
            frames = self.frames.reshape((-1,1))[idx,...].repeat(self.H * self.W, axis=0)[..., None] # (H*W, 1)
        elif self.split == 'render_video':
            pose = self.poses[0, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))
            rays_o = torch.reshape(rays_o, [-1,3]) # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1,3]) # (H*W, 3)
            target = None
            frames = None
            object_pose = self.object_poses[idx, ...]
        
        result = (rays_o, rays_d)
        if target is not None:
            result += (target,)
        if frames is not None:
            result += (frames,)        
        if self.split == 'render_video':
            result += (object_pose,)
        return result

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


    
    def collate_sample_pts_and_viewdirs(self, batch):
        """
        Used in the dataloader as collate_fn to generate point samples along the rays and also view directions
        Create sampled points and view directions using ray origins and directions
        """
        
        if self.split == 'render_video':
            rays_o, rays_d, object_pose = default_collate(batch)
            target = None
            frames = None
        elif self.split == 'train_online' or self.split == 'val_online' or 'train_render':
            rays_o, rays_d, target, frames = default_collate(batch)
        else:
            rays_o, rays_d, target = default_collate(batch)
            frames = None
        

        if (self.split == 'train_appearance' and not self.use_batching) or (self.split == 'train_online' and not self.use_batching) or self.split in ['val_appearance', 'val_online', 'test', 'render_video', 'train_render']: # TODO what else?
            # Batch size is 1 in this case
            rays_o = rays_o[0]
            rays_d = rays_d[0]
            if target is not None:
                target = target[0]
            if frames is not None:
                frames = frames[0]
            if self.split == 'render_video':
                object_pose = object_pose[0]
                

        viewdirs = None
        if self.use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True) # [N_rays, 3]
            #viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        if self.ndc:
            # for forward facing scenes TODO remove ndc
            rays_o, rays_d = ndc_rays(self.H, self.W, self.K[0][0], 1., rays_o, rays_d)

        near, far = self.near * torch.ones_like(rays_d[...,:1]), self.far * torch.ones_like(rays_d[...,:1])
        
        t_vals = torch.linspace(0., 1., steps=self.N_samples)
        if not self.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        N_rays = rays_o.shape[0]
        z_vals = z_vals.expand([N_rays, self.N_samples])

        if self.split == 'train' and self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand        

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        if self.split == 'render_video':
            return pts, viewdirs, z_vals, rays_o, rays_d, object_pose
        else:
            return pts, viewdirs, z_vals, rays_o, rays_d, target, frames
    
    @staticmethod
    def validate_split(split):
        splits = ['train_appearance', 'val_appearance', 'train_online', 'val_online', 'test', 'render_video', 'train_render'] 
        assert split in splits, "Dataset split should be one of " + ", ".join(splits)

    
    @staticmethod
    def move_batch_to_device(batch, device):
        for i in range(len(batch)):
            if batch[i] is not None:
                batch[i] = batch[i].to(device, non_blocking=True)  
        

        