import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from models.rendering import get_rays, ndc_rays, get_rays_np



class BaseDataset(Dataset):
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


        if self.split == 'train' and self.use_batching:
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




    def __len__(self):
        if self.split == 'train':
            if self.use_batching:
                return len(self.rays_o)
            else:
                return len(self.imgs)
        elif self.split == 'val':
            return len(self.imgs)
        elif self.split == 'test':
            return len(self.imgs)
        elif self.split == 'render_video':
            return len(self.poses)
        else:
            raise ValueError("invalid dataset split")

    def __getitem__(self, idx):
        if self.split == 'train':
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
        elif self.split == 'val' or self.split == 'test':
            # Load all of the rays of one image
            target = self.imgs[idx] 
            target = torch.Tensor(target) 
            pose = self.poses[idx, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            rays_o = torch.reshape(rays_o, [-1,3]) # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1,3]) # (H*W, 3)
            target = torch.reshape(target, [-1,3]) # (H*W, 3)
        elif self.split == 'render_video':
            pose = self.poses[idx, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))
            rays_o = torch.reshape(rays_o, [-1,3]) # (H*W, 3)
            rays_d = torch.reshape(rays_d, [-1,3]) # (H*W, 3)
            if self.render_test:
                target = self.imgs[idx]
                target = torch.Tensor(target) 
                target = torch.reshape(target, [-1,3]) # (H*W, 3)
            else:
                return rays_o, rays_d
            
        
        return rays_o, rays_d, target


    

    def collate_sample_pts_and_viewdirs(self, batch):
        """
        Used in the dataloader as collate_fn to generate point samples along the rays and also view directions
        Create sampled points and view directions using ray origins and directions
        """
        
        if self.split == 'render_video' and not self.render_test:
            rays_o, rays_d = default_collate(batch)
            target = None
        else:
            rays_o, rays_d, target = default_collate(batch)
        

        if (self.split == 'train' and not self.use_batching) or self.split in ['val', 'test', 'render_video']:
            # Batch size is 1 in this case
            rays_o = rays_o[0]
            rays_d = rays_d[0]
            if target is not None:
                target = target[0]
                

        viewdirs = None
        if self.use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True) # [N_rays, 3]
            #viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        if self.ndc:
            # for forward facing scenes
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


        return pts, viewdirs, z_vals, rays_o, rays_d, target
    
    

    '''
    def collate_sample_pts_and_viewdirs(self, batch):
        """
        Used in the dataloader as collate_fn to generate point samples along the rays and also view directions
        Create sampled points and view directions using ray origins and directions
        """
        if self.split == 'render_video' and not self.render_test:
            rays_o, rays_d = default_collate(batch)
            target = None
        else:
            rays_o, rays_d, target = default_collate(batch)
        
        if (self.split == 'train' and not self.use_batching) or self.split in ['val', 'test', 'render_video']:
            # Batch size is 1 in this case
            rays_o = rays_o[0]
            rays_d = rays_d[0]
            if target is not None:
                target = target[0]
                
        viewdirs = None
        if self.use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True) # [N_rays, 3]
            #viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        if self.ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(self.H, self.W, self.K[0][0], 1., rays_o, rays_d)

        near, far = self.near * torch.ones_like(rays_d[...,:1]), self.far * torch.ones_like(rays_d[...,:1])
        

        pts_chunks, z_vals_chunks = [], []
        for i in range(0, rays_o.shape[0], self.chunk):
            end_i = min(rays_o.shape[0], i + self.chunk)
            N_rays_chunk = end_i - i
            near_chunk = near[i:end_i, :]
            far_chunk = far[i:end_i, :]
            rays_o_chunk = rays_o[i:end_i, :]
            rays_d_chunk = rays_d[i:end_i, :]

            t_vals = torch.linspace(0., 1., steps=self.N_samples)
            if not self.lindisp:
                z_vals_chunk = near_chunk * (1.-t_vals) + far_chunk * (t_vals)
            else:
                z_vals_chunk = 1./(1./near_chunk * (1.-t_vals) + 1./far_chunk * (t_vals))

            z_vals_chunk = z_vals_chunk.expand([N_rays_chunk, self.N_samples])

            if self.split == 'train' and self.perturb > 0.:
                # get intervals between samples
                mids = .5 * (z_vals_chunk[...,1:] + z_vals_chunk[...,:-1])    
                upper = torch.cat([mids, z_vals_chunk[...,-1:]], -1)
                lower = torch.cat([z_vals_chunk[...,:1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals_chunk.shape)
                z_vals_chunk = lower + (upper - lower) * t_rand        

            pts_chunk = rays_o_chunk[...,None,:] + rays_d_chunk[...,None,:] * z_vals_chunk[...,:,None] # N_rays_chunk, N_sample, 3

            pts_chunks.append(pts_chunk)
            z_vals_chunks.append(z_vals_chunk)

        pts = torch.cat(pts_chunks, axis=0) # N_rays, N_samples, 3
        z_vals = torch.cat(z_vals_chunks, axis=0) # N_rays, N_samples

        return pts, viewdirs, z_vals, rays_o, rays_d, target
        '''



            
    @staticmethod
    def move_batch_to_device(batch, device):
        for i in range(len(batch)):
            if batch[i] is not None:
                batch[i] = batch[i].to(device)
        

        