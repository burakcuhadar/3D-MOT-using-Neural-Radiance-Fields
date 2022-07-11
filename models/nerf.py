import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.embedder import get_embedder
from models.rendering import raw2outputs



# Model
class NeRF(nn.Module):
    def __init__(self, D, W, args):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W

        # Create embedders for positional encoding
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        input_ch_views = 0
        self.embeddirs_fn = None
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
        
        # Used only when not using viewdirs TODO is 5th dim ever used?
        output_ch = 5 if args.N_importance > 0 else 4
    
        skips = [4]

        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = args.use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + \
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release:
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if args.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1) 
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        
        self.netchunk = args.netchunk
        self.raw_noise_std = args.raw_noise_std # should only be used for training
        self.white_bkgd = args.white_bkgd

    def forward(self, pts, viewdirs, z_vals, rays_d):
        """
        1. Embed the input points and view directions if given
        2. Forward pass embedded inputs through MLP to get rgb and density
        3. Estimate expected color integral
        """
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]]) # [N_rays*N_samples, 3]
        if viewdirs is not None:
            input_dirs = viewdirs[:,None].expand(pts.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) # [N_rays*N_samples, 3]

        raw_rgb_chunks, raw_alpha_chunks = [], []
        # Forward by chunks to avoid OOM
        for i in range(0, pts_flat.shape[0], self.netchunk):    
            end_i = min(pts_flat.shape[0], i + self.netchunk)
            pts_chunk = pts_flat[i:end_i, :]
            # 1. Embed the input points and view directions if given
            embedded_pts = self.embed_fn(pts_chunk)

            if viewdirs is not None:
                input_dirs_chunk = input_dirs_flat[i:end_i, :]
                embedded_dirs = self.embeddirs_fn(input_dirs_chunk)

            # 2. Forward pass embedded inputs through MLP to get rgb and density
            h = embedded_pts
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([embedded_pts, h], -1)

            if self.use_viewdirs:
                raw_alpha = self.alpha_linear(h)
                feature = self.feature_linear(h)
                h = torch.cat([feature, embedded_dirs], -1)
            
                for i, l in enumerate(self.views_linears):
                    h = self.views_linears[i](h)
                    h = F.relu(h)

                raw_rgb = self.rgb_linear(h)

            else:
                output = self.output_linear(h)
                raw_rgb = output[:,:3]
                raw_alpha = output[:,3]
            
            raw_rgb_chunks.append(raw_rgb)
            raw_alpha_chunks.append(raw_alpha)
        
        raw_rgb = torch.cat(raw_rgb_chunks, axis=0) # N_rays*N_samples, 3
        raw_alpha = torch.cat(raw_alpha_chunks, axis=0) # N_rays*N_samples
        raw_rgb = torch.reshape(raw_rgb, pts.shape) # N_rays, N_samples, 3
        raw_alpha = torch.reshape(raw_alpha, pts.shape[:-1]) # N_rays, N_samples

        # 3. Estimate expected color integral and return extras
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_alpha, raw_rgb, z_vals, rays_d, 
            self.raw_noise_std if self.training else 0, self.white_bkgd)
        
        return rgb_map, disp_map, acc_map, weights, depth_map
