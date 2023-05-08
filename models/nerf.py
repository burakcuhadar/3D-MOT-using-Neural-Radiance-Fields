import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tinycudann as tcnn

from models.embedder import get_embedder
from models.rendering import raw2outputs
from models.resnet import ResnetFC


# Model
class NeRF(nn.Module):
    def __init__(self, D, W, args):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W

        # Create embedders for positional encoding
        embedder, input_ch = get_embedder(args.multires, args.end_barf, args.i_embed)

        input_ch_views = 0
        self.embedder_dirs = None
        if args.use_viewdirs:
            embedder_dirs, input_ch_views = get_embedder(args.multires_views, args.end_barf, args.i_embed)
        
        # Used only when not using viewdirs TODO is 5th dim ever used?
        output_ch = 5 if args.N_importance > 0 else 4
    
        
        self.embedder = embedder
        self.embedder_dirs = embedder_dirs
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = args.use_viewdirs
        
        self.pts_net = ResnetFC(input_ch, d_out=W, n_blocks=D//2, d_hidden=W)
        '''self.pts_net = tcnn.Network(
            n_input_dims=input_ch, n_output_dims=W,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": W,
                "n_hidden_layers": D,
            }
        )'''       
        
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

        # Weight Initialization
        for layer in self.views_linears:
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.kaiming_normal_(self.alpha_linear.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.alpha_linear.bias)
        torch.nn.init.xavier_uniform_(self.rgb_linear.weight)

    def forward(self, pts, viewdirs, z_vals=None, rays_d=None, step=None):
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
            embedded_pts = self.embedder(pts_chunk, step=step)

            if viewdirs is not None:
                input_dirs_chunk = input_dirs_flat[i:end_i, :]
                embedded_dirs = self.embedder_dirs(input_dirs_chunk, step=step)

            # 2. Forward pass embedded inputs through MLP to get rgb and density
            h = embedded_pts
            h = self.pts_net(h)
                
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

        if z_vals is None:
            return raw_alpha, raw_rgb

        # 3. Estimate expected color integral and return extras
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_alpha, raw_rgb, z_vals, rays_d, 
            self.raw_noise_std if self.training else 0, self.white_bkgd)
        
        return rgb_map, disp_map, acc_map, weights, depth_map

    
