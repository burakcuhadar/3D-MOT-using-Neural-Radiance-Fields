from importlib.metadata import requires
import torch
import torch.nn as nn

from lietorch import SE3

from models.nerf import NeRF
from models.rendering import raw2outputs, raw2outputs_star, sample_pdf
from utils.io import device


class STaR(nn.Module):
    def __init__(self, num_frames, args, gt_poses=None):
        super(STaR, self).__init__()

        self.gt_poses = gt_poses

        self.N_importance = args.N_importance

        self.static_coarse_nerf  = NeRF(D=args.netdepth,      W=args.netwidth,      args=args)
        self.static_fine_nerf    = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, args=args)
        
        self.dynamic_coarse_nerf = NeRF(D=args.netdepth,      W=args.netwidth,      args=args)
        self.dynamic_fine_nerf   = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, args=args)

        # At time 0, the pose is defined to be identity, therefore we don't optimize it, hence num_frames-1
        self.poses_ = nn.Parameter(torch.zeros((num_frames-1, 6)), requires_grad=True)

    def get_poses(self):
        pose0 = torch.zeros((1, 6), requires_grad=False, device=self.poses_.device)
        poses = torch.cat((pose0, self.poses_), dim=0)
        return poses


    def get_nerf_params(self):
        return list(self.static_coarse_nerf.parameters()) + list(self.static_fine_nerf.parameters()) + \
               list(self.dynamic_coarse_nerf.parameters()) + list(self.dynamic_fine_nerf.parameters())

    def forward(self, pts, viewdirs, z_vals, rays_d, frames=None, is_coarse=True):
        """STaR's forward
        Args:
            pts: [N_rays, N_samples, 3]. Points sampled according to stratified sampling.
            viewdirs: [N_rays, 3]. View directions of rays.
            z_vals: [N_rays, N_samples]. Integration time.
            rays_d: [N_rays, 3]. Unnormalized directions of rays.
            frames: [N_rays,1]. Time steps of the rays. None during appearance init.
            is_coarse: True if render using coarse models, False if render using fine models
        Returns:
            TODO
        """
        N_rays = pts.shape[0]
        N_samples = pts.shape[1]

        if is_coarse:
            static_model = self.static_coarse_nerf
            dynamic_model = self.dynamic_coarse_nerf
        else:
            static_model = self.static_fine_nerf
            dynamic_model = self.dynamic_fine_nerf

        raw_alpha_static, raw_rgb_static = static_model(pts, viewdirs)

        # During appearance initialization only static part is trained
        if frames is None:
            return raw2outputs(raw_alpha_static, raw_rgb_static, z_vals, rays_d, 
                static_model.raw_noise_std if self.training else 0, static_model.white_bkgd, ret_entropy=False) # TODO experiment with entropy during appearance init

        # Transform points and viewdirs according to learned poses or ground-truth poses if given for debugging
        if self.gt_poses is not None:
            pose_matrices = self.gt_poses[frames,...] # [N_rays, 1, 4, 4]
        else:
            pose = self.get_poses()[frames,...]
            pose_matrices = SE3.exp(pose).matrix() # [N_rays, 1, 4, 4]
        
        pose_matrices_pts = pose_matrices.expand((N_rays, N_samples, 4, 4)) # [N_rays, N_samples, 4, 4]
        pose_matrices_pts_flat = pose_matrices_pts.reshape((N_rays*N_samples, 4, 4)) # [N_rays*N_samples, 4, 4]
        
        pts_homog = torch.cat([pts, torch.ones((N_rays, N_samples, 1), device=device)], dim=-1) # [N_rays, N_samples, 4]
        pts_homog_flat = pts_homog.reshape((-1, 4)) # [N_rays*N_samples, 4]
        
        pts_dynamic_homog_flat = torch.einsum('nab,nb->na', pose_matrices_pts_flat, pts_homog_flat)
        pts_dynamic_homog = pts_dynamic_homog_flat.reshape((N_rays, N_samples, 4))
        pts_dynamic = pts_dynamic_homog[..., :3] # [N_rays, N_samples, 3]
        
        viewdirs_dynamic = torch.einsum('nab,nb->na', pose_matrices[:, 0, :3, :3], viewdirs)

        raw_alpha_dynamic, raw_rgb_dynamic = dynamic_model(pts_dynamic, viewdirs_dynamic)

        return raw2outputs_star(raw_alpha_static, raw_rgb_static, raw_alpha_dynamic, raw_rgb_dynamic, z_vals, rays_d, 
            static_model.raw_noise_std if (self.training and frames is None) else 0, #From the paper: "we add small Gaussian noise to the density outputs during appearance initialization but turn it off during online training."
            static_model.white_bkgd, ret_entropy=True)

        
        