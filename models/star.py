from importlib.metadata import requires
import torch
import torch.nn as nn

from lietorch import SE3
from pytorch3d.transforms import se3_exp_map

from models.nerf import NeRF
from models.rendering import raw2outputs, raw2outputs_star, sample_pdf
from utils.io import device


class STaR(nn.Module):
    def __init__(self, num_frames, args, gt_poses=None):
        super(STaR, self).__init__()

        self.gt_poses = gt_poses
        self.chunk = args.chunk

        self.N_importance = args.N_importance

        self.static_coarse_nerf  = NeRF(D=args.netdepth,      W=args.netwidth,      args=args)
        self.static_fine_nerf    = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, args=args)
        
        self.dynamic_coarse_nerf = NeRF(D=args.netdepth,      W=args.netwidth,      args=args)
        self.dynamic_fine_nerf   = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, args=args)

        # At time 0, the pose is defined to be identity, therefore we don't optimize it, hence num_frames-1
        self.poses_ = nn.Parameter(torch.zeros((num_frames-1, 6)), requires_grad=True)
        #self.poses_grad = [] #TODO

    def get_poses(self):
        pose0 = torch.zeros((1, 6), requires_grad=False, device=self.poses_.device)
        poses = torch.cat((pose0, self.poses_), dim=0)
        return poses

    def get_poses_grad(self):
        with torch.no_grad():
            #for p in self.poses_grad:
            #    print(p.shape)
            #poses_grad = torch.cat(self.poses_grad, axis=0)
            poses_grad = self.poses_grad[0]
            self.poses_grad = []
        return poses_grad


    def get_nerf_params(self):
        return list(self.static_coarse_nerf.parameters()) + list(self.static_fine_nerf.parameters()) + \
               list(self.dynamic_coarse_nerf.parameters()) + list(self.dynamic_fine_nerf.parameters())

    def forward(self, pts, viewdirs, z_vals, rays_d, frames=None, is_coarse=True, object_pose=None):
        rgb_map_chunks, disp_map_chunks, acc_map_chunks, weights_chunks, depth_map_chunks = [], [], [], [], []
        if frames is not None or object_pose is not None:
            entropy = 0
            rgb_map_static_chunks, rgb_map_dynamic_chunks = [], []
        if frames is not None and object_pose is None:
            transformed_pts_chunks = []

        for i in range(0, pts.shape[0], self.chunk):
            end_i = min(pts.shape[0], i + self.chunk)
            pts_chunk = pts[i:end_i, ...]
            viewdirs_chunk = viewdirs[i:end_i, ...]
            z_vals_chunk = z_vals[i:end_i, ...]
            rays_d_chunk = rays_d[i:end_i, ...]
            
            if frames is not None:
                frames_chunk = frames[i:end_i, ...]
            else:
                frames_chunk = None

            chunk_result = self.forward_chunk(pts_chunk, viewdirs_chunk, z_vals_chunk, rays_d_chunk, frames_chunk, is_coarse, object_pose=object_pose)

            if frames is None and object_pose is None:
                rgb_map, disp_map, acc_map, weights, depth_map = chunk_result # TODO adapt for entropy
            else:
                if frames is not None:
                    rgb_map, disp_map, acc_map, weights, depth_map, entropy_, rgb_map_static, rgb_map_dynamic, transformed_pts = chunk_result
                    transformed_pts_chunks.append(transformed_pts)
                else:
                    rgb_map, disp_map, acc_map, weights, depth_map, entropy_, rgb_map_static, rgb_map_dynamic = chunk_result
                entropy += entropy_
                rgb_map_static_chunks.append(rgb_map_static)
                rgb_map_dynamic_chunks.append(rgb_map_dynamic)
    
            
            rgb_map_chunks.append(rgb_map)
            disp_map_chunks.append(disp_map)
            acc_map_chunks.append(acc_map)
            weights_chunks.append(weights)
            depth_map_chunks.append(depth_map)
        
        rgb_map = torch.cat(rgb_map_chunks, dim=0)
        disp_map = torch.cat(disp_map_chunks, dim=0)
        acc_map = torch.cat(acc_map_chunks, dim=0)
        weights = torch.cat(weights_chunks, dim=0)
        depth_map = torch.cat(depth_map_chunks, dim=0)
        
        if frames is None and object_pose is None:
            return rgb_map, disp_map, acc_map, weights, depth_map
        else:
            if frames is not None:
                transformed_pts = torch.cat(transformed_pts_chunks, dim=0)
                #transformed_pts.retain_grad()
                #transformed_pts.register_hook(print)
            rgb_map_static = torch.cat(rgb_map_static_chunks, dim=0)
            rgb_map_dynamic = torch.cat(rgb_map_dynamic_chunks, dim=0)
            if frames is not None:
                return rgb_map, disp_map, acc_map, weights, depth_map, entropy, rgb_map_static, rgb_map_dynamic, transformed_pts
            else:
                return rgb_map, disp_map, acc_map, weights, depth_map, entropy, rgb_map_static, rgb_map_dynamic

    def forward_chunk(self, pts, viewdirs, z_vals, rays_d, frames=None, is_coarse=True, object_pose=None):
        """STaR's forward
        Args:
            pts: [N_rays, N_samples, 3]. Points sampled according to stratified sampling.
            viewdirs: [N_rays, 3]. View directions of rays.
            z_vals: [N_rays, N_samples]. Integration time.
            rays_d: [N_rays, 3]. Unnormalized directions of rays.
            frames: [N_rays,1]. Time steps of the rays. None during appearance init.
            is_coarse: True if render using coarse models, False if render using fine models
            object_pose: [4, 4]. Pose of the dynamic object, same for all rays, used in testing.
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
        if frames is None and object_pose is None:
            return raw2outputs(raw_alpha_static, raw_rgb_static, z_vals, rays_d, 
                static_model.raw_noise_std if self.training else 0, static_model.white_bkgd, ret_entropy=False) # TODO experiment with entropy during appearance init

        # Transform points and viewdirs according to learned poses or ground-truth poses if given for debugging
        if object_pose is not None:
            pose_matrices = object_pose[None, None, ...].expand((N_rays, 1, 4, 4)) # [N_rays, 1, 4, 4]
        elif self.gt_poses is not None:
            pose_matrices = self.gt_poses[frames,...] # [N_rays, 1, 4, 4]
        else:
            # pose = self.get_poses()[frames][:,0,:] # [N_rays, 3]
            # pose_matrices_ = se3_exp_map(pose)
            # pose_matrices = torch.zeros((N_rays, 4, 4), device=pose.device)
            # pose_matrices[:, :3, :3] = pose_matrices_[:, :3, :3]
            # pose_matrices[:, :3, 3] = pose_matrices_[:, 3, :3]
            # pose_matrices[:, 3, 3] = 1.
            # pose_matrices = pose_matrices[:, None, ...]
            pose = self.get_poses()[frames,...]
            pose_matrices = SE3.exp(pose).matrix() # [N_rays, 1, 4, 4]
            


        pose_matrices_pts = pose_matrices.expand((N_rays, N_samples, 4, 4)) # [N_rays, N_samples, 4, 4]
        pose_matrices_pts_flat = pose_matrices_pts.reshape((N_rays*N_samples, 4, 4)) # [N_rays*N_samples, 4, 4]
        
        pts_homog = torch.cat([pts, torch.ones((N_rays, N_samples, 1), device=device)], dim=-1) # [N_rays, N_samples, 4]
        pts_homog_flat = pts_homog.reshape((-1, 4)) # [N_rays*N_samples, 4]
        
        pts_dynamic_homog_flat = torch.einsum('nab,nb->na', pose_matrices_pts_flat, pts_homog_flat)
        pts_dynamic_homog = pts_dynamic_homog_flat.reshape((N_rays, N_samples, 4))
        pts_dynamic = pts_dynamic_homog[..., :3] # [N_rays, N_samples, 3]
        #pts_dynamic.retain_grad()
        #print(pts_dynamic.grad)
        #pts_dynamic.register_hook(print)
        # if self.training:
        #     pts_dynamic.register_hook(lambda grad: self.poses_grad.append(grad))

        viewdirs_dynamic = torch.einsum('nab,nb->na', pose_matrices[:, 0, :3, :3], viewdirs)

        raw_alpha_dynamic, raw_rgb_dynamic = dynamic_model(pts_dynamic, viewdirs_dynamic)

        result = raw2outputs_star(raw_alpha_static, raw_rgb_static, raw_alpha_dynamic, raw_rgb_dynamic, z_vals, rays_d, 
            static_model.raw_noise_std if (self.training and frames is None) else 0, #From the paper: "we add small Gaussian noise to the density outputs during appearance initialization but turn it off during online training."
            static_model.white_bkgd, ret_entropy=True)

        if frames is not None and object_pose is None:
            result += (pts_dynamic,)

        return result


        