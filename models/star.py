from collections import defaultdict

import torch
import torch.nn as nn

from lietorch import SO3, SE3

# from pytorch3d.transforms import se3_exp_map

from models.nerf import NeRF
from models.rendering import raw2outputs, raw2outputs_star
from utils.io import device


class STaR(nn.Module):
    def __init__(self, num_frames, args):
        super(STaR, self).__init__()

        self.chunk = args.chunk

        self.N_importance = args.N_importance

        self.static_coarse_nerf = NeRF(D=args.netdepth, W=args.netwidth, args=args)
        self.static_fine_nerf = NeRF(
            D=args.netdepth_fine, W=args.netwidth_fine, args=args
        )

        self.dynamic_coarse_nerf = NeRF(D=args.netdepth, W=args.netwidth, args=args)
        self.dynamic_fine_nerf = NeRF(
            D=args.netdepth_fine, W=args.netwidth_fine, args=args
        )

    def get_nerf_params(self):
        return (
            list(self.static_coarse_nerf.parameters())
            + list(self.static_fine_nerf.parameters())
            + list(self.dynamic_coarse_nerf.parameters())
            + list(self.dynamic_fine_nerf.parameters())
        )

    def forward(
        self,
        pts,
        viewdirs,
        z_vals,
        rays_d,
        pose=None,
        is_coarse=True,
        object_pose=None,
        step=None,
    ):
        result = defaultdict(list)

        for i in range(0, pts.shape[0], self.chunk):
            end_i = min(pts.shape[0], i + self.chunk)
            pts_chunk = pts[i:end_i, ...]
            viewdirs_chunk = viewdirs[i:end_i, ...]
            z_vals_chunk = z_vals[i:end_i, ...]
            rays_d_chunk = rays_d[i:end_i, ...]

            chunk_result = self.forward_chunk(
                pts_chunk,
                viewdirs_chunk,
                z_vals_chunk,
                rays_d_chunk,
                pose,
                is_coarse,
                object_pose=object_pose,
                step=step,
            )

            for k, v in chunk_result.items():
                result[k] += [v]

        for k, v in result.items():
            if len(result[k][0].shape) == 0:
                result[k] = sum(v)
            else:
                result[k] = torch.cat(v, 0)

        return result

    '''
    def forward_chunk(
        self,
        pts,
        viewdirs,
        z_vals,
        rays_d,
        pose=None,
        is_coarse=True,
        object_pose=None,
        step=None,
    ):
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
        """
        N_rays = pts.shape[0]
        N_samples = pts.shape[1]

        if is_coarse:
            static_model = self.static_coarse_nerf
            dynamic_model = self.dynamic_coarse_nerf
        else:
            static_model = self.static_fine_nerf
            dynamic_model = self.dynamic_fine_nerf

        raw_alpha_static, raw_rgb_static = static_model(pts, viewdirs, step=None)

        # During appearance initialization only static part is trained
        if pose is None and object_pose is None:
            return raw2outputs(
                raw_alpha_static,
                raw_rgb_static,
                z_vals,
                rays_d,
                static_model.raw_noise_std if self.training else 0,
                static_model.white_bkgd,
                ret_entropy=False,
            )

        # Transform points and viewdirs according to learned poses or ground-truth poses if given for
        # debugging
        if object_pose is not None:
            pose_matrices = object_pose[None, None, ...].expand(
                (N_rays, 1, 4, 4)
            )  # [N_rays, 1, 4, 4]
        elif len(pose.shape) == 1:
            # pose = self.get_poses()[frames][:,0,:] # [N_rays, 3]
            # pose_matrices_ = se3_exp_map(pose)
            # pose_matrices = torch.zeros((N_rays, 4, 4), device=pose.device)
            # pose_matrices[:, :3, :3] = pose_matrices_[:, :3, :3]
            # pose_matrices[:, :3, 3] = pose_matrices_[:, 3, :3]
            # pose_matrices[:, 3, 3] = 1.
            # pose_matrices = pose_matrices[:, None, ...]
            # pose = self.get_poses()[frames[0]][0]
            trans = pose[:3]
            # rot = pose[:, 0, 3:]
            # pose_matrices = SE3.exp(pose).matrix() # [N_rays, 1, 4, 4]
            pose_matrices = torch.eye(4, device=pts.device, dtype=torch.float32)[
                None, None, ...
            ].repeat_interleave(N_rays, dim=0)
            # pose_matrices[:, 0, :3, :3] = SO3.exp(rot).matrix()[:, :3, :3]
            pose_matrices[:, 0, :3, 3] = trans
        # Gt poses matrices are provided
        elif len(pose.shape) == 2:
            pose_matrices = pose[None, None, ...].repeat_interleave(N_rays, dim=0)
        else:
            raise NotImplementedError

        # [N_rays, N_samples, 4, 4]
        pose_matrices_pts = pose_matrices.expand((N_rays, N_samples, 4, 4))
        # [N_rays*N_samples, 4, 4]
        pose_matrices_pts_flat = pose_matrices_pts.reshape((N_rays * N_samples, 4, 4))

        # [N_rays, N_samples, 4]
        pts_homog = torch.cat(
            [pts, torch.ones((N_rays, N_samples, 1), device=device)], dim=-1
        )
        pts_homog_flat = pts_homog.reshape((-1, 4))  # [N_rays*N_samples, 4]

        pts_dynamic_homog_flat = torch.einsum(
            "nab,nb->na", pose_matrices_pts_flat, pts_homog_flat
        )
        pts_dynamic_homog = pts_dynamic_homog_flat.reshape((N_rays, N_samples, 4))
        pts_dynamic = pts_dynamic_homog[..., :3]  # [N_rays, N_samples, 3]

        viewdirs_dynamic = torch.einsum(
            "nab,nb->na", pose_matrices[:, 0, :3, :3], viewdirs
        )

        raw_alpha_dynamic, raw_rgb_dynamic = dynamic_model(
            pts_dynamic, viewdirs_dynamic, step=step
        )

        result = raw2outputs_star(
            raw_alpha_static,
            raw_rgb_static,
            raw_alpha_dynamic,
            raw_rgb_dynamic,
            z_vals,
            rays_d,
            # From the paper: "we add small Gaussian noise to the density outputs during
            # appearance initialization but turn it off during online training."
            0,
            static_model.white_bkgd,
        )

        return result
    '''

    def forward_chunk(
        self,
        pts,
        viewdirs,
        z_vals,
        rays_d,
        pose=None,
        is_coarse=True,
        object_pose=None,
        step=None,
    ):
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
        """
        N_rays = pts.shape[0]
        N_samples = pts.shape[1]

        if is_coarse:
            static_model = self.static_coarse_nerf
            dynamic_model = self.dynamic_coarse_nerf
        else:
            static_model = self.static_fine_nerf
            dynamic_model = self.dynamic_fine_nerf

        raw_alpha_static, raw_rgb_static = static_model(pts, viewdirs, step=None)

        # During appearance initialization only static part is trained
        if pose is None and object_pose is None:
            return raw2outputs(
                raw_alpha_static,
                raw_rgb_static,
                z_vals,
                rays_d,
                static_model.raw_noise_std if self.training else 0,
                static_model.white_bkgd,
                ret_entropy=False,
            )

        if object_pose is not None:
            raise NotImplementedError  # implement for testing with different ojbect poses
        elif len(pose.shape) == 2:
            pose_matrix = pose
        elif len(pose.shape) == 1:
            rot = pose[3:]
            pose_matrix = torch.eye(4, device=pts.device, dtype=torch.float32)
            pose_matrix[:3, :3] = SO3.exp(rot).matrix()[:3, :3]
            pose_matrix[:3, 3] = pose[:3]
        else:
            raise NotImplementedError

        pts_homog = torch.cat(
            [pts, torch.ones((N_rays, N_samples, 1), device=device)], dim=-1
        )  # [N_rays, N_samples, 4]
        pts_homog_flat = pts_homog.reshape((-1, 4))  # [N_rays*N_samples, 4]

        pts_dynamic_homog_flat = torch.einsum("ij,nj->ni", pose_matrix, pts_homog_flat)
        pts_dynamic_homog = pts_dynamic_homog_flat.reshape((N_rays, N_samples, 4))
        pts_dynamic = pts_dynamic_homog[..., :3]  # [N_rays, N_samples, 3]

        viewdirs_dynamic = torch.einsum("ij,nj->ni", pose_matrix[:3, :3], viewdirs)

        raw_alpha_dynamic, raw_rgb_dynamic = dynamic_model(
            pts_dynamic, viewdirs_dynamic, step=step
        )

        result = raw2outputs_star(
            raw_alpha_static,
            raw_rgb_static,
            raw_alpha_dynamic,
            raw_rgb_dynamic,
            z_vals,
            rays_d,
            # From the paper: "we add small Gaussian noise to the density outputs during
            # appearance initialization but turn it off during online training."
            0,
            static_model.white_bkgd,
        )

        return result
