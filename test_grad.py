from models.star import STaR
from models.rendering import render_star
from utils.io import *

parser = config_parser()
args = parser.parse_args()
args.use_viewdirs = True
args.N_importance = 16

'''
            pts: [N_rays, N_samples, 3]. Points sampled according to stratified sampling.
            viewdirs: [N_rays, 3]. View directions of rays.
            z_vals: [N_rays, N_samples]. Integration time.
            rays_d: [N_rays, 3]. Unnormalized directions of rays.
            frames: [N_rays,1]. Time steps of the rays. None during appearance init.
            is_coarse: True if render using coarse models, False if render using fine models
            object_pose: [4, 4]. Pose of the dynamic object, same for all rays, used in testing.

'''

star_model = STaR(num_frames=15, args=args)

N_rays = 16
N_samples = 4
pts = torch.zeros((N_rays, N_samples, 3), requires_grad=True) 
#pts.retain_grad()
viewdirs = torch.ones((N_rays, 3))
z_vals = torch.ones((N_rays, N_samples))
rays_o = torch.ones((N_rays,3))
rays_d = torch.ones((N_rays,3))
frames = torch.ones((N_rays, 1), dtype=torch.long)


rgb, disp, acc, extras, entropy = render_star(star_model, pts, viewdirs, z_vals, rays_o, rays_d, 
    frames=frames, retraw=True, N_importance=args.N_importance)

gt_rgb = torch.randn((N_rays, 3)) 

loss = torch.sum(gt_rgb - rgb)
loss.backward()
#print(extras['transformed_pts'].grad)
#print(pts.grad)


