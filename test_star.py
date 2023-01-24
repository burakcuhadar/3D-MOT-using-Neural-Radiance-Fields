import imageio
import os

from models.rendering import render_path, render_path_star
from models.rendering import mse2psnr, img2mse, to8b
from models.star import STaR
from datasets import dataset_dict
from utils.io import *
from utils.logging import LoggerWandb
from torch.utils.data import DataLoader
from utils.metrics import evaluate_trajectory
from lietorch import SO3


def evaluate_rpe(star_poses, gt_poses):
    '''
        star_poses: [num_frames, 6], torch.Tensor
        gt_poses: [num_frames, 4, 4]
    '''
    assert star_poses.shape[0] == gt_poses.shape[0]
    num_frames = gt_poses.shape[0]

    traj_gt = {}
    traj_est = {}
    for i in range(num_frames):
        traj_gt[i] = gt_poses[i]

        mat_est = np.eye(4, dtype=np.float32)
        mat_est[:3,:3] = SO3.exp(star_poses[i,3:]).matrix().cpu().detach().numpy()[:3,:3] 
        mat_est[:3,3] = star_poses[i,:3].cpu().detach().numpy() 
        traj_est[i] = mat_est
    
    result = evaluate_trajectory(traj_gt,
                                 traj_est,
                                 param_max_pairs=10000,
                                 param_fixed_delta=True, #TODO?
                                 param_delta=1.00)

    #stamps = numpy.array(result)[:,0]
    trans_error = np.array(result)[:,4]
    rot_error = np.array(result)[:,5]

    print("RPE EVALUATION:")
    print("compared_pose_pairs %d pairs"%(len(trans_error)))

    print("translational_error.rmse %f m"%np.sqrt(np.dot(trans_error,trans_error) / len(trans_error)))
    print("translational_error.mean %f m"%np.mean(trans_error))
    print("translational_error.median %f m"%np.median(trans_error))
    print("translational_error.std %f m"%np.std(trans_error))
    print("translational_error.min %f m"%np.min(trans_error))
    print("translational_error.max %f m"%np.max(trans_error))

    print("rotational_error.rmse %f deg"%(np.sqrt(np.dot(rot_error,rot_error) / len(rot_error)) * 180.0 / np.pi))
    print("rotational_error.mean %f deg"%(np.mean(rot_error) * 180.0 / np.pi))
    print("rotational_error.median %f deg"%(np.median(rot_error) * 180.0 / np.pi))
    print("rotational_error.std %f deg"%(np.std(rot_error) * 180.0 / np.pi))
    print("rotational_error.min %f deg"%(np.min(rot_error) * 180.0 / np.pi))
    print("rotational_error.max %f deg"%(np.max(rot_error) * 180.0 / np.pi))

    
def evaluate_ate(star_poses, gt_poses):
    '''
        star_poses: [num_frames, 6], torch.Tensor
        gt_poses: [num_frames, 6]
    '''
    assert star_poses.shape[0] == gt_poses.shape[0]

    gt_trans = gt_poses[:,:3].cpu().detach().numpy()
    star_trans = star_poses[:,:3].cpu().detach().numpy() 
    diff = (star_trans - gt_trans).T # [3, num_frames]
    trans_error = np.sqrt(np.sum(diff * diff, 0))
    
    print("compared_pose_pairs %d pairs"%(len(trans_error)))
    print("absolute_translational_error.rmse %f m"%np.sqrt(np.dot(trans_error,trans_error) / len(trans_error)))
    print("absolute_translational_error.mean %f m"%np.mean(trans_error))
    print("absolute_translational_error.median %f m"%np.median(trans_error))
    print("absolute_translational_error.std %f m"%np.std(trans_error))
    print("absolute_translational_error.min %f m"%np.min(trans_error))
    print("absolute_translational_error.max %f m"%np.max(trans_error))



def test():
    parser = config_parser()
    args = parser.parse_args()
    logger_wandb = LoggerWandb(project_name=args.expname+'_test', args=args)

    dataset_class = dataset_dict[args.dataset_type]

    create_log_dir(args.basedir, args.expname+'_test')
    copy_config_save_args(args.basedir, args.expname+'_test', args)

    star_model = STaR(num_frames=15, args=args) # TODO num_frames from args
    star_model.to(device)
    star_model.eval()

    if args.online_ckpt_path is None:
        print("No checkpoint given, cannot test star")
        return
    step = load_star_ckpt_for_test(args.online_ckpt_path, star_model)

    
    render_video_dataset = dataset_class(args, split='render_video')
    render_video_dataloader = DataLoader(
        render_video_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=render_video_dataset.collate_sample_pts_and_viewdirs)

    # Save the rendered images of views used to create the video
    if args.save_video_frames:
        video_frames_savedir = os.path.join(args.basedir, args.expname+'_test', 'video_frames')
        os.makedirs(video_frames_savedir, exist_ok=True)

    evaluate_rpe(star_model.get_poses()[1:], render_video_dataset.gt_relative_poses_matrices)
    evaluate_ate(star_model.get_poses()[1:], render_video_dataset.gt_relative_poses[1:])

    
    # Create video
    print('Starting video rendering')
    rgbs, disps, rgb_statics0, rgb_dynamics0, rgb_statics, rgb_dynamics = render_path_star(render_video_dataloader, 
        star_model, args.N_importance, device, savedir=video_frames_savedir if args.save_video_frames else None, step=step)

    print('Done video rendering, saving', rgbs.shape, disps.shape)
    moviebase = os.path.join(args.basedir, args.expname+'_test', '{}_translation'.format(args.expname))
    rgb_path = moviebase + 'rgb.mp4'
    disp_path = moviebase + 'disp.mp4'
    rgb_static_path = moviebase + 'rgb_static.mp4'
    rgb_dynamic_path = moviebase + 'rgb_dynamic.mp4'
    imageio.mimwrite(rgb_path, rgbs, fps=30, quality=8)
    imageio.mimwrite(disp_path, disps, fps=30, quality=8)
    imageio.mimwrite(rgb_static_path, rgb_statics, fps=30, quality=8)
    imageio.mimwrite(rgb_dynamic_path, rgb_dynamics, fps=30, quality=8)
    logger_wandb.log_video(rgb_path, disp_path, rgb_static_path, rgb_dynamic_path)
    
    #TODO log rgb_static0/dynamic0 video as well?
    
    

if __name__=='__main__':
    set_seeds()
    test()
