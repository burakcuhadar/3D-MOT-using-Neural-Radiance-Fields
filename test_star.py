import imageio
import os

from models.rendering import render_path, render_path_star
from models.rendering import mse2psnr, img2mse, to8b
from models.star import STaR
from datasets import dataset_dict
from utils.io import *
from utils.logging import LoggerWandb
from torch.utils.data import DataLoader



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
    load_star_ckpt_for_test(args.online_ckpt_path, star_model)

    
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

    # Create video
    print('Starting video rendering')
    rgbs, disps, rgb_statics0, rgb_dynamics0, rgb_statics, rgb_dynamics = render_path_star(render_video_dataloader, 
        star_model, args.N_importance, device, savedir=video_frames_savedir if args.save_video_frames else None)

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
    test()
