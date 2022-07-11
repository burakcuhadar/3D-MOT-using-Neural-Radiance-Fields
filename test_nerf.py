import os
import numpy as np
import torch
import imageio
from utils.logging import LoggerWandb
from torch.utils.data import DataLoader
from models.nerf import NeRF
from tqdm import tqdm, trange


from models.rendering import render_path
from models.rendering import mse2psnr, img2mse, to8b
from datasets import dataset_dict
from utils.io import *


def test():

    parser = config_parser()
    args = parser.parse_args()

    # Create test dataset & loader
    dataset_class = dataset_dict[args.dataset_type]
    test_dataset = dataset_class(args, split='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=test_dataset.collate_sample_pts_and_viewdirs)

    # Create dataset for video rendering & loader
    render_video_dataset = dataset_class(args, split='render_video')
    render_video_dataloader = DataLoader(
        render_video_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=render_video_dataset.collate_sample_pts_and_viewdirs)

    create_log_dir(args.basedir, args.expname)
    #copy_config_save_args(args.basedir, args.expname, args)

    #Create models
    model_coarse = NeRF(D=args.netdepth, W=args.netwidth, args=args)
    model_coarse.to(device)
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, args=args)
        model_fine.to(device)
    
    model_coarse.eval()
    model_fine.eval()

    # Load checkpoint
    if args.ckpt_path is None:
        print("No checkpoint given, cannot test nerf")
        return
    load_ckpt_for_test(args.ckpt_path, model_coarse, model_fine)

    logger_wandb = LoggerWandb(project_name=args.expname+'_test', args=args)

    
    # Render test poses and save them to disk
    testsavedir = os.path.join(args.basedir, args.expname, 'testset')
    os.makedirs(testsavedir, exist_ok=True)
    rgbs, disps, test_loss, test_psnr = render_path(test_dataloader, model_coarse, model_fine, 
        args.N_importance, device, render_factor=args.render_factor, savedir=testsavedir)
    print('Saved test set renderings')
    # Choose random views to log
    indices = np.random.choice(len(rgbs), 5, replace=False)
    logger_wandb.log_test(test_loss, test_psnr, rgbs[indices], disps[indices])
    

    # Create video
    print('Starting video rendering')
    rgbs, disps, _, _ = render_path(render_video_dataloader, model_coarse, model_fine, args.N_importance, 
        device, render_factor=args.render_factor)
    print('Done video rendering, saving', rgbs.shape, disps.shape)
    moviebase = os.path.join(args.basedir, args.expname, '{}_spiral'.format(args.expname))
    rgb_path = moviebase + 'rgb.mp4'
    disp_path = moviebase + 'disp.mp4'
    imageio.mimwrite(rgb_path, rgbs, fps=30, quality=8)
    imageio.mimwrite(disp_path, disps, fps=30, quality=8)
    logger_wandb.log_video(rgb_path, disp_path)


if __name__=='__main__':
    test()


