import os
import torch
import random
import numpy as np
from collections import OrderedDict
import logging


def configure_logger(dirpath, filename):
    os.makedirs(dirpath)
    logging.basicConfig(filename=f"{dirpath}/{filename}", level=logging.INFO)
    logging.info("Logging started.")


def create_log_dir(basedir, expname):
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)


def copy_config_save_args(basedir, expname, args):
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())


def load_star_network_from_ckpt(ckpt_path, star_network):
    ckpt = torch.load(ckpt_path)
    state_dict = OrderedDict()
    for key in ckpt["state_dict"]:
        if key.startswith("star_network"):
            state_dict[key.split("star_network", 1)[1][1:]] = ckpt["state_dict"][key]

    star_network.load_state_dict(state_dict)


def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--job_id", help="slurm job id")
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./logs/", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/llff/fern", help="input data directory"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=15,
        help="number of object poses in the dataset",
    )
    parser.add_argument(
        "--num_vehicles",
        type=int,
        default=1,
        help="number of objects in the dataset",
    )
    parser.add_argument(
        "--has_depth_data",
        action="store_true",
        help="whether the dataset contains depth images in addition to semantic and rgb",
    )

    # training options
    parser.add_argument(
        "--epochs", type=int, default=0, help="number of epochs the model trained"
    )
    parser.add_argument(
        "--epochs_appearance",
        type=int,
        default=0,
        help="maximum number of epochs for star appearance initialization",
    )
    parser.add_argument(
        "--epochs_online",
        type=int,
        default=0,
        help="number of epochs for star online training",
    )
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="layers in fine network"
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lrate_static", type=float, default=5e-4, help="learning rate for static nerf"
    )
    parser.add_argument(
        "--lrate_dynamic",
        type=float,
        default=5e-4,
        help="learning rate for dynamic nerf",
    )
    parser.add_argument(
        "--lrate_pose",
        type=float,
        default=5e-4,
        help="learning rate for pose parameters",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="number of batches to accumulate grads, 1 means no grad accumulation",
    )

    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=500,
        help="Period of learning rate decay in epochs",
    )
    parser.add_argument(
        "--lrate_decay_rate", type=float, default=0.1, help="Learning rate decay rate"
    )
    parser.add_argument(
        "--lrate_decay_steps",
        nargs="+",
        type=int,
        default=[],
        help="scheduler decay steps for multisteplr",
    )

    parser.add_argument(
        "--pose_lrate_decay",
        type=int,
        default=500,
        help="Period of pose learning rate decay in epochs",
    )
    parser.add_argument(
        "--pose_lrate_decay_rate",
        type=float,
        default=0.1,
        help="Pose Learning rate decay rate",
    )
    parser.add_argument(
        "--pose_lrate_decay_steps",
        nargs="+",
        type=int,
        default=[],
        help="pose scheduler decay steps for multisteplr",
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help="""number of pts sent through network in parallel, decrease if running out of 
                              memory""",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="checkpoint file to load state"
    )

    # star training options
    parser.add_argument(
        "--skip_appearance_init",
        action="store_true",
        help="skip apperance initialization step",
    )
    parser.add_argument(
        "--appearance_ckpt_path",
        type=str,
        default=None,
        help="appearance init checkpoint file to load state",
    )
    parser.add_argument(
        "--online_ckpt_path",
        type=str,
        default=None,
        help="online training checkpoint file to load state",
    )

    parser.add_argument(
        "--car_sample_ratio",
        type=float,
        default=0.5,
        help="ratio of the car rays to non-car rays used for each mini batch during training",
    )
    parser.add_argument(
        "--end_barf",
        type=int,
        default=-1,
        help="the epoch that barf encoding becomes equal to the original positional encoding",
    )

    parser.add_argument(
        "--load_gt_poses", action="store_true", help="load gt poses to debug"
    )
    parser.add_argument(
        "--noisy_pose_init",
        action="store_true",
        help="initialize poses with perturbed gt poses",
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--use_viewdirs", action="store_true", help="use full 5D input instead of 3D"
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--save_video_frames",
        action="store_true",
        help="denotes whether frames of the rendered video will also be saved as png",
    )
    parser.add_argument(
        "--no_test_set",
        action="store_true",
        help="denotes whether the dataset has no test views",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    # parser.add_argument("--render_factor", type=int, default=0,
    #                     help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    parser.add_argument(
        "--scale_factor", type=float, default=-1, help="scaling factor for large scenes"
    )

    # training options
    """NOTE: precrop is not implemented
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    """

    # star hyperparameters
    parser.add_argument(
        "--appearance_init_thres",
        type=float,
        default=2e-3,
        help="""threshold for loss to finish appearance initialization training (m1 in the 
                             paper)""",
    )
    parser.add_argument(
        "--online_thres",
        type=float,
        default=1e-3,
        help="threshold for loss to finish online training (m2 in the paper)",
    )
    parser.add_argument(
        "--initial_num_frames",
        type=int,
        default=5,
        help="initial number of frames to start online training (k0 in the paper)",
    )
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=2e-3,
        help="entropy regularization weight (beta in the paper)",
    )

    # dataset options
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="blender",
        help="options: llff / blender / deepvoxels / carla_static / carla_star",
    )
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="""will load 1/N images from test/val sets, useful for large datasets like 
                             deepvoxels""",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of workers used in torch dataloader",
    )
    parser.add_argument("--near", type=float, default=3.0, help="near limit for rays")
    parser.add_argument("--far", type=float, default=80.0, help="far limit for rays")
    parser.add_argument(
        "--far_dist",
        type=float,
        default=1e10,
        help="the distance between the farthest sample and the sample before that",
    )

    ## blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )
    parser.add_argument(
        "--half_res",
        action="store_true",
        help="load blender synthetic data at 400x400 instead of 800x800",
    )

    ## llff flags
    parser.add_argument(
        "--factor", type=int, default=8, help="downsample factor for LLFF images"
    )
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument(
        "--lindisp",
        action="store_true",
        help="sampling linearly in disparity rather than depth",
    )

    # logging/saving options
    parser.add_argument(
        "--epoch_ckpt", type=int, default=100, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--epoch_print",
        type=int,
        default=10,
        help="frequency of logging loss and metrics to console",
    )
    parser.add_argument(
        "--epoch_val", type=int, default=50, help="frequency of validation view saving"
    )

    parser.add_argument(
        "--mixed_precision", action="store_true", help="train with mixed precision"
    )

    # DSNerf options
    parser.add_argument(
        "--depth_loss", action="store_true", help="train with depth loss"
    )
    parser.add_argument("--depth_lambda", type=float, help="depth lambda used for loss")
    parser.add_argument(
        "--sigma_loss", action="store_true", help="train with sigma loss"
    )
    parser.add_argument("--sigma_lambda", type=float, help="sigma lambda used for loss")

    return parser


def set_matmul_precision():
    # Check whether the gpu has tensor cores, in that case set matmul precision
    major, _ = torch.cuda.get_device_capability(torch.device("cuda"))
    # Ampere and later leverage tensor cores, where this setting becomes useful
    ampere_or_later = major >= 8
    if ampere_or_later:
        print("Setting matmul precision to medium...")
        torch.set_float32_matmul_precision("medium")
