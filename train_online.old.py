import os
import torch
import numpy as np
import pytorch_lightning as pl
import pypose as pp
from lietorch import SE3
from torch import inf

from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from torch.utils.data import DataLoader
from models.star import STaR
import torch.nn as nn
from tqdm import tqdm

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import StructuralSimilarityIndexMeasure
import imageio
from einops import rearrange

from pytorch3d.transforms import se3_log_map


from pytorch_lightning.utilities import grad_norm

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning import Trainer, seed_everything

from models.rendering import render_star_online, sample_pts
from models.rendering import mse2psnr, img2mse, to8b
from models.loss import compute_sigma_loss, compute_depth_loss
from datasets.carla_star_online import StarOnlineDataset
from optimizer.hybrid_optimizer import HybridOptim, HybridLRS
from callbacks.online_training_callback import StarOnlineCallback
from utils.logging import log_val_table_online, log_test_table_online
from utils.visualization import visualize_depth
from utils.metrics import get_pose_metrics, evaluate_rpe
from utils.io import *


def get_scheduler(
    optimizer, lrate_decay_rate, lrate_decay=None, lrate_decay_steps=None
):
    if lrate_decay_steps:
        return MultiStepLR(
            optimizer, milestones=lrate_decay_steps, gamma=lrate_decay_rate
        )
    elif lrate_decay:
        return StepLR(optimizer, step_size=lrate_decay, gamma=lrate_decay_rate)
    else:
        return None

class StarOnline(pl.LightningModule):
    def __init__(self, args, star_network):
        super().__init__()

        self.args = args
        self.star_network = star_network

        '''self.register_parameter(
            "poses",
            nn.Parameter(torch.zeros((args.num_frames - 1, 6)), requires_grad=True),
        )'''

        self.poses = nn.ParameterList(
            #[nn.Parameter(torch.zeros(1,7), requires_grad=True) for i in range(args.num_frames-1)]
            #[nn.Parameter(SE3.Identity(1, dtype=torch.float32).log()) for i in range(args.num_frames-1)]
            [nn.Parameter(pp.identity_SE3().tensor().unsqueeze(0)) for i in range(args.num_frames-1)]
        )        
        print("poses:", self.poses[0].shape)
        print("len poses:", len(self.poses))
        '''
        self.poses = nn.ParameterList(
            [pp.Parameter(pp.identity_SE3(1, dtype=torch.float32), requires_grad=True)  for i in range(args.num_frames-1)]
        )
        '''

        self.register_buffer(
            "current_frame_num",
            torch.tensor([args.initial_num_frames], dtype=torch.long),
        )

        self.register_buffer("start_frame", torch.tensor([0], dtype=torch.long))

        self.training_fine_losses = []

        #self.loss = nn.HuberLoss(delta=0.5)
        self.loss = nn.MSELoss()

    def forward(self, batch):
        pts, z_vals = sample_pts(
            batch["rays_o"],
            batch["rays_d"],
            self.train_dataset.near,
            self.train_dataset.far,
            self.args.N_samples,
            self.args.perturb,
            self.args.lindisp,
            self.training,
        )

        viewdirs = batch["rays_d"] / torch.norm(
            batch["rays_d"], dim=-1, keepdim=True
        )  # [N_rays, 3]

        if self.args.load_gt_poses:
            """pose0 = torch.zeros((1, 6), requires_grad=False, device=self.device)
            poses = torch.cat((pose0, batch["gt_relative_poses"][1:, ...]), dim=0)
            pose = poses[batch["frames"][0]][0]"""

            pose = batch["gt_relative_poses"][batch["frames"][0]][0]

            """pose = self.train_dataset.gt_vehicle_poses.to(
                batch["rays_o"].device
            )[batch["frames"][0]][0]"""
        elif self.trainer.testing:
            pose = batch["object_pose"]
        else:
            pose0 = pp.identity_SE3(device=self.device).tensor().unsqueeze(0)
            poses = torch.cat(([pose0] + [posei for posei in self.poses]), dim=0)
            pose = poses[batch["frames"][0]][0]

        return render_star_online(
            self.star_network,
            pts,
            viewdirs,
            z_vals,
            batch["rays_o"],
            batch["rays_d"],
            self.args.N_importance,
            pose,
            step=self.current_epoch,
        )

    def training_step(self, batch, batch_idx):
        result = self(batch)

        #img_loss = img2mse(result["rgb"], batch["target"])
        img_loss = self.loss(result["rgb"], batch["target"])

        loss = img_loss
        psnr = mse2psnr(img_loss)
        #img_loss0 = img2mse(result["rgb0"], batch["target"])
        img_loss0 = self.loss(result["rgb0"], batch["target"])

        loss += img_loss0
        psnr0 = mse2psnr(img_loss0)
        loss += self.args.entropy_weight * (result["entropy"] + result["entropy0"])

        if self.args.depth_loss:
            depth_loss = compute_depth_loss(
                result["depth"],
                batch["target_depth"],
                self.train_dataset.near,
                self.train_dataset.far,
            )
            loss = loss + self.args.depth_lambda * depth_loss
        if self.args.sigma_loss:
            sigma_loss = compute_sigma_loss(
                result["weights"],
                result["z_vals"],
                result["dists"],
                batch["target_depth"],
            )
            loss = loss + self.args.sigma_lambda * sigma_loss

        self.training_fine_losses.append(img_loss)
        self.log("train/fine_loss", img_loss, on_step=False, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/psnr", psnr, on_step=False, on_epoch=True)
        self.log("train/psnr0", psnr0, on_step=False, on_epoch=True)
        self.log("train/zero dist count", result["zero_dist_count"], on_step=False, on_epoch=True)

        if self.args.depth_loss:
            self.log("train/depth_loss", depth_loss, on_step=False, on_epoch=True)
        if self.args.sigma_loss:
            self.log("train/sigma_loss", sigma_loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.star_network.static_coarse_nerf.parameters())
                    + list(self.star_network.static_fine_nerf.parameters()),
                    "lr": self.args.lrate_static,
                },
                {
                    "params": list(self.star_network.dynamic_coarse_nerf.parameters())
                    + list(self.star_network.dynamic_fine_nerf.parameters()),
                    "lr": self.args.lrate_dynamic,
                },
            ],
            lr=self.args.lrate,
            betas=(0.9, 0.999),
        )
        pose_optimizer = torch.optim.Adam(self.poses, lr=self.args.lrate_pose)
        #pose_optimizer = torch.optim.NAdam(self.poses, lr=self.args.lrate_pose)
        
        #self.dummy = nn.Parameter(torch.zeros(1,1), requires_grad=True)
        #pose_optimizer = torch.optim.Adam([self.dummy], lr=self.args.lrate_pose)
        
        
        #pose_optimizer = torch.optim.SGD(self.poses, lr=self.args.lrate_pose)
        #pose_optimizer = torch.optim.RAdam(self.poses, lr=self.args.lrate_pose)
        #pose_optimizer = torch.optim.RAdam(self.poses, lr=self.args.lrate_pose)

        scheduler = get_scheduler(
            optimizer,
            self.args.lrate_decay_rate,
            lrate_decay=self.args.lrate_decay,
            lrate_decay_steps=self.args.lrate_decay_steps,
        )
        pose_scheduler = get_scheduler(
            pose_optimizer,
            self.args.pose_lrate_decay_rate,
            lrate_decay=self.args.pose_lrate_decay,
            lrate_decay_steps=self.args.pose_lrate_decay_steps,
        )

        hoptim = HybridOptim([optimizer, pose_optimizer])

        return [hoptim], [
            HybridLRS(hoptim, 0, scheduler),
            HybridLRS(hoptim, 1, pose_scheduler),
        ]

    def validation_step(self, batch, batch_idx):
        result = self(batch)

        val_H = self.val_dataset.H
        val_W = self.val_dataset.W

        self.log("val/frame", batch["frames"][0, 0])

        #val_mse = img2mse(result["rgb"], batch["target"])
        val_mse = self.loss(result["rgb"], batch["target"])
        psnr = mse2psnr(val_mse)
        lpips = self.eval_lpips(
            torch.reshape(result["rgb"], (val_H, val_W, 3)), 
            torch.reshape(batch["target"], (val_H, val_W, 3))
        )
        ssim = self.eval_ssim(
            torch.reshape(result["rgb"], (val_H, val_W, 3)), 
            torch.reshape(batch["target"], (val_H, val_W, 3))
        )

        self.log("val/mse", val_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True)
        self.log("val/lpips", lpips, on_step=False, on_epoch=True)
        self.log("val/ssim", ssim, on_step=False, on_epoch=True)
        self.log("val/zero dist count", result["zero_dist_count"], on_step=False, on_epoch=True)

        # Log visualizations for the first view (see dataset implementation)
        rgb0 = to8b(
            torch.reshape(result["rgb0"], (val_H, val_W, 3)).cpu().detach().numpy(),
            "rgb0",
        )
        depth0 = visualize_depth(result["depth0"]).reshape((val_H, val_W, 3))
        z_std = to8b(
            torch.reshape(result["z_std"], (val_H, val_W, 1)).cpu().detach().numpy(),
            "z_std",
        )
        rgb = to8b(
            torch.reshape(result["rgb"], (val_H, val_W, 3)).cpu().detach().numpy(),
            "rgb",
        )
        depth = visualize_depth(result["depth"]).reshape((val_H, val_W, 3))
        target = to8b(
            torch.reshape(batch["target"], (val_H, val_W, 3)).cpu().detach().numpy(),
            "target",
        )

        # Visualize static and dynamic nerfs separately
        rgb_static0 = to8b(
            torch.reshape(result["rgb_static0"], (val_H, val_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_static0",
        )
        rgb_dynamic0 = to8b(
            torch.reshape(result["rgb_dynamic0"], (val_H, val_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_dynamic0",
        )
        rgb_static = to8b(
            torch.reshape(result["rgb_static"], (val_H, val_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_static",
        )
        rgb_dynamic = to8b(
            torch.reshape(result["rgb_dynamic"], (val_H, val_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_dynamic",
        )
        depth_static = visualize_depth(result["depth_static"]).reshape(
            (val_H, val_W, 3)
        )
        depth_dynamic = visualize_depth(result["depth_dynamic"]).reshape(
            (val_H, val_W, 3)
        )
        depth_static0 = visualize_depth(result["depth_static0"]).reshape(
            (val_H, val_W, 3)
        )
        depth_dynamic0 = visualize_depth(result["depth_dynamic0"]).reshape(
            (val_H, val_W, 3)
        )

        self.logger.log_image(
            "val/imgs", 
            [
                rgb,
                target,
                rgb_dynamic,
                rgb_static,
                depth,
                depth_dynamic,
                depth_static,
                rgb0,
                rgb_dynamic0,
                rgb_static0,
                depth0,
                depth_dynamic0,
                depth_static0,
                z_std,
            ], 
            step=self.global_step
        )

        """log_val_table_online(
            self.logger,
            self.current_epoch,
            rgb,
            target,
            rgb_dynamic,
            rgb_static,
            depth,
            depth_dynamic,
            depth_static,
            rgb0,
            rgb_dynamic0,
            rgb_static0,
            depth0,
            depth_dynamic0,
            depth_static0,
            z_std,
        )"""

        # Log pose metrics
        trans_error, rot_error, last_trans_error, last_rot_error, rot_error_euler, last_rot_error_euler = get_pose_metrics(
            torch.cat(list(self.poses[0 : self.current_frame_num - 1]), dim=0),
            batch["gt_relative_poses"][1 : self.current_frame_num, ...],
        )
        self.log("val/trans_error", trans_error)
        self.log("val/rot_error", rot_error)
        self.log("val/euler_rot_error", rot_error_euler)
        self.log("val/last_trans_error", last_trans_error)
        self.log("val/last_rot_error", last_rot_error)
        self.log("val/last_euler_rot_error", last_rot_error_euler)

        rpe_trans_rmse, rpe_rot_rmse = evaluate_rpe( #TODO does rpe work for quat
            torch.cat(list(self.poses[: self.current_frame_num - 1]), dim=0),
            self.train_dataset.gt_relative_poses_matrices[
                1 : self.current_frame_num
            ].to(self.device),
        )
        self.log("val/rpe_trans_rmse", rpe_trans_rmse)
        self.log("val/rpe_rot_rmse", rpe_rot_rmse)

    def test_step(self, batch, batch_idx):
        result = self(batch)

        test_H = self.test_dataset.H
        test_W = self.test_dataset.W

        rgb0 = to8b(
            torch.reshape(result["rgb0"], (test_H, test_W, 3)).cpu().detach().numpy(),
            "rgb0",
        )
        depth0 = visualize_depth(result["depth0"]).reshape((test_H, test_W, 3))
        z_std = to8b(
            torch.reshape(result["z_std"], (test_H, test_W, 1)).cpu().detach().numpy(),
            "z_std",
        )
        rgb = to8b(
            torch.reshape(result["rgb"], (test_H, test_W, 3)).cpu().detach().numpy(),
            "rgb",
        )
        depth = visualize_depth(result["depth"]).reshape((test_H, test_W, 3))

        # Visualize static and dynamic nerfs separately
        rgb_static0 = to8b(
            torch.reshape(result["rgb_static0"], (test_H, test_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_static0",
        )
        rgb_dynamic0 = to8b(
            torch.reshape(result["rgb_dynamic0"], (test_H, test_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_dynamic0",
        )
        rgb_static = to8b(
            torch.reshape(result["rgb_static"], (test_H, test_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_static",
        )
        rgb_dynamic = to8b(
            torch.reshape(result["rgb_dynamic"], (test_H, test_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_dynamic",
        )
        depth_static = visualize_depth(result["depth_static"]).reshape(
            (test_H, test_W, 3)
        )
        depth_dynamic = visualize_depth(result["depth_dynamic"]).reshape(
            (test_H, test_W, 3)
        )
        depth_static0 = visualize_depth(result["depth_static0"]).reshape(
            (test_H, test_W, 3)
        )
        depth_dynamic0 = visualize_depth(result["depth_dynamic0"]).reshape(
            (test_H, test_W, 3)
        )

        log_test_table_online(
            self.logger,
            batch_idx,
            rgb,
            rgb_dynamic,
            rgb_static,
            depth,
            depth_dynamic,
            depth_static,
            rgb0,
            rgb_dynamic0,
            rgb_static0,
            depth0,
            depth_dynamic0,
            depth_static0,
            z_std,
        )

        self.test_rgbs.append(rgb)
        self.test_depths.append(depth)
        self.test_rgb_statics.append(rgb_static)
        self.test_rgb_dynamics.append(rgb_dynamic)

    def on_test_start(self):
        self.video_basepath = os.path.join("videos", self.logger.version)
        self.test_rgbs = []
        self.test_depths = []
        self.test_rgb_statics = []
        self.test_rgb_dynamics = []

    def on_test_end(self):
        self.test_rgbs = np.stack(self.test_rgbs, 0)
        self.test_depths = np.stack(self.test_depths, 0)
        self.test_rgb_statics = np.stack(self.test_rgb_statics, 0)
        self.test_rgb_dynamics = np.stack(self.test_rgb_dynamics, 0)

        #TODO compute and log ms2e, lpips and ssim

        rgb_path = self.video_basepath + "rgb.mp4"
        depth_path = self.video_basepath + "depth.mp4"
        rgb_static_path = self.video_basepath + "rgb_static.mp4"
        rgb_dynamic_path = self.video_basepath + "rgb_dynamic.mp4"
        imageio.mimwrite(rgb_path, self.test_rgbs, fps=30, quality=8)
        imageio.mimwrite(depth_path, self.test_depths, fps=30, quality=8)
        imageio.mimwrite(rgb_static_path, self.test_rgb_statics, fps=30, quality=8)
        imageio.mimwrite(rgb_dynamic_path, self.test_rgb_dynamics, fps=30, quality=8)

    """
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)
    """

    def setup(self, stage):
        self.train_dataset = StarOnlineDataset(
            self.args,
            "train",
            self.args.num_frames,
            self.args.initial_num_frames,
        )
        self.val_dataset = StarOnlineDataset(
            self.args, 
            "val", 
            self.args.num_frames, 
            self.args.initial_num_frames
        )
        self.test_dataset = StarOnlineDataset(
            self.args,
            "test",
            self.args.num_frames,
            self.args.initial_num_frames,
        )

        if self.args.load_gt_poses:
            pass
            """TODO remove or uncomment: self.register_buffer(
                "gt_poses", self.train_dataset.get_gt_vehicle_poses(self.args)
            )"""
        elif self.args.noisy_pose_init and (self.args.online_ckpt_path is None):
            with torch.no_grad():
                for i, noisy_gt_relative_pose in enumerate(self.train_dataset.get_noisy_gt_relative_poses()[1:].to(self.device)):
                    print("noisy_gt_relative_pose shape", noisy_gt_relative_pose.shape)
                    print("poses[i] shape", self.poses[i].shape)
                    self.poses[i].copy_(noisy_gt_relative_pose)

                print("poses shape", torch.cat(list(self.poses), dim=0).shape)
                print("gt_relative_poses shape", self.train_dataset.gt_relative_poses[1:].shape)

                trans_error, rot_error, _, _, _, _ = get_pose_metrics(
                    torch.cat(list(self.poses), dim=0),
                    self.train_dataset.gt_relative_poses[1:],
                    reduce=False
                )
                print("trans errors:\n", trans_error)
                print("rot errors:\n", rot_error)
        
        # Init SSIM metric
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        # Init LPIPS metric
        self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg', normalize=True)
        for p in self.val_lpips.net.parameters():
            p.requires_grad = False


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=1,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            num_workers=1,
            pin_memory=True,
        )

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint.keys()):
            if k.startswith("val_lpips") or k.startswith("val_ssim"):
                checkpoint.pop(k)
        
    def eval_lpips(self, pred_img, gt_img):
        # Rearrange for torchmetrics library
        pred_img_rearranged = rearrange(pred_img[None, ...], "B H W C -> B C H W")
        gt_img_rearranged = rearrange(gt_img[None, ...], "B H W C -> B C H W")

        self.val_lpips(torch.clip(pred_img_rearranged, 0, 1), torch.clip(gt_img_rearranged, 0, 1))
        lpips = self.val_lpips.compute()
        self.val_lpips.reset()
        return lpips
    
    def eval_ssim(self, pred_img, gt_img):
        # Rearrange for torchmetrics library
        pred_img_rearranged = rearrange(pred_img[None, ...], "B H W C -> B C H W")
        gt_img_rearranged = rearrange(gt_img[None, ...], "B H W C -> B C H W")

        self.val_ssim(pred_img_rearranged, gt_img_rearranged)
        ssim = self.val_ssim.compute()
        self.val_ssim.reset()
        return ssim


def create_model(args):
    network = STaR(args)

    if args.appearance_ckpt_path:
        load_star_network_from_ckpt(args.appearance_ckpt_path, network)
    elif not args.online_ckpt_path:
        print("Either app init ckpt or online ckpt should be provided")
        raise NotImplementedError

    model = StarOnline(args, network)
    # model = torch.compile(model)

    return model


def train(args, model):
    logger = WandbLogger(project=args.expname)
    #logger.watch(model, log="all")
    logger.experiment.config.update(args)
    logger.experiment.log_code(root=args.code_dir)

    configure_logger(f"logs/online/{logger.version}", "training.log")
    logging.info(f"Slurm job id is {args.job_id}")

    epoch_val = (
        args.epoch_val
        if args.accumulate_grad_batches == 1
        else args.accumulate_grad_batches
    )

    print("Saving checkpoints at:", f"ckpts/online/{logger.version}")
    ckpt_cb = ModelCheckpoint(
        dirpath=f"ckpts/online/{logger.version}",
        filename="{epoch:d}",
        every_n_epochs=epoch_val,
        save_on_train_epoch_end=True,
        save_top_k=-1,
    )

    star_online_cb = StarOnlineCallback(args)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1), star_online_cb, lr_monitor]

    trainer = Trainer(
        max_epochs=args.epochs_online,
        check_val_every_n_epoch=epoch_val,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=False,
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
        # detect_anomaly=True,
        # profiler="simple",
    )


    trainer.fit(model, ckpt_path=args.online_ckpt_path) 
    #trainer.test(model, ckpt_path=args.online_ckpt_path)

if __name__ == "__main__":
    print("torch version", torch.__version__)
    print("torch cuda version", torch.version.cuda)
    print("torch gpu available", torch.cuda.is_available())
    print(
        "gpu memory(in GiB)",
        torch.cuda.get_device_properties(0).total_memory / 1073741824,
    )

    np.seterr(all="raise", under="warn")
    torch.autograd.set_detect_anomaly(True)
    torch.set_warn_always(True)
    torch.set_printoptions(precision=20, threshold=inf)

    set_matmul_precision()
    seed_everything(42, workers=True)

    parser = config_parser()
    args = parser.parse_args()

    model = create_model(args)

    train(args, model)
