import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from torch.utils.data import DataLoader
from models.star import STaR
import torch.nn as nn

import imageio

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
from utils.metrics import get_pose_metrics
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

        self.register_parameter(
            "poses",
            nn.Parameter(torch.zeros((args.num_frames - 1, 6)), requires_grad=True),
        )

        print("self.poses", self.poses)

        self.register_buffer(
            "current_frame_num",
            torch.tensor([args.initial_num_frames], dtype=torch.long),
        )

        self.training_fine_losses = []

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
            # pose = self.gt_poses[batch["frames"][0, 0]]
            pose0 = torch.zeros((1, 6), requires_grad=False, device=self.device)
            poses = torch.cat((pose0, batch["gt_relative_poses"][1:, ...]), dim=0)
            pose = poses[batch["frames"][0]][0]
        elif self.trainer.testing:
            pose = batch["object_pose"]
        else:
            pose0 = torch.zeros((1, 6), requires_grad=False, device=self.device)
            # TODO remove
            """pose0 = self.gt_pose"""
            poses = torch.cat((pose0, self.poses), dim=0)
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

        img_loss = img2mse(result["rgb"], batch["target"])

        loss = img_loss
        psnr = mse2psnr(img_loss)
        img_loss0 = img2mse(result["rgb0"], batch["target"])

        loss = loss + img_loss0
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
        pose_optimizer = torch.optim.Adam([self.poses], lr=self.args.lrate_pose)

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

        val_mse = img2mse(result["rgb"], batch["target"])
        psnr = mse2psnr(val_mse)

        # TODO does this gather for multi gpu setup correctly?
        self.log("val/mse", val_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True)

        # Log visualizations for the first view (see dataset implementation)
        val_H = self.val_dataset.H
        val_W = self.val_dataset.W

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

        log_val_table_online(
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
        )

        # Log pose metrics
        trans_error, rot_error = get_pose_metrics(
            self.poses[0 : self.current_frame_num, ...],
            batch["gt_relative_poses"][1 : self.current_frame_num + 1, ...],
            # self.gt_poses[1 : self.current_frame_num + 1, ...],  # TODO remove
        )
        self.log("val/trans_error", trans_error)
        self.log("val/rot_error", rot_error)

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
        target = to8b(
            torch.reshape(batch["target"], (test_H, test_W, 3)).cpu().detach().numpy(),
            "target",
        )

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
            self.args, "train", self.args.initial_num_frames
        )
        self.val_dataset = StarOnlineDataset(
            self.args, "val", self.args.initial_num_frames
        )
        self.test_dataset = StarOnlineDataset(
            self.args,
            "test",
            self.args.initial_num_frames,  # TODO init num frames needed?
        )

        if self.args.load_gt_poses:
            self.register_buffer(
                "gt_poses", self.train_dataset.get_gt_vehicle_poses(self.args)
            )
        elif self.args.noisy_pose_init:
            with torch.no_grad():
                self.poses += self.train_dataset.get_noisy_gt_relative_poses()[1:].to(
                    self.device
                )

        # TODO remove!
        """with torch.no_grad():
            gt_vehicle_poses = self.train_dataset.get_gt_vehicle_poses(self.args)
            gt_pose = torch.eye(4, dtype=torch.float32)
            gt_pose[:3, :3] = gt_vehicle_poses[0, :3, :3]
            gt_pose[3, :3] = gt_vehicle_poses[0, :3, 3]
            gt_pose = se3_log_map(gt_pose[None, ...])
        print("gt pose", gt_pose)
        self.register_buffer("gt_pose", gt_pose)"""

        # TODO remove
        """with torch.no_grad():
            gt_poses = torch.eye(4, dtype=torch.float32)
            gt_poses = gt_poses.repeat(16, 1, 1)
            gt_poses[:, :3, :3] = gt_vehicle_poses[:, :3, :3]
            gt_poses[:, 3, :3] = gt_vehicle_poses[:, :3, 3]
            gt_poses = se3_log_map(gt_poses)
        print("gt poses", gt_poses)
        self.register_buffer("gt_poses", gt_poses)"""

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


def create_model(args):
    network = STaR(args.num_frames, args)

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
    logger.watch(model, log="all")
    logger.experiment.config.update(args)

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
        # detect_anomaly=True,
        # profiler="simple",
    )

    trainer.fit(model, ckpt_path=args.online_ckpt_path)


if __name__ == "__main__":
    print("torch version", torch.__version__)
    print("torch cuda version", torch.version.cuda)
    print("torch gpu available", torch.cuda.is_available())
    print(
        "gpu memory(in GiB)",
        torch.cuda.get_device_properties(0).total_memory / 1073741824,
    )

    set_matmul_precision()
    seed_everything(42, workers=True)

    parser = config_parser()
    args = parser.parse_args()
    model = create_model(args)

    train(args, model)
