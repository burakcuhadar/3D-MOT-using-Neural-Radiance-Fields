import os
from typing import Any, Dict
import wandb
import numpy as np
import torch
import pytorch_lightning as pl
import pypose as pp
from torch.utils.data import DataLoader
from models.star_mipnerf import STaR
import torch.nn as nn
from einops import rearrange
from torch import inf

from utils.optim import get_scheduler
import imageio
import logging

from pytorch3d.transforms import se3_log_map
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import StructuralSimilarityIndexMeasure


from pytorch_lightning.utilities import grad_norm

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning import Trainer, seed_everything

from models.rendering__ import render_star_online, sample_pts
from models.rendering__ import mse2psnr, img2mse, to8b
from models.loss import compute_sigma_loss, compute_depth_loss
from datasets.carla_star_online__ import StarOnlineDataset
from optimizer.hybrid_optimizer import HybridOptim, HybridLRS
from callbacks.online_training_callback import StarOnlineCallback
from utils.logging__ import (
    log_val_table_online,
    log_test_table_online,
    log_2d_iou,
    log_3d_iou,
)
from utils.visualization import visualize_depth
from utils.metrics import get_pose_metrics_multi
from utils.io import *
from utils.test import test_step_for_one_frame


class StarOnline(pl.LightningModule):
    def __init__(self, args, star_network: STaR):
        super().__init__()

        self.args = args
        self.star_network = star_network

        """
        self.register_parameter(
            "poses",
            nn.Parameter(torch.zeros((args.num_vehicles, args.num_frames - 1, 6)), requires_grad=True),
        )
        """

        self.poses = nn.ParameterList(
            [
                nn.Parameter(pp.identity_SE3(args.num_vehicles).tensor().unsqueeze(0))
                for i in range(args.num_frames - 1)
            ]
        )

        print("self.poses", self.poses)

        self.register_buffer(
            "current_frame_num",
            torch.tensor([args.initial_num_frames], dtype=torch.long),
        )

        self.register_buffer("start_frame", torch.tensor([0], dtype=torch.long))

        self.running_fine_loss = 0

        self.loss = nn.MSELoss()

    def forward(self, batch):
        viewdirs = batch["rays_d"] / torch.norm(
            batch["rays_d"], dim=-1, keepdim=True
        )  # [N_rays, 3]

        if self.args.load_gt_poses:
            pose = self.gt_poses_quat[:, batch["frames"][0], ...][:, 0]
        else:
            pose0 = (
                pp.identity_SE3(self.args.num_vehicles, device=self.device)
                .tensor()
                .unsqueeze(0)
            )  # 1 2 7

            poses = torch.cat(
                ([pose0] + [posei for posei in self.poses]), dim=0
            )  # 15 2 7

            pose = poses[batch["frames"][0], :, ...][0]  # 2 7

        return self.star_network(batch["rays_o"], viewdirs, pose)

    def training_step(self, batch, batch_idx):
        result = self(batch)

        # img_loss = img2mse(result["rgb"], batch["target"])
        img_loss = self.loss(result["rgb"], batch["target"])

        # loss = img_loss
        psnr = mse2psnr(img_loss)
        # img_loss0 = img2mse(result["rgb0"], batch["target"])
        img_loss0 = self.loss(result["rgb0"], batch["target"])

        loss = img_loss + 0.1 * img_loss0
        psnr0 = mse2psnr(img_loss0)

        # entropy_loss = (result["entropy"] + result["entropy0"]) / 2.0
        # loss += self.args.entropy_weight * entropy_loss

        """
        "loss_alpha_entropy": loss_alpha_entropy,
        "loss_dynamic_vs_static_reg": loss_dynamic_vs_static_reg,
        "loss_ray_reg": loss_ray_reg,
        "loss_static_reg": loss_static_reg,
        """

        if self.args.lambda_alpha_entropy > 0:
            loss_alpha_entropy = (
                result["loss_alpha_entropy"] + result["loss_alpha_entropy0"]
            ) / 2.0
            loss += args.lambda_alpha_entropy * loss_alpha_entropy
            self.log(
                "train/alpha_entropy",
                loss_alpha_entropy,
                on_step=False,
                on_epoch=True,
                batch_size=self.args.N_rand,
            )

        if self.args.lambda_dynamic_vs_static_reg > 0:
            loss_dynamic_vs_static_reg = (
                result["loss_dynamic_vs_static_reg"]
                + result["loss_dynamic_vs_static_reg0"]
            ) / 2.0
            loss += args.lambda_dynamic_vs_static_reg * loss_dynamic_vs_static_reg
            self.log(
                "train/dynamic_vs_static_reg",
                loss_dynamic_vs_static_reg,
                on_step=False,
                on_epoch=True,
                batch_size=self.args.N_rand,
            )

        if self.args.lambda_ray_reg > 0:
            loss_ray_reg = (result["loss_ray_reg"] + result["loss_ray_reg0"]) / 2.0
            loss += args.lambda_ray_reg * loss_ray_reg
            self.log(
                "train/ray_reg",
                loss_ray_reg,
                on_step=False,
                on_epoch=True,
                batch_size=self.args.N_rand,
            )

        if self.args.lambda_static_reg > 0:
            loss_lambda_static_reg = (
                result["loss_static_reg"] + result["loss_static_reg0"]
            ) / 2.0
            loss += self.args.lambda_static_reg * loss_lambda_static_reg
            self.log(
                "train/static_reg",
                loss_lambda_static_reg,
                on_step=False,
                on_epoch=True,
                batch_size=self.args.N_rand,
            )

        if (
            self.args.lambda_dynamic_reg > 0
            and self.args.epoch_start_dynamic_reg <= self.current_epoch
        ):
            loss_lambda_dynamic_reg = (
                result["loss_dynamic_reg"] + result["loss_dynamic_reg0"]
            ) / 2.0
            loss += self.args.lambda_dynamic_reg * loss_lambda_dynamic_reg
            self.log(
                "train/dynamic_reg",
                loss_lambda_dynamic_reg,
                on_step=False,
                on_epoch=True,
                batch_size=self.args.N_rand,
            )

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

        self.running_fine_loss += img_loss.detach()

        self.log(
            "train/fine_loss",
            img_loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.N_rand,
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.N_rand,
        )
        self.log(
            "train/psnr",
            psnr,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.N_rand,
        )
        self.log(
            "train/psnr0",
            psnr0,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.N_rand,
        )
        # self.log("train/entropy_loss", entropy_loss, on_step=False, on_epoch=True, batch_size=self.args.N_rand)

        if self.args.depth_loss:
            self.log(
                "train/depth_loss",
                depth_loss,
                on_step=False,
                on_epoch=True,
                batch_size=self.args.N_rand,
            )
        if self.args.sigma_loss:
            self.log(
                "train/sigma_loss",
                sigma_loss,
                on_step=False,
                on_epoch=True,
                batch_size=self.args.N_rand,
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.star_network.static_nerf.parameters()),
                    "lr": self.args.lrate_static,
                },
                {
                    "params": list(self.star_network.dynamic_nerfs.parameters()),
                    "lr": self.args.lrate_dynamic,
                },
            ],
            lr=self.args.lrate,
            betas=(0.9, 0.999),
        )
        pose_optimizer = torch.optim.Adam(self.poses, lr=self.args.lrate_pose)

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

        """
        if self.lr_schedulers() is None:
            logging.info("config optimizers first call")

            optimizer = torch.optim.Adam(
                [
                    {
                        "params": list(
                            self.star_network.static_coarse_nerf.parameters()
                        )
                        + list(self.star_network.static_fine_nerf.parameters()),
                        "lr": self.args.lrate_static,
                    },
                    {
                        "params": list(
                            self.star_network.dynamic_coarse_nerfs.parameters()
                        )
                        + list(self.star_network.dynamic_fine_nerfs.parameters()),
                        "lr": self.args.lrate_dynamic,
                    },
                ],
                lr=self.args.lrate,
                betas=(0.9, 0.999),
            )
            pose_optimizer = torch.optim.Adam(
                self.poses[: self.current_frame_num[0]], lr=self.args.lrate_pose
            )

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
        else:
            logging.info("config optimizers new call")
            lr_schedulers = self.lr_schedulers()

            print("last lr 0 ", lr_schedulers[0].get_last_lr()[0])
            print("last lr 1 ", lr_schedulers[0].get_last_lr()[1])
            print("pose last lr  ", lr_schedulers[1].get_last_lr()[0])
            print("lr sched last epoch", lr_schedulers[0].last_epoch)
            print("pose lr sched last epoch", lr_schedulers[1].last_epoch)

            optimizer = torch.optim.Adam(
                [
                    {
                        "params": list(
                            self.star_network.static_coarse_nerf.parameters()
                        )
                        + list(self.star_network.static_fine_nerf.parameters()),
                        "lr": lr_schedulers[0].get_last_lr()[0],
                        "initial_lr": self.args.lrate_static,
                    },
                    {
                        "params": list(
                            self.star_network.dynamic_coarse_nerfs.parameters()
                        )
                        + list(self.star_network.dynamic_fine_nerfs.parameters()),
                        "lr": lr_schedulers[0].get_last_lr()[1],
                        "initial_lr": self.args.lrate_dynamic,
                    },
                ],
                betas=(0.9, 0.999),
            )

            pose_optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.poses[self.start_frame[0] :],
                        "lr": lr_schedulers[1].get_last_lr()[0],
                        "initial_lr": self.args.lrate_pose,
                    }
                ]
            )

            scheduler = get_scheduler(
                optimizer,
                self.args.lrate_decay_rate,
                lrate_decay=self.args.lrate_decay,
                lrate_decay_steps=self.args.lrate_decay_steps,
                last_epoch=lr_schedulers[0].last_epoch,
            )

            pose_scheduler = get_scheduler(
                pose_optimizer,
                self.args.pose_lrate_decay_rate,
                lrate_decay=self.args.pose_lrate_decay,
                lrate_decay_steps=self.args.pose_lrate_decay_steps,
                last_epoch=lr_schedulers[1].last_epoch,
            )

            hoptim = HybridOptim([optimizer, pose_optimizer])

            return [hoptim], [
                HybridLRS(hoptim, 0, scheduler),
                HybridLRS(hoptim, 1, pose_scheduler),
            ]
        """

    def validation_step(self, batch, batch_idx):
        result = self(batch)

        val_H = self.val_dataset.H
        val_W = self.val_dataset.W

        self.log("val/frame", batch["frames"][0, 0])

        # val_mse = img2mse(result["rgb"], batch["target"])
        val_mse = self.loss(result["rgb"], batch["target"])
        psnr = mse2psnr(val_mse)
        lpips = self.eval_lpips(
            torch.reshape(result["rgb"], (val_H, val_W, 3)),
            torch.reshape(batch["target"], (val_H, val_W, 3)),
        )
        ssim = self.eval_ssim(
            torch.reshape(result["rgb"], (val_H, val_W, 3)),
            torch.reshape(batch["target"], (val_H, val_W, 3)),
        )

        self.log("val/mse", val_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True)
        self.log("val/lpips", lpips, on_step=False, on_epoch=True)
        self.log("val/ssim", ssim, on_step=False, on_epoch=True)

        # Log visualizations for the first view (see dataset implementation)
        rgb0 = to8b(
            torch.reshape(result["rgb0"], (val_H, val_W, 3)).cpu().detach().numpy(),
            "rgb0",
        )
        depth0 = visualize_depth(result["depth0"]).reshape((val_H, val_W, 3))
        # z_std = to8b(
        #     torch.reshape(result["z_std"], (val_H, val_W, 1)).cpu().detach().numpy(),
        #     "z_std",
        # )
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
        rgb_dynamic0s = to8b(
            result["rgb_dynamic0"]
            .transpose(0, 1)
            .reshape((self.args.num_vehicles, val_H, val_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_dynamic0s",
        )
        rgb_static = to8b(
            torch.reshape(result["rgb_static"], (val_H, val_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_static",
        )
        rgb_dynamics = to8b(
            result["rgb_dynamic"]
            .transpose(0, 1)
            .reshape((self.args.num_vehicles, val_H, val_W, 3))
            .cpu()
            .detach()
            .numpy(),
            "rgb_dynamics",
        )
        depth_static = visualize_depth(result["depth_static"]).reshape(
            (val_H, val_W, 3)
        )
        depth_dynamics = visualize_depth(
            result["depth_dynamic"].transpose(0, 1),
            val_H,
            val_W,
            multi_vehicle=True,
        )
        depth_static0 = visualize_depth(result["depth_static0"]).reshape(
            (val_H, val_W, 3)
        )
        depth_dynamic0s = visualize_depth(
            result["depth_dynamic0"].transpose(0, 1),
            val_H,
            val_W,
            multi_vehicle=True,
        )

        self.logger.log_image(
            "val/imgs",
            [
                rgb,
                target,
                *rgb_dynamics,
                rgb_static,
                depth,
                *depth_dynamics,
                depth_static,
                rgb0,
                *rgb_dynamic0s,
                rgb_static0,
                depth0,
                *depth_dynamic0s,
                depth_static0,
                # z_std,
            ],
            step=self.global_step,
        )

        # Log pose metrics
        (
            trans_errors,
            rot_errors,
            last_trans_errors,
            last_rot_errors,
            rot_error_eulers,
            last_rot_error_eulers,
        ) = get_pose_metrics_multi(
            # self.poses[:, 0 : self.current_frame_num - 1, ...],
            torch.cat(list(self.poses[0 : self.current_frame_num - 1]), dim=0),
            batch["gt_relative_poses"][:, 1 : self.current_frame_num, ...].transpose(
                0, 1
            ),
        )
        for i in range(self.args.num_vehicles):
            self.log(f"val/trans_error{i}", trans_errors[i])
            self.log(f"val/rot_error{i}", rot_errors[i])
            self.log(f"val/last_trans_error{i}", last_trans_errors[i])
            self.log(f"val/last_rot_error{i}", last_rot_errors[i])
            self.log(f"val/euler_rot_error{i}", rot_error_eulers[i])
            self.log(f"val/last_euler_rot_error{i}", last_rot_error_eulers[i])

        """TODO adapt for multi vehicle
        rpe_trans_rmse, rpe_rot_rmse = evaluate_rpe(
            self.poses[: self.current_frame_num - 1],
            self.train_dataset.gt_relative_poses_matrices[
                1 : self.current_frame_num
            ].to(self.device),
        )
        self.log("val/rpe_trans_rmse", rpe_trans_rmse)
        self.log("val/rpe_rot_rmse", rpe_rot_rmse)
        """

        for pose in self.poses:
            print(pose.data, "\n")

    def test_step(self, batch, batch_idx):
        self.test_rgb_gts.append([])
        self.test_rgbs.append([])
        self.test_depths.append([])
        self.test_rgb_statics.append([])
        self.test_rgb_dynamics.append([])
        self.test_depths_statics.append([])
        self.test_depths_dynamics.append([])

        for i in range(self.args.eval_last_frame):
            frame = torch.from_numpy(np.array([i])[:, None]).to(self.device)  # 1,1
            batch["frames"] = frame

            result = self(batch)
            mse = self.loss(result["rgb"], batch["target"][i])
            psnr = mse2psnr(mse)
            test_H = self.test_dataset.H
            test_W = self.test_dataset.W
            lpips = self.eval_lpips(
                torch.reshape(result["rgb"], (test_H, test_W, 3)),
                torch.reshape(batch["target"][i], (test_H, test_W, 3)),
            )
            ssim = self.eval_ssim(
                torch.reshape(result["rgb"], (test_H, test_W, 3)),
                torch.reshape(batch["target"][i], (test_H, test_W, 3)),
            )

            self.test_mse_mean += mse
            self.test_psnr_mean += psnr
            self.test_lpips_mean += lpips
            self.test_ssim_mean += ssim

            test_result = test_step_for_one_frame(
                self.test_dataset,
                i,
                batch,
                batch_idx,
                result,
                self.args.num_vehicles,
                pp.SE3(self.poses[i - 1][0]).matrix().cpu().numpy()
                if i != 0
                else np.repeat(
                    np.eye(4, dtype=np.float32)[np.newaxis, :, :],
                    self.args.num_vehicles,
                    axis=0,
                ),
                self.test_dataset.gt_vehicle_poses[:, i].numpy(),
                self.test_dataset.bbox_local_vertices
                if batch_idx == self.args.bbox_view
                else None,
                self.test_dataset.gt_vehicle_poses[:, 0].numpy(),
            )

            self.test_rgb_gts[-1].append(test_result["rgb_gt"])
            self.test_rgbs[-1].append(test_result["rgb"])
            self.test_depths[-1].append(test_result["depth"])
            self.test_rgb_statics[-1].append(test_result["rgb_static"])
            self.test_rgb_dynamics[-1].append(test_result["rgb_dynamics"])
            self.test_depths_statics[-1].append(test_result["depth_static"])
            self.test_depths_dynamics[-1].append(test_result["depth_dynamics"])

            semantic_mask_count = (
                torch.count_nonzero(batch["semantic_mask"][i]).cpu().item()
            )
            if semantic_mask_count > 0:
                self.test_iou2d += test_result["iou_2d"]
                self.test_iou2d_count += 1
                log_2d_iou(
                    test_result["iou_2d"],
                    test_result["predicted_masks"],
                    batch["semantic_mask"][i],
                    test_result["rgb_gt"],
                    i,
                    batch_idx,
                )

            if self.args.has_bbox and batch_idx == self.args.bbox_view:
                self.test_iou3d += test_result["iou_3d"]
                log_3d_iou(
                    test_result["iou_3d"],
                    test_result["bboxes"],
                    test_result["gt_bboxes"],
                    test_result["rgb_gt"],
                    self.test_dataset.K,
                    batch["cam_pose"],
                )

            self.logger.log_image(
                "test/imgs",
                [
                    test_result["rgb"],
                    test_result["rgb_gt"],
                    *test_result["rgb_dynamics"],
                    test_result["rgb_static"],
                    test_result["depth"],
                    *test_result["depth_dynamics"],
                    test_result["depth_static"],
                    test_result["rgb0"],
                    *test_result["rgb_dynamic0s"],
                    test_result["rgb_static0"],
                    test_result["depth0"],
                    *test_result["depth_dynamic0s"],
                    test_result["depth_static0"],
                ],
            )

        self.test_rgb_gts[-1] = np.stack(self.test_rgb_gts[-1], 0)
        self.test_rgbs[-1] = np.stack(self.test_rgbs[-1], 0)
        self.test_depths[-1] = np.stack(self.test_depths[-1], 0)
        self.test_rgb_statics[-1] = np.stack(self.test_rgb_statics[-1], 0)
        self.test_rgb_dynamics[-1] = np.stack(
            self.test_rgb_dynamics[-1], 0
        )  # num_vehicles, num_frames, H, W, 3
        self.test_depths_statics[-1] = np.stack(self.test_depths_statics[-1], 0)
        self.test_depths_dynamics[-1] = np.stack(
            self.test_depths_dynamics[-1], 0
        )  # num_vehicles, num_frames, H, W, 3

    def on_test_start(self):
        self.video_basepath = os.path.join("videos", self.logger.version)
        self.test_rgb_gts = []
        self.test_rgbs = []
        self.test_depths = []
        self.test_rgb_statics = []
        self.test_rgb_dynamics = []
        self.test_depths_statics = []
        self.test_depths_dynamics = []

        self.test_mse_mean = 0
        self.test_psnr_mean = 0
        self.test_lpips_mean = 0
        self.test_ssim_mean = 0
        self.test_iou3d = np.zeros(self.args.num_vehicles)
        self.test_iou2d = 0
        self.test_iou2d_count = 0

    def on_test_end(self):
        self.test_mse_mean /= self.test_dataset.imgs.shape[0]
        self.test_psnr_mean /= self.test_dataset.imgs.shape[0]
        self.test_lpips_mean /= self.test_dataset.imgs.shape[0]
        self.test_ssim_mean /= self.test_dataset.imgs.shape[0]

        self.log("test/mse", self.test_mse_mean)
        self.log("test/psnr", self.test_psnr_mean)
        self.log("test/lpips", self.test_lpips_mean)
        self.log("test/ssim", self.test_ssim_mean)

        iou_2d = self.test_iou2d / self.test_iou2d_count
        self.log("test/iou_2d", iou_2d)

        if self.args.has_bbox:
            iou_3d = self.test_iou3d / (
                self.args.eval_last_frame - 1
            )  # -1, since for the first frame we use gt pose
            self.log("test/iou_3d", iou_3d)

        return  # TODO delete
        fps = 10
        quality = 8

        # Creating video only for this view
        view = 0

        self.test_rgb_gts = np.stack(self.test_rgb_gts, 0)
        self.test_rgbs = np.stack(self.test_rgbs, 0)
        self.test_depths = np.stack(self.test_depths, 0)
        self.test_rgb_statics = np.stack(self.test_rgb_statics, 0)
        self.test_rgb_dynamics = np.stack(
            self.test_rgb_dynamics, 0
        )  # num_views, num_frames, num_vehicles, H, W, 3
        self.test_depths_statics = np.stack(self.test_depths_statics, 0)
        self.test_depths_dynamics = np.stack(
            self.test_depths_dynamics, 0
        )  # num_views, num_frames, num_vehicles , H, W, 3

        rgb_gt_path = self.video_basepath + "rgb_gt.mp4"
        rgb_path = self.video_basepath + "rgb.mp4"
        depth_path = self.video_basepath + "depth.mp4"
        rgb_static_path = self.video_basepath + "rgb_static.mp4"
        depth_static_path = self.video_basepath + "depth_static.mp4"
        imageio.mimwrite(rgb_gt_path, self.test_rgb_gts[view], fps=fps, quality=quality)
        imageio.mimwrite(rgb_path, self.test_rgbs[view], fps=fps, quality=quality)
        imageio.mimwrite(depth_path, self.test_depths[view], fps=fps, quality=quality)
        imageio.mimwrite(
            rgb_static_path, self.test_rgb_statics[view], fps=fps, quality=quality
        )
        imageio.mimwrite(
            depth_static_path, self.test_depths_statics[view], fps=fps, quality=quality
        )

        wandb.log({"test/rgb_gt": wandb.Video(rgb_gt_path, fps=fps, format="mp4")})
        wandb.log({"test/rgb": wandb.Video(rgb_path, fps=fps, format="mp4")})
        wandb.log({"test/depth": wandb.Video(depth_path, fps=fps, format="mp4")})
        wandb.log(
            {"test/rgb_static": wandb.Video(rgb_static_path, fps=fps, format="mp4")}
        )
        wandb.log(
            {"test/depth_static": wandb.Video(depth_static_path, fps=fps, format="mp4")}
        )

        for i in range(self.args.num_vehicles):
            rgb_dynamic_path = self.video_basepath + f"rgb_dynamic{i}.mp4"
            depth_dynamic_path = self.video_basepath + f"depth_dynamic{i}.mp4"

            imageio.mimwrite(
                rgb_dynamic_path,
                self.test_rgb_dynamics[view, :, i, ...],
                fps=fps,
                quality=quality,
            )
            imageio.mimwrite(
                depth_dynamic_path,
                self.test_depths_dynamics[view, :, i, ...],
                fps=fps,
                quality=quality,
            )

            wandb.log(
                {
                    f"test/rgb_dynamic{i}": wandb.Video(
                        rgb_dynamic_path, fps=fps, format="mp4"
                    )
                }
            )
            wandb.log(
                {
                    f"test/depth_dynamic{i}": wandb.Video(
                        depth_dynamic_path, fps=fps, format="mp4"
                    )
                }
            )

    def setup(self, stage):
        self.train_dataset = StarOnlineDataset(
            self.args,
            "train",
            self.args.num_frames,
            self.args.initial_num_frames,
            self.args.num_vehicles,
        )
        self.val_dataset = StarOnlineDataset(
            self.args,
            "val",
            self.args.num_frames,
            self.args.initial_num_frames,
            self.args.num_vehicles,
        )
        self.test_dataset = StarOnlineDataset(
            self.args,
            "test",
            self.args.num_frames,
            self.args.eval_last_frame,
            self.args.num_vehicles,
        )

        if self.args.load_gt_poses:
            self.register_buffer(
                "gt_poses", self.train_dataset.get_gt_vehicle_poses(self.args)
            )
            self.register_buffer("gt_poses_quat", self.train_dataset.gt_relative_poses)
        elif self.args.noisy_pose_init and (self.args.online_ckpt_path is None):
            with torch.no_grad():
                noisy_poses = self.train_dataset.get_noisy_gt_relative_poses()[
                    :, 1:
                ].to(self.device)

                for frame_idx in range(noisy_poses.shape[1]):
                    self.poses[frame_idx].copy_(noisy_poses[:, frame_idx, ...])

                trans_error, rot_error, _, _, _, _ = get_pose_metrics_multi(
                    torch.cat(list(self.poses), dim=0),
                    self.train_dataset.gt_relative_poses[:, 1:, ...]
                    .transpose(0, 1)
                    .to(self.device),
                    reduce=False,
                )

                print("total trans errors", trans_error)
                print("total rot errors", rot_error)

        # Init SSIM metric
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        # Init LPIPS metric
        self.val_lpips = LearnedPerceptualImagePatchSimilarity("vgg", normalize=True)
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

        self.val_lpips(
            torch.clip(pred_img_rearranged, 0, 1), torch.clip(gt_img_rearranged, 0, 1)
        )
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

    if args.appearance_ckpt_path and (args.online_ckpt_path is None):
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
    logger.experiment.log_code(root=args.code_dir)

    configure_logger(f"logs/online/{logger.version}", "training.log")
    logging.info(f"Slurm job id is {args.job_id}")

    print("Saving checkpoints at:", f"ckpts/online/{logger.version}")
    ckpt_cb = ModelCheckpoint(
        dirpath=f"ckpts/online/{logger.version}",
        filename="{epoch:d}",
        every_n_epochs=args.epoch_val,
        save_on_train_epoch_end=True,
        save_top_k=-1,
    )

    star_online_cb = StarOnlineCallback(args)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1), star_online_cb, lr_monitor]

    trainer = Trainer(
        max_epochs=args.epochs_online,
        check_val_every_n_epoch=args.epoch_val,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=False,
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=1.0,
        detect_anomaly=True,
        # profiler="simple",
    )

    trainer.fit(model, ckpt_path=args.online_ckpt_path)


def test(args, model: pl.LightningModule):
    """model.load_from_checkpoint(
        args.online_ckpt_path, strict=False, args=args, star_network=model.star_network
    )"""

    logger = WandbLogger(project=args.expname)
    logger.watch(model, log="all")
    logger.experiment.config.update(args)
    logger.experiment.log_code(root=args.code_dir)

    configure_logger(f"logs/online/{logger.version}", "training.log")
    logging.info(f"Slurm job id is {args.job_id}")

    star_online_cb = StarOnlineCallback(args)

    callbacks = [TQDMProgressBar(refresh_rate=1), star_online_cb]

    trainer = Trainer(
        max_epochs=args.epochs_online,
        check_val_every_n_epoch=args.epoch_val,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=False,
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=1.0,
        # detect_anomaly=True,
        # profiler="simple",
    )

    trainer.test(model, ckpt_path=args.online_ckpt_path)


if __name__ == "__main__":
    print("torch version", torch.__version__)
    print("torch cuda version", torch.version.cuda)
    print("torch gpu available", torch.cuda.is_available())
    print(
        "gpu memory(in GiB)",
        torch.cuda.get_device_properties(0).total_memory / 1073741824,
    )

    np.seterr(all="raise", under="warn")
    torch.set_warn_always(True)
    torch.set_printoptions(precision=20, threshold=inf)

    set_matmul_precision()
    # seed_everything(42, workers=True)
    seed_everything(1453, workers=True)

    parser = config_parser()
    args = parser.parse_args()

    model = create_model(args)

    if args.test:
        test(args, model)
    else:
        train(args, model)
