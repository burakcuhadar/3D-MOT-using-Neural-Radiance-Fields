import os
from typing import Any, Dict
import wandb
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from models.nerf_time import NerfTime
import torch.nn as nn
from einops import rearrange
from torch import inf

from utils.optim import get_scheduler
import imageio
import logging

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

from models.rendering__ import sample_pts, render_nerf_time
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

from utils.io import *
from utils.test import test_step_for_one_frame_nerftime


class NerfTimeModule(pl.LightningModule):
    def __init__(self, args, nerf_network):
        super().__init__()

        self.args = args
        self.nerf_network = nerf_network

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

        return render_nerf_time(
            self.nerf_network,
            pts,
            viewdirs,
            z_vals,
            batch["rays_o"],
            batch["rays_d"],
            self.args.N_importance,
            batch["frames"].item(),
        )

    def training_step(self, batch, batch_idx):
        result = self(batch)

        # img_loss = img2mse(result["rgb"], batch["target"])
        img_loss = self.loss(result["rgb"], batch["target"])

        # loss = img_loss
        psnr = mse2psnr(img_loss)
        # img_loss0 = img2mse(result["rgb0"], batch["target"])
        img_loss0 = self.loss(result["rgb0"], batch["target"])

        loss = img_loss + img_loss0
        psnr0 = mse2psnr(img_loss0)

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

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.nerf_network.parameters()),
            lr=self.args.lrate,
            betas=(0.9, 0.999),
        )

        scheduler = get_scheduler(
            optimizer,
            self.args.lrate_decay_rate,
            lrate_decay=self.args.lrate_decay,
            lrate_decay_steps=self.args.lrate_decay_steps,
        )

        return [optimizer], [scheduler]

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

        self.logger.log_image(
            "val/imgs",
            [
                rgb,
                target,
                depth,
                rgb0,
                depth0,
                z_std,
            ],
            step=self.global_step,
        )

    def test_step(self, batch, batch_idx):
        self.test_rgb_gts.append([])
        self.test_rgbs.append([])
        self.test_depths.append([])

        for i in range(self.args.eval_last_frame):
            frame = torch.from_numpy(np.array([i])[:, None]).to(self.device)  # 1,1
            batch["frames"] = frame

            result = self(batch)
            mse = self.loss(result["rgb"], batch["target"][i])
            psnr = mse2psnr(mse)
            gt_dynamic_mask = batch["semantic_mask"][i]
            gt_static_mask = ~gt_dynamic_mask
            psnr_dynamic = mse2psnr(
                img2mse(
                    result["rgb"][gt_dynamic_mask],
                    batch["target"][i][gt_dynamic_mask],
                )
            )
            psnr_static = mse2psnr(
                img2mse(
                    result["rgb"][gt_static_mask],
                    batch["target"][i][gt_static_mask],
                )
            )
            test_H = self.test_dataset.H
            test_W = self.test_dataset.W

            lpips = self.eval_lpips(
                torch.reshape(result["rgb"], (test_H, test_W, 3)),
                torch.reshape(batch["target"][i], (test_H, test_W, 3)),
            )
            rgb_gt_dynamic = batch["target"][i].clone()
            rgb_gt_dynamic[gt_static_mask] = 0
            rgb_pred_dynamic = result["rgb"].clone()
            rgb_pred_dynamic[gt_static_mask] = 0
            lpips_dynamic = self.eval_lpips(
                torch.reshape(rgb_pred_dynamic, (test_H, test_W, 3)),
                torch.reshape(rgb_gt_dynamic, (test_H, test_W, 3)),
            )
            rgb_gt_static = batch["target"][i].clone()
            rgb_gt_static[gt_dynamic_mask] = 0
            rgb_pred_static = result["rgb"].clone()
            rgb_pred_static[gt_dynamic_mask] = 0
            lpips_static = self.eval_lpips(
                torch.reshape(rgb_pred_static, (test_H, test_W, 3)),
                torch.reshape(rgb_gt_static, (test_H, test_W, 3)),
            )

            ssim, ssim_img = self.eval_ssim(
                torch.reshape(result["rgb"], (test_H, test_W, 3)),
                torch.reshape(batch["target"][i], (test_H, test_W, 3)),
                return_full_image=True,
            )
            ssim_dynamic = ssim_img[0, :].reshape(3, -1)[:, gt_dynamic_mask].mean()
            ssim_static = ssim_img[0, :].reshape(3, -1)[:, gt_static_mask].mean()

            self.test_mse_mean += mse
            self.test_psnr_mean += psnr
            self.test_dynamic_psnr_mean += psnr_dynamic
            self.test_static_psnr_mean += psnr_static

            self.test_lpips_mean += lpips
            self.test_static_lpips_mean += lpips_static
            self.test_dynamic_lpips_mean += lpips_dynamic

            self.test_ssim_mean += ssim
            self.test_dynamic_ssim_mean += ssim_dynamic
            self.test_static_ssim_mean += ssim_static

            test_result = test_step_for_one_frame_nerftime(
                self.test_dataset,
                i,
                batch,
                batch_idx,
                result,
            )

            self.test_rgb_gts[-1].append(test_result["rgb_gt"])
            self.test_rgbs[-1].append(test_result["rgb"])
            self.test_depths[-1].append(test_result["depth"])

            self.logger.log_image(
                "test/imgs",
                [
                    test_result["rgb"],
                    test_result["rgb_gt"],
                    test_result["depth"],
                    test_result["rgb0"],
                    test_result["depth0"],
                    to8b(
                        rgb_gt_static.reshape((test_H, test_W, 3))
                        .cpu()
                        .detach()
                        .numpy(),
                        "rgb_gt_static",
                    ),
                    to8b(
                        rgb_pred_static.reshape((test_H, test_W, 3))
                        .cpu()
                        .detach()
                        .numpy(),
                        "rgb_pred_static",
                    ),
                    to8b(
                        rgb_gt_dynamic.reshape((test_H, test_W, 3))
                        .cpu()
                        .detach()
                        .numpy(),
                        "rgb_gt_dynamic",
                    ),
                    to8b(
                        rgb_pred_dynamic.reshape((test_H, test_W, 3))
                        .cpu()
                        .detach()
                        .numpy(),
                        "rgb_dynamic_all",
                    ),
                ],
            )

        self.test_rgb_gts[-1] = np.stack(self.test_rgb_gts[-1], 0)
        self.test_rgbs[-1] = np.stack(self.test_rgbs[-1], 0)
        self.test_depths[-1] = np.stack(self.test_depths[-1], 0)

    def on_test_start(self):
        self.video_basepath = os.path.join("videos", self.logger.version)
        self.test_rgb_gts = []
        self.test_rgbs = []
        self.test_depths = []

        self.test_mse_mean = 0
        self.test_psnr_mean = 0
        self.test_lpips_mean = 0
        self.test_ssim_mean = 0

        self.test_dynamic_psnr_mean = 0
        self.test_static_psnr_mean = 0
        self.test_dynamic_lpips_mean = 0
        self.test_static_lpips_mean = 0
        self.test_dynamic_ssim_mean = 0
        self.test_static_ssim_mean = 0

    def on_test_end(self):
        eval_len = self.test_dataset.imgs.shape[0] * self.args.eval_last_frame
        self.test_mse_mean /= eval_len
        self.test_psnr_mean /= eval_len
        self.test_lpips_mean /= eval_len
        self.test_ssim_mean /= eval_len

        self.test_dynamic_psnr_mean /= eval_len
        self.test_dynamic_lpips_mean /= eval_len
        self.test_dynamic_ssim_mean /= eval_len

        self.test_static_psnr_mean /= eval_len
        self.test_static_lpips_mean /= eval_len
        self.test_static_ssim_mean /= eval_len

        print("test/mse", self.test_mse_mean)
        print("test/psnr", self.test_psnr_mean)
        print("test/lpips", self.test_lpips_mean)
        print("test/ssim", self.test_ssim_mean)

        print("test/dynamic_psnr", self.test_dynamic_psnr_mean)
        print("test/dynamic_lpips", self.test_dynamic_lpips_mean)
        print("test/dynamic_ssim", self.test_dynamic_ssim_mean)

        print("test/static_psnr", self.test_static_psnr_mean)
        print("test/static_lpips", self.test_static_lpips_mean)
        print("test/static_ssim", self.test_static_ssim_mean)

        return  # TODO move below to another file/func
        fps = 10
        quality = 8

        # Creating video only for this view
        view = 0  # TODO determine this

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
            self.args.num_frames,
            self.args.num_vehicles,
        )
        self.val_dataset = StarOnlineDataset(
            self.args,
            "val",
            self.args.num_frames,
            self.args.num_frames,
            self.args.num_vehicles,
        )
        self.test_dataset = StarOnlineDataset(
            self.args,
            "test",
            self.args.num_frames,
            self.args.eval_last_frame,
            self.args.num_vehicles,
        )

        # Init SSIM metric
        self.val_ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0, return_full_image=True
        )
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

    def eval_ssim(self, pred_img, gt_img, return_full_image=False):
        # Rearrange for torchmetrics library
        pred_img_rearranged = rearrange(pred_img[None, ...], "B H W C -> B C H W")
        gt_img_rearranged = rearrange(gt_img[None, ...], "B H W C -> B C H W")

        self.val_ssim(pred_img_rearranged, gt_img_rearranged)
        ssim, ssim_img = self.val_ssim.compute()
        self.val_ssim.reset()

        if return_full_image:
            return ssim, ssim_img

        return ssim


def create_model(args):
    network = NerfTime(args)
    model = NerfTimeModule(args, network)
    # model = torch.compile(model)

    return model


def train(args, model):
    logger = WandbLogger(project=args.expname)
    logger.watch(model, log="all")
    logger.experiment.config.update(args)
    logger.experiment.log_code(root=args.code_dir)

    configure_logger(f"logs/nerftime/{logger.version}", "training.log")
    logging.info(f"Slurm job id is {args.job_id}")

    print("Saving checkpoints at:", f"ckpts/nerftime/{logger.version}")
    ckpt_cb = ModelCheckpoint(
        dirpath=f"ckpts/nerftime/{logger.version}",
        filename="{epoch:d}",
        every_n_epochs=args.epoch_val,
        save_on_train_epoch_end=True,
        save_top_k=-1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1), lr_monitor]

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

    callbacks = [TQDMProgressBar(refresh_rate=1)]

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
