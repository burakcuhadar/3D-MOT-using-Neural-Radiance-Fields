import os
import wandb
import numpy as np
import torch
import torch._dynamo
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from utils.logging import log_val_table_app_init_nerfacc
from torch.utils.data import DataLoader
from models.star_nerfacc import STaR
from nerfacc.estimators.occ_grid import OccGridEstimator

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything

from models.rendering import render_image_with_occgrid, Rays
from models.rendering import mse2psnr, img2mse, to8b
from models.loss import compute_sigma_loss
from datasets.carla_star_app_init_nerfacc import StarAppInitDatasetNerfacc

from utils.visualization import visualize_depth
from utils.io import *



def get_scheduler(args, optimizer):
    if args.lrate_decay_steps:
        return MultiStepLR(
            optimizer, milestones=args.lrate_decay_steps, gamma=args.lrate_decay_rate
        )
    else:
        return StepLR(
            optimizer, step_size=args.lrate_decay, gamma=args.lrate_decay_rate
        )


class StarAppInit(pl.LightningModule):
    def __init__(self, args, star_network):
        super().__init__()

        self.args = args
        self.star_network = star_network


        self.register_buffer(
            "aabb", 
            torch.tensor([-1., -1., -1., 1., 1., 1.])
        )
        """self.register_buffer(
            "aabb", 
            torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
        )"""
        """self.register_buffer(
            "aabb", 
            torch.tensor([-2., -2., -2., 2., 2., 2.])
        )"""
        self.args.render_step_size = ((self.aabb[3:] - self.aabb[:3]) ** 2).sum().sqrt().item() / 1000
        self.estimator = OccGridEstimator(
            roi_aabb=self.aabb, resolution=args.grid_resolution, levels=args.grid_nlvl
        )

    def forward(self, batch):
        def occ_eval_fn(x):
            density = self.star_network.static_nerf.query_density(x)
            return density * self.args.render_step_size 

        if self.training:
            # update occupancy grid
            self.estimator.update_every_n_steps(
                step=self.global_step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2, #TODO finetune?
            )

        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            self.star_network.static_nerf,
            self.estimator,
            Rays(batch["rays_o"], batch["rays_d"]),
            # rendering options
            near_plane=0.0,
            far_plane=1e10,
            render_step_size=self.args.render_step_size,
            render_bkgd=None,
            test_chunk_size=self.args.chunk
        )
        
        return rgb, acc, depth, n_rendering_samples

    def training_step(self, batch, batch_idx):        
        rgb, acc, depth, n_rendering_samples = self(batch)
        
        if n_rendering_samples == 0:
            return None

        if self.args.target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = batch["rays_o"].shape[0]
            num_rays = int(
                num_rays * (self.args.target_sample_batch_size / float(n_rendering_samples))
            )
            self.train_dataset.update_num_rays(num_rays)

        loss = F.smooth_l1_loss(rgb, batch["target"]) #TODO also try the usual loss

        mse_loss = img2mse(rgb, batch["target"])
        psnr = mse2psnr(mse_loss)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/psnr", psnr, on_step=False, on_epoch=True)
        self.log("train/mse_loss", mse_loss, on_step=False, on_epoch=True)

        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.star_network.parameters(),
            lr=self.args.lrate,
            betas=(0.9, 0.999),
        )
        scheduler = get_scheduler(self.args, optimizer)

        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        rgb, acc, depth, _ = self(batch)

        val_mse = img2mse(rgb, batch["target"])
        psnr = mse2psnr(val_mse)

        self.log("val/mse", val_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True)

        # Log visualizations for a random view (see dataset implementation)
        val_H = self.val_dataset.H
        val_W = self.val_dataset.W

        rgb = to8b(
            torch.reshape(rgb, (val_H, val_W, 3)).cpu().detach().numpy(),
            "rgb",
        )
        depth = visualize_depth(depth, app_init=True).reshape((val_H, val_W, 3))
        target = to8b(
            torch.reshape(batch["target"], (val_H, val_W, 3)).cpu().detach().numpy(),
            "target",
        )

        log_val_table_app_init_nerfacc(
            self.logger,
            self.current_epoch,
            rgb,
            target,
            depth,
        )

    def setup(self, stage):
        self.train_dataset = StarAppInitDatasetNerfacc(self.args, split="train")
        self.val_dataset = StarAppInitDatasetNerfacc(self.args, split="val")

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


def train():
    parser = config_parser()
    args = parser.parse_args()

    network = STaR(args)
    model = StarAppInit(args, network)
    # model = torch.compile(model)

    logger = WandbLogger(project=args.expname)
    logger.watch(model, log="all")
    logger.experiment.config.update(args)

    configure_logger(f"logs/appinit/{logger.version}", "training.log")
    logging.info(f"Slurm job id is {args.job_id}")

    ckpt_cb = ModelCheckpoint(
        dirpath=f"ckpts/appinit/{logger.version}",
        filename="{epoch:d}",
        every_n_epochs=args.epoch_ckpt,
        save_on_train_epoch_end=True,
        save_top_k=-1,
    )

    early_stopping_cb = EarlyStopping(
        monitor="train/mse_loss",
        mode="min",
        stopping_threshold=args.appearance_init_thres,
    )

    callbacks = [
        ckpt_cb,
        TQDMProgressBar(refresh_rate=1),
        early_stopping_cb,
    ]

    trainer = Trainer(
        max_epochs=args.epochs_appearance,
        check_val_every_n_epoch=args.epoch_val,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=False,
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,
        precision=16 if args.mixed_precision else 32,
        detect_anomaly=False,  # NOTE: disable for faster training?
    )

    trainer.fit(model, ckpt_path=args.appearance_ckpt_path)


if __name__ == "__main__":
    print("torch version", torch.__version__)
    print("torch cuda version", torch.version.cuda)
    set_matmul_precision()
    seed_everything(42, workers=True)
    train()
    print("done.")