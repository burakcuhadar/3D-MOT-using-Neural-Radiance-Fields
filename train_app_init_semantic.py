import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from utils.logging import log_val_table_app_init_semantic
from torch.utils.data import DataLoader
from models.star import STaR

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything

from models.rendering import sample_pts, render_star_appinit_semantic
from models.rendering import mse2psnr, img2mse, to8b
from datasets.carla_star_app_init_semantic import StarAppInitSemanticDataset
from utils.visualization import visualize_depth, combine_static_dynamic
from utils.io import *


torch.autograd.set_detect_anomaly(True)


def get_scheduler(args, optimizer):
    if args.lrate_decay_steps:
        return MultiStepLR(
            optimizer, milestones=args.lrate_decay_steps, gamma=args.lrate_decay_rate
        )
    else:
        return StepLR(
            optimizer, step_size=args.lrate_decay, gamma=args.lrate_decay_rate
        )


class StarAppInitSemantic(pl.LightningModule):
    def __init__(self, args, star_network):
        super().__init__()

        self.args = args
        self.star_network = star_network

    def forward(self, batch):
        pts_car, z_vals_car = sample_pts(
            batch["rays_o_car"],
            batch["rays_d_car"],
            self.train_dataset.near,
            self.train_dataset.far,
            self.args.N_samples,
            self.args.perturb,
            self.args.lindisp,
            self.training,
        )
        pts_noncar, z_vals_noncar = sample_pts(
            batch["rays_o_noncar"],
            batch["rays_d_noncar"],
            self.train_dataset.near,
            self.train_dataset.far,
            self.args.N_samples,
            self.args.perturb,
            self.args.lindisp,
            self.training,
        )

        viewdirs_car = batch["rays_d_car"] / torch.norm(
            batch["rays_d_car"], dim=-1, keepdim=True
        )  # [N_rays, 3]
        viewdirs_noncar = batch["rays_d_noncar"] / torch.norm(
            batch["rays_d_noncar"], dim=-1, keepdim=True
        )  # [N_rays, 3]

        return render_star_appinit_semantic(
            self.star_network,
            pts_car,
            viewdirs_car,
            z_vals_car,
            batch["rays_o_car"],
            batch["rays_d_car"],
            pts_noncar,
            viewdirs_noncar,
            z_vals_noncar,
            batch["rays_o_noncar"],
            batch["rays_d_noncar"],
            self.args.N_importance,
        )

    def training_step(self, batch, batch_idx):
        result = self(batch)

        img_loss_car = img2mse(result["rgb_car"], batch["target_car"])
        img_loss_noncar = img2mse(result["rgb_noncar"], batch["target_noncar"])
        img_loss = (img_loss_car + img_loss_noncar) / 2
        loss = img_loss
        psnr = mse2psnr(img_loss)
        img_loss0 = (
            img2mse(result["rgb0_car"], batch["target_car"])
            + img2mse(result["rgb0_noncar"], batch["target_noncar"])
        ) / 2
        loss = loss + img_loss0
        psnr0 = mse2psnr(img_loss0)

        self.log("train/fine_loss", img_loss, on_step=False, on_epoch=True)
        self.log("train/fine_loss_car", img_loss_car, on_step=False, on_epoch=True)
        self.log(
            "train/fine_loss_noncar", img_loss_noncar, on_step=False, on_epoch=True
        )
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/psnr", psnr, on_step=False, on_epoch=True)
        self.log("train/psnr0", psnr0, on_step=False, on_epoch=True)

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
        result = self(batch)

        # Log visualizations for a random view (see dataset implementation)
        val_H = self.val_dataset.H
        val_W = self.val_dataset.W

        rgb0_car = to8b(
            torch.reshape(result["rgb0_car"], (val_H, val_W, 3)).cpu().numpy(),
            "rgb0_car",
        )
        rgb0_noncar = to8b(
            torch.reshape(result["rgb0_noncar"], (val_H, val_W, 3)).cpu().numpy(),
            "rgb0_noncar",
        )

        rgb_car = to8b(
            torch.reshape(result["rgb_car"], (val_H, val_W, 3)).cpu().numpy(),
            "rgb_car",
        )
        rgb_noncar = to8b(
            torch.reshape(result["rgb_noncar"], (val_H, val_W, 3)).cpu().numpy(),
            "rgb_noncar",
        )

        target = to8b(
            torch.reshape(batch["target_car"], (val_H, val_W, 3)).cpu().numpy(),
            "target",
        )

        log_val_table_app_init_semantic(
            self.logger,
            self.current_epoch,
            rgb_car,
            rgb_noncar,
            target,
            rgb0_car,
            rgb0_noncar,
        )

    def setup(self, stage):
        self.train_dataset = StarAppInitSemanticDataset(self.args, split="train")
        self.val_dataset = StarAppInitSemanticDataset(self.args, split="val")

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

    network = STaR(args.num_frames, args, gt_poses=None)
    model = StarAppInitSemantic(args, network)
    # model = torch.compile(model)

    logger = WandbLogger(project=args.expname)
    logger.watch(model, log="all")
    logger.experiment.config.update(args)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"ckpts/appinit_semantic/{logger.version}",
        filename="{epoch:d}",
        every_n_epochs=args.epoch_ckpt,
        save_on_train_epoch_end=True,
        save_top_k=-1,
    )

    early_stopping_cb = EarlyStopping(
        monitor="train/fine_loss",
        mode="min",
        stopping_threshold=args.appearance_init_thres,
    )

    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1), early_stopping_cb]

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
    )

    trainer.fit(model, ckpt_path=args.appearance_ckpt_path)


if __name__ == "__main__":
    print("torch version", torch.__version__)
    print("torch cuda version", torch.version.cuda)
    seed_everything(42, workers=True)
    train()
