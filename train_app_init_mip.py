import os
import wandb
import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from utils.logging__ import log_val_table_app_init
from torch.utils.data import DataLoader
from models.star_mipnerf import STaR
from torch import inf

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything

from models.rendering__ import sample_pts, render_star_appinit
from models.rendering__ import mse2psnr, img2mse, to8b
from models.loss import compute_sigma_loss
from datasets.carla_star_app_init import StarAppInitDataset
from utils.visualization import visualize_depth
from utils.io import *
from callbacks.check_batch_grad import CheckBatchGradient


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

    def forward(self, batch):
        viewdirs = batch["rays_d"] / torch.norm(
            batch["rays_d"], dim=-1, keepdim=True
        )  # [N_rays, 3]

        return self.star_network(
            batch["rays_o"],
            viewdirs,
        )

    def training_step(self, batch, batch_idx):
        result = self(batch)

        img_loss = img2mse(result["rgb"], batch["target"])
        loss = img_loss
        img_loss0 = img2mse(result["rgb0"], batch["target"])
        loss = loss + 0.1 * img_loss0

        psnr = mse2psnr(img_loss)
        psnr0 = mse2psnr(img_loss0)

        self.log("train/fine_loss", img_loss, on_step=False, on_epoch=True)
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

        val_mse = img2mse(result["rgb"], batch["target"])
        psnr = mse2psnr(val_mse)

        self.log("val/mse", val_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True)

        # Log visualizations for a random view (see dataset implementation)
        val_H = self.val_dataset.H
        val_W = self.val_dataset.W

        rgb0 = to8b(
            torch.reshape(result["rgb0"], (val_H, val_W, 3)).cpu().detach().numpy(),
            "rgb0",
        )
        depth0 = visualize_depth(result["depth0"], multi_vehicle=False).reshape(
            (val_H, val_W, 3)
        )
        rgb = to8b(
            torch.reshape(result["rgb"], (val_H, val_W, 3)).cpu().detach().numpy(),
            "rgb",
        )
        depth = visualize_depth(result["depth"], multi_vehicle=False).reshape(
            (val_H, val_W, 3)
        )
        target = to8b(
            torch.reshape(batch["target"], (val_H, val_W, 3)).cpu().detach().numpy(),
            "target",
        )

        log_val_table_app_init(
            self.logger,
            self.current_epoch,
            rgb,
            target,
            depth,
            rgb0,
            depth0,
            None,
        )

    def setup(self, stage):
        self.train_dataset = StarAppInitDataset(self.args, split="train")
        self.val_dataset = StarAppInitDataset(self.args, split="val")

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
    logger.experiment.log_code(root=args.code_dir)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"ckpts/appinit/{logger.version}",
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
        detect_anomaly=True,  # NOTE: disable for faster training?
    )

    trainer.fit(model, ckpt_path=args.appearance_ckpt_path)


if __name__ == "__main__":
    print("torch version", torch.__version__)
    print("torch cuda version", torch.version.cuda)
    set_matmul_precision()
    seed_everything(42, workers=True)
    torch.set_printoptions(precision=20, threshold=inf)
    train()
