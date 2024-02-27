import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from utils.metrics import get_pose_metrics_multi
import numpy as np
from utils.optim import get_scheduler
from optimizer.hybrid_optimizer import HybridOptim, HybridLRS
from pytorch_lightning.core.optimizer import (
    LightningOptimizer,
    _configure_schedulers_automatic_opt,
    _configure_schedulers_manual_opt,
)


"""
Callback to handle early stopping when num of frames is reached,
increasing number of frames when online threshold is reached, and
setting number of frames in datasets on ckpt load.
"""


class StarOnlineCallback(Callback):
    def __init__(self, args):
        self.args = args
        self.online_thres = args.online_thres
        self.max_num_frames = args.num_frames
        self.count = 0

    """
    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        avg_loss = torch.stack(pl_module.training_fine_losses).mean().cpu().item()
        pl_module.training_fine_losses.clear()

        if avg_loss <= self.online_thres:
            if pl_module.current_frame_num == self.args.initial_num_frames:
                pl_module.start_frame = torch.tensor(
                    self.args.initial_num_frames - 2, device=pl_module.device
                )
                pl_module.current_frame_num = torch.tensor(
                    self.args.initial_num_frames + 1, device=pl_module.device
                )
            else:
                pl_module.start_frame += 1
                pl_module.current_frame_num += 1

            pl_module.log("train/current_frame_num", pl_module.current_frame_num)
            pl_module.log("train/start_frame", pl_module.start_frame)

            torch.cuda.synchronize(device=pl_module.device)

            self.online_thres = 85e-5

            # Stop the training if maximum number of frames is reached
            if pl_module.current_frame_num.item() > self.max_num_frames:
                trainer.should_stop = True
                return

            pl_module.train_dataset.current_frame = pl_module.current_frame_num.item()
            pl_module.train_dataset.start_frame = pl_module.start_frame.item()
            pl_module.val_dataset.current_frame = pl_module.current_frame_num.item()
            pl_module.val_dataset.start_frame = pl_module.start_frame.item()
    """

    """ 5-frame sliding window
    def on_train_epoch_end(self, trainer, pl_module):
        avg_loss = torch.stack(pl_module.training_fine_losses).mean().cpu().item()
        pl_module.training_fine_losses.clear()

        if avg_loss <= self.online_thres:
            pl_module.start_frame += 1
            pl_module.current_frame_num += 1

            pl_module.log("train/current_frame_num", pl_module.current_frame_num)
            pl_module.log("train/start_frame", pl_module.start_frame)

            torch.cuda.synchronize(device=pl_module.device)

            # Stop the training if maximum number of frames is reached
            if pl_module.current_frame_num.item() > self.max_num_frames:
                trainer.should_stop = True
                return

            pl_module.train_dataset.current_frame = pl_module.current_frame_num.item()
            pl_module.train_dataset.start_frame = pl_module.start_frame.item()
            pl_module.val_dataset.current_frame = pl_module.current_frame_num.item()
            pl_module.val_dataset.start_frame = pl_module.start_frame.item()
    """

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module: pl.LightningModule):
        # avg_loss = torch.stack(pl_module.training_fine_losses).mean().cpu().item()
        avg_loss = pl_module.running_fine_loss / pl_module.train_dataset.step_num
        pl_module.running_fine_loss = 0.0

        if trainer.current_epoch < pl_module.args.precrop_iters:
            return
        else:
            # print("crop finished")
            pl_module.train_dataset.crop = False
            pl_module.val_dataset.crop = False

        if pl_module.current_frame_num == self.args.initial_num_frames:
            if avg_loss <= self.online_thres:
                # pl_module.start_frame = torch.tensor([2], dtype=pl_module.start_frame.dtype, device=pl_module.start_frame.device)
                # pl_module.start_frame += 1
                pl_module.current_frame_num += 1
                torch.cuda.synchronize(device=pl_module.device)

                # freeze second frame pose
                # pl_module.poses[0].requires_grad = False

                pl_module.logger.log_metrics(
                    {"train/current_frame_num": pl_module.current_frame_num}
                )
                pl_module.logger.log_metrics(
                    {"train/start_frame": pl_module.start_frame}
                )

                # Set new threshold
                self.online_thres = 95e-5
                # Decrease lr of nerfs
                # for g in pl_module.optimizers().optimizer.optimizers[0].param_groups:
                #     g["lr"] = 1e-5

                # freeze nerf?
                # for p in pl_module.star_network.parameters():
                #     p.requires_grad = False
                #     p.grad = None

                # Reload optimizer and scheduler since we updated start_frame
                # trainer.strategy.setup(trainer)

        else:
            self.count = self.count + 1
            if self.count > 70 and avg_loss <= self.online_thres:
                self.count = 0
                # pl_module.start_frame += 1
                pl_module.current_frame_num += 1
                torch.cuda.synchronize(device=pl_module.device)

                # freeze starting pose
                # pl_module.poses[pl_module.start_frame - 1].requires_grad = False

                pl_module.logger.log_metrics(
                    {"train/current_frame_num": pl_module.current_frame_num}
                )
                pl_module.logger.log_metrics(
                    {"train/start_frame": pl_module.start_frame}
                )

        # Stop the training if maximum number of frames is reached
        if pl_module.current_frame_num.item() > self.max_num_frames:
            trainer.should_stop = True
            return

        # TODO if no init, then set the next pose the same as the previous one

        pl_module.train_dataset.current_frame = pl_module.current_frame_num.item()
        pl_module.train_dataset.start_frame = pl_module.start_frame.item()
        pl_module.val_dataset.current_frame = pl_module.current_frame_num.item()
        pl_module.val_dataset.start_frame = pl_module.start_frame.item()

    '''
    def on_train_epoch_end(self, trainer, pl_module):
        avg_loss = torch.stack(pl_module.training_fine_losses).mean().cpu().item()
        pl_module.training_fine_losses.clear()

        # Check whether online threshold is reached and increase num of frames
        if avg_loss <= self.online_thres:
            if pl_module.current_frame_num == self.args.initial_num_frames:
                pl_module.start_frame = torch.tensor(
                    [self.args.initial_num_frames],
                    dtype=torch.long,
                    device=pl_module.device,
                )  # TODO also try with self.args.initial_num_frames-1?
            else:
                pl_module.start_frame += 1

            pl_module.current_frame_num += 1
            torch.cuda.synchronize(device=pl_module.device)

            pl_module.log("train/current_frame_num", pl_module.current_frame_num)
            pl_module.log("train/start_frame", pl_module.start_frame)

            self.online_thres = 85e-5

            # Stop the training if maximum number of frames is reached
            if pl_module.current_frame_num.item() > self.max_num_frames:
                trainer.should_stop = True
                return

            pl_module.train_dataset.current_frame = pl_module.current_frame_num.item()
            pl_module.train_dataset.start_frame = pl_module.start_frame.item()
            pl_module.val_dataset.current_frame = pl_module.current_frame_num.item()
            pl_module.val_dataset.start_frame = pl_module.start_frame.item()

            """pl_module.train_dataset = StarOnlineDataset(
                self.args,
                "train",
                pl_module.current_frame_num.item(),
                start_frame=pl_module.start_frame.item(),
            )
            pl_module.train_dataloader = lambda: DataLoader(
                pl_module.train_dataset,
                batch_size=None,
                shuffle=True,
                num_workers=pl_module.args.num_workers,
                pin_memory=True,
            )

            pl_module.val_dataset = StarOnlineDataset(
                self.args,
                "val",
                pl_module.current_frame_num.item(),
                start_frame=pl_module.start_frame.item(),
            )
            pl_module.val_dataloader = lambda: DataLoader(
                pl_module.val_dataset,
                batch_size=None,
                num_workers=1,
                pin_memory=True,
            )"""
    '''

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        trans_error, rot_error, _, _, rot_error_euler, _ = get_pose_metrics_multi(
            torch.cat(list(pl_module.poses), dim=0),
            pl_module.train_dataset.gt_relative_poses[:, 1:, ...]
            .transpose(0, 1)
            .to(pl_module.device),
            reduce=False,
        )
        print(f"trans errors:\n {trans_error}")
        print(f"rot errors:\n {rot_error}")
        print(f"rot errors euler:\n {rot_error_euler}")
        print(f"loaded start frame: {checkpoint['state_dict']['start_frame']} \n")
        print(
            f"loaded current frame: {checkpoint['state_dict']['current_frame_num']} \n"
        )

        # Set num_frames of datasets
        pl_module.train_dataset.current_frame = checkpoint["state_dict"][
            "current_frame_num"
        ].item()
        pl_module.train_dataset.start_frame = checkpoint["state_dict"][
            "start_frame"
        ].item()
        pl_module.val_dataset.current_frame = checkpoint["state_dict"][
            "current_frame_num"
        ].item()
        pl_module.val_dataset.start_frame = checkpoint["state_dict"][
            "start_frame"
        ].item()
        pl_module.test_dataset.current_frame = checkpoint["state_dict"][
            "current_frame_num"
        ].item()
        pl_module.test_dataset.start_frame = checkpoint["state_dict"][
            "start_frame"
        ].item()

        # self.ckpt_loaded = True

    """
    def on_train_start(self, trainer, pl_module):
        if self.ckpt_loaded:
            # Reload optimizer and scheduler since we updated start_frame
            #trainer.strategy.setup(trainer)
            self.ckpt_loaded = False
    """
