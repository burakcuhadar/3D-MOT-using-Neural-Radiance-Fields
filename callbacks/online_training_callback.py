import torch
from pytorch_lightning.callbacks import Callback
from datasets.carla_star_online import StarOnlineDataset
from torch.utils.data import DataLoader

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
    def on_train_epoch_end(self, trainer, pl_module):
        avg_loss = torch.stack(pl_module.training_fine_losses).mean().cpu().item()
        
        if pl_module.current_frame_num == self.args.initial_num_frames:
            if avg_loss <= self.online_thres:
                """pl_module.start_frame = torch.tensor(
                    [self.args.initial_num_frames],
                    dtype=torch.long,
                    device=pl_module.device,
                )  # TODO also try with self.args.initial_num_frames-1?"""
                pl_module.start_frame = torch.tensor([3], dtype=pl_module.start_frame.dtype, device=pl_module.start_frame.device)
                pl_module.current_frame_num += 1

                pl_module.log("train/current_frame_num", pl_module.current_frame_num)
                pl_module.log("train/start_frame", pl_module.start_frame)
        else:
            self.count = self.count + 1
            if self.count > 40 and avg_loss <= self.online_thres:
                self.count = 0
                pl_module.start_frame += 1
                pl_module.current_frame_num += 1

                pl_module.log("train/current_frame_num", pl_module.current_frame_num)
                pl_module.log("train/start_frame", pl_module.start_frame)

        pl_module.training_fine_losses.clear()
        torch.cuda.synchronize(device=pl_module.device)

        # Stop the training if maximum number of frames is reached
        if pl_module.current_frame_num.item() > self.max_num_frames:
            trainer.should_stop = True
            return
        
        #TODO if no init, then set the next pose the same as the previous one

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
        # Log poses loaded from checkpoint
        print("Loaded poses:")
        print(checkpoint["state_dict"]["poses"])

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

        """
        self.train_dataset = StarOnlineDataset(
            self.args,
            "train",
            checkpoint["state_dict"]["current_frame_num"].item(),
            checkpoint["state_dict"]["start_frame"].item(),
        )
        self.val_dataset = StarOnlineDataset(
            self.args,
            "val",
            checkpoint["state_dict"]["current_frame_num"].item(),
            checkpoint["state_dict"]["start_frame"].item(),
        )
        self.test_dataset = StarOnlineDataset(
            self.args, "test", checkpoint["state_dict"]["current_frame_num"].item()
        )
        """
        # TODO need to re-set the dataloaders?
