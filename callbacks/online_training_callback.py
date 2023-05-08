import torch
from pytorch_lightning.callbacks import Callback

"""
Callback to handle early stopping when num of frames is reached,
increasing number of frames when online threshold is reached, and
setting number of frames in datasets on ckpt load.
"""
class StarOnlineCallback(Callback):
    def __init__(self, args):
        self.online_thres = args.online_thres
        self.max_num_frames = args.num_frames
    
    def on_train_epoch_end(self, trainer, pl_module):
        avg_loss = torch.stack(pl_module.training_fine_losses).mean().cpu().item()
        pl_module.training_fine_losses.clear()

        # Check whether online threshold is reached and increase num of frames
        if avg_loss <= self.online_thres:
            pl_module.current_frame_num += 1
            pl_module.train_dataset.num_frames += 1
            pl_module.val_dataset.num_frames += 1

        # Stop the training if maximum number of frames is reached
        if pl_module.current_frame_num.item() > self.max_num_frames:
            trainer.should_stop = True

    def on_load_checkpoint(trainer, pl_module, checkpoint):
        # Set num_frames of datasets
        pl_module.train_dataset.num_frames = checkpoint['current_frame_num']
        pl_module.val_dataset.num_frames = checkpoint['current_frame_num']


