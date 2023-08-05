from train_online import create_model
from utils.io import *

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar

from callbacks.online_training_callback import StarOnlineCallback


import torch


def test(args, model):
    logger = WandbLogger(project=args.expname)
    logger.experiment.config.update(args)

    # To load current frame num
    star_online_cb = StarOnlineCallback(args)
    callbacks = [TQDMProgressBar(refresh_rate=1), star_online_cb]

    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=False,
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,
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

    set_matmul_precision()
    seed_everything(42, workers=True)

    parser = config_parser()
    args = parser.parse_args()
    model = create_model(args)

    test(args, model)
