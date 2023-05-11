import wandb
import os


def log_val_table_app_init(logger, epoch, rgb, rgb_gt, depth, rgb0, depth0, z_std):
    columns = ["epoch", "rgb", "gt rgb", "depth", "rgb_coarse", "depth_coarse", "z_std"]
    data = [
        [
            epoch,
            wandb.Image(rgb),
            wandb.Image(rgb_gt),
            wandb.Image(depth),
            wandb.Image(rgb0),
            wandb.Image(depth0),
            wandb.Image(z_std),
        ]
    ]
    logger.log_table(key="val table", columns=columns, data=data)


def log_val_table_app_init_semantic(
    logger, epoch, rgb_car, rgb_noncar, rgb_gt, rgb0_car, rgb0_noncar
):
    columns = [
        "epoch",
        "rgb car",
        "rgb noncar",
        "gt rgb",
        "rgb_coarse car",
        "rgb_coarse noncar",
    ]
    data = [
        [
            epoch,
            wandb.Image(rgb_car),
            wandb.Image(rgb_noncar),
            wandb.Image(rgb_gt),
            wandb.Image(rgb0_car),
            wandb.Image(rgb0_noncar),
        ]
    ]
    logger.log_table(key="val table", columns=columns, data=data)


def log_val_table_online(
    logger,
    epoch,
    rgb,
    rgb_gt,
    rgb_dynamic,
    rgb_static,
    depth,
    depth_dynamic,
    depth_static,
    rgb0,
    rgb0_dynamic,
    rgb0_static,
    depth0,
    depth_dynamic0,
    depth_static0,
    z_std,
):
    columns = [
        "epoch",
        "rgb",
        "gt rgb",
        "dynamic rgb",
        "static rgb",
        "depth",
        "depth dynamic",
        "depth static",
        "rgb_coarse",
        "dynamic rgb coarse",
        "static rgb coarse",
        "depth0",
        "depth dynamic0",
        "depth static0",
        "z_std",
    ]
    data = [
        [
            epoch,
            wandb.Image(rgb),
            wandb.Image(rgb_gt),
            wandb.Image(rgb_dynamic),
            wandb.Image(rgb_static),
            wandb.Image(depth),
            wandb.Image(depth_dynamic),
            wandb.Image(depth_static),
            wandb.Image(rgb0),
            wandb.Image(rgb0_dynamic),
            wandb.Image(rgb0_static),
            wandb.Image(depth0),
            wandb.Image(depth_dynamic0),
            wandb.Image(depth_static0),
            wandb.Image(z_std),
        ]
    ]
    logger.log_table(key="val table", columns=columns, data=data)
