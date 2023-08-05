import torch
import imageio

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar

from train_online import create_model
from utils.io import *
from callbacks.online_training_callback import StarOnlineCallback
from datasets.carla_star_online import StarOnlineDataset
from models.rendering import sample_pts, render_star_online, to8b

def render_batch(dataset, batch, args, star_network):
    pts, z_vals = sample_pts(
        batch["rays_o"],
        batch["rays_d"],
        dataset.near,
        dataset.far,
        args.N_samples,
        args.perturb,
        args.lindisp,
        False,
    )

    viewdirs = batch["rays_d"] / torch.norm(
        batch["rays_d"], dim=-1, keepdim=True
    )  # [N_rays, 3]

    pose = batch["object_pose"]
    
    return render_star_online(
        star_network,
        pts,
        viewdirs,
        z_vals,
        batch["rays_o"],
        batch["rays_d"],
        args.N_importance,
        pose,
        step=None,
    )


def test(args, model):
    logger = WandbLogger(project=args.expname)
    logger.experiment.config.update(args)

    star_network = model.star_network
    star_network.eval()
    device = torch.device("cuda")
    star_network.to(device)

    # Get car poses
    ds = StarOnlineDataset(
        args,
        "test",
        args.num_frames,
        args.initial_num_frames,
    )
    #object_poses = ds.object_poses
    dataloader = DataLoader(
        ds,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
    )

    test_H = ds.H
    test_W = ds.W

    # Render dynamic nerf with the given poses
    for i,batch in enumerate(dataloader):
        with torch.no_grad():
            for key in batch:
                if batch[key] is not None:
                    batch[key] = batch[key].to(device)

            print("object pose", batch["object_pose"])
            
            result = render_batch(ds, batch, args, star_network)

            rgb_dynamic0 = to8b(
                torch.reshape(result["rgb_dynamic0"], (test_H, test_W, 3))
                .cpu()
                .detach()
                .numpy(),
                "rgb_dynamic0",
            )
            rgb_dynamic = to8b(
                torch.reshape(result["rgb_dynamic"], (test_H, test_W, 3))
                .cpu()
                .detach()
                .numpy(),
                "rgb_dynamic",
            )
            rgb = to8b(
                torch.reshape(result["rgb"], (test_H, test_W, 3)).cpu().detach().numpy(),
                "rgb",
            )   

            filename = os.path.join("out", f'{i}_coarse.png')
            imageio.imwrite(filename, rgb_dynamic0)
            filename = os.path.join("out", f'{i}_fine.png')
            imageio.imwrite(filename, rgb_dynamic)
            filename = os.path.join("out", f'{i}_rgb.png')
            imageio.imwrite(filename, rgb)

    # Generate video, upload to wandb
    #TODO

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
