import torch

from pytorch_lightning import seed_everything

from utils.io import *

from train_online import train, StarOnline, create_model


class StarOnlineSemantic(StarOnline):
    def forward(self, batch, batch_idx):
        """
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

        if self.args.load_gt_poses:
            # pose = self.gt_poses[batch["frames"][0, 0]]
            pose0 = torch.zeros((1, 6), requires_grad=False, device=self.device)
            poses = torch.cat((pose0, batch["gt_relative_poses"][1:, ...]), dim=0)
            pose = poses[batch["frames"][0]][0]
        else:
            pose0 = torch.zeros((1, 6), requires_grad=False, device=self.device)
            poses = torch.cat((pose0, self.poses), dim=0)
            pose = poses[batch["frames"][0]][0]

        return render_star_online(
            self.star_network,
            pts,
            viewdirs,
            z_vals,
            batch["rays_o"],
            batch["rays_d"],
            self.args.N_importance,
            pose,
            step=self.current_epoch,
        )
        """
        pass

    def training_step(self, batch, batch_idx):
        # TODO need to adapt?
        pass

    def validation_step(self, batch, batch_idx):
        # TODO need to adapt?
        pass

    def setup(self, stage):
        # TODO
        pass


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

    train(args, model)
