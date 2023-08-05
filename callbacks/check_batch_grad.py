import torch

from pytorch_lightning.callbacks import Callback
from models.loss import compute_sigma_loss_per_ray


class CheckBatchGradient(Callback):
    def on_train_start(self, trainer, module):
        n = 0

        example_input = module.train_dataset.__getitem__(0)
        for key in example_input.keys():
            example_input[key] = torch.from_numpy(example_input[key]).to(module.device)

        example_input["rays_o"].requires_grad = True
        example_input["rays_d"].requires_grad = True

        module.zero_grad()
        result = module(example_input)
        result["rgb"][n].abs().sum().backward()

        zero_grad_inds = list(range(example_input["rays_o"].size(0)))
        zero_grad_inds.pop(n)

        if example_input["rays_o"].grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")

        if example_input["rays_d"].grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")

        # Check for sigma loss
        if module.args.sigma_loss:
            module.zero_grad()
            example_input["rays_o"].grad.zero_()
            example_input["rays_d"].grad.zero_()

            result = module(example_input)
            sigma_loss = compute_sigma_loss_per_ray(
                result["weights"],
                result["z_vals"],
                result["dists"],
                example_input["target_depth"],
            )
            sigma_loss[n].backward()

            if example_input["rays_o"].grad[zero_grad_inds].abs().sum().item() > 0:
                raise RuntimeError("Your model mixes data across the batch dimension!")

            if example_input["rays_d"].grad[zero_grad_inds].abs().sum().item() > 0:
                raise RuntimeError("Your model mixes data across the batch dimension!")
