from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR


def get_scheduler(
    optimizer, lrate_decay_rate, lrate_decay=None, lrate_decay_steps=None, last_epoch=-1
):
    if lrate_decay_steps:
        print("Using MultiStepLR")
        return MultiStepLR(
            optimizer,
            milestones=lrate_decay_steps,
            gamma=lrate_decay_rate,
            last_epoch=last_epoch,
        )
    elif lrate_decay:
        print("Using StepLR")
        return StepLR(
            optimizer,
            step_size=lrate_decay,
            gamma=lrate_decay_rate,
            last_epoch=last_epoch,
        )
    else:
        print("Using CosineAnnealingLR")
        return CosineAnnealingLR(
            optimizer, T_max=1000 * 60, eta_min=0.0001, last_epoch=last_epoch
        )
