from torch.optim.lr_scheduler import StepLR, MultiStepLR


def get_scheduler(
    optimizer, lrate_decay_rate, lrate_decay=None, lrate_decay_steps=None, last_epoch=-1
):
    if lrate_decay_steps:
        return MultiStepLR(
            optimizer,
            milestones=lrate_decay_steps,
            gamma=lrate_decay_rate,
            last_epoch=last_epoch,
        )
    elif lrate_decay:
        return StepLR(
            optimizer,
            step_size=lrate_decay,
            gamma=lrate_decay_rate,
            last_epoch=last_epoch,
        )
    else:
        return None
