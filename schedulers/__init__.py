import torch


def MultiStepLR(cfg, optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=cfg.milestones,
        gamma=cfg.gamma
    )


def StepLR(cfg, optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=cfg.step_size,
        gamma=cfg.gamma
    )


def CosineAnnealingLR(cfg, optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg.T_max,
        eta_min=cfg.eta_min
    )


class ConstantLR:
    def __init__(self, *args, **kwargs):
        pass

    def load_state_dict(self, *args, **kwargs):
        pass

    def state_dict(self, *args, **kwargs):
        return 0

    def step(self, *args, **kwargs):
        pass


def create_scheduler(cfg, optimizer):
    type2scheduler = dict(
        MultiStepLR=MultiStepLR,
        CosineAnnealingLR=CosineAnnealingLR,
        StepLR=StepLR,
        ConstantLR=ConstantLR
    )
    return type2scheduler[cfg.scheduler_type](cfg, optimizer)
