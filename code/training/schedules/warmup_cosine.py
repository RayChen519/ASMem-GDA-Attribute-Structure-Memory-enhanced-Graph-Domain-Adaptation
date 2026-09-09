import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, config):
    # LambdaLR index 0 is installed before the first optimizer step.
    # Epochs 1..5 use 1/5..5/5; epoch max_epochs reaches the floor.
    def ratio(index):
        if index < config.warmup_epochs:
            return (index + 1) / config.warmup_epochs
        progress = min(1.0, (index + 1 - config.warmup_epochs) /
                       max(1, config.max_epochs - config.warmup_epochs))
        return config.minimum_lr_ratio + (1 - config.minimum_lr_ratio) * (
            1 + math.cos(math.pi * progress)) / 2
    return LambdaLR(optimizer, ratio)
