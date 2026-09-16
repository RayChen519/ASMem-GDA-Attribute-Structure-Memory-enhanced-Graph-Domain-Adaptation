from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class WarmupConfig:
    max_epochs: int = 100
    min_epochs: int = 20
    patience: int = 15
    encoder_lr: float = 5e-4
    classifier_lr: float = 1e-3
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 5.0
    warmup_epochs: int = 5
    minimum_lr_ratio: float = 0.1
    smoke: bool = False
    development: bool = False
    sensitivity: bool = False
    variant: str = "B6"

    def __post_init__(self):
        from experiments.registry import variant
        variant(self.variant)
        from experiments.configuration import validate_numbers
        validate_numbers(self)
        if not 1 <= self.min_epochs <= self.max_epochs or self.patience < 1:
            raise ValueError('Invalid early stopping bounds')
        if not self.smoke and (self.max_epochs, self.min_epochs, self.patience) != (100, 20, 15):
            raise ValueError('Shortened epochs require explicit smoke=True')
        if not self.development and (self.encoder_lr, self.classifier_lr, self.weight_decay,
                self.gradient_clip_norm, self.warmup_epochs, self.minimum_lr_ratio) != (
                5e-4, 1e-3, 5e-4, 5.0, 5, 0.1):
            raise ValueError('Optimizer/scheduler must follow the training plan')

    def resolved(self):
        return {**asdict(self), 'architecture': [6775, 256, 128],
                'classifier': [128, 64], 'dropout': 0.5,
                'selection': 'source_validation_macro_f1_then_loss',
                'scheduler': 'linear_5_epochs_then_cosine_v1'}


@dataclass
class EarlyStopState:
    best_epoch: int = 0
    best_macro_f1: float = -1.0
    best_loss: float = float('inf')
    bad_epochs: int = 0

    def update(self, epoch, macro_f1, loss):
        if not math.isfinite(macro_f1) or not math.isfinite(loss):
            raise ValueError('Nonfinite Source validation metric')
        improved = (macro_f1 > self.best_macro_f1 or
                    (macro_f1 == self.best_macro_f1 and loss < self.best_loss))
        if improved:
            self.best_epoch, self.best_macro_f1, self.best_loss = epoch, macro_f1, loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    def should_stop(self, epoch, config):
        return epoch >= config.min_epochs and self.bad_epochs >= config.patience
