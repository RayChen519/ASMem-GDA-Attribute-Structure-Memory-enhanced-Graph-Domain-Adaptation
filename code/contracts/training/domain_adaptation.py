from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class DAConfig:
    max_epochs: int = 150
    min_epochs: int = 30
    patience: int = 20
    encoder_lr: float = 5e-4
    as_lr: float = 1e-3
    classifier_lr: float = 1e-3
    discriminator_lr: float = 1e-3
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 5.
    warmup_epochs: int = 5
    minimum_lr_ratio: float = .1
    lambda_adv_max: float = .1
    lambda_sp: float = 1e-4
    P: int = 128
    metis_seed: int = 0
    smoke: bool = False

    def __post_init__(self):
        if not 1 <= self.min_epochs <= self.max_epochs or self.patience < 1:
            raise ValueError('Invalid early stopping bounds')
        if not self.smoke and (self.max_epochs, self.min_epochs, self.patience) != (150, 30, 20):
            raise ValueError('Shortened epochs require explicit smoke=True')
        if self.smoke and self.max_epochs > 3:
            raise ValueError('DA smoke is limited to 1-3 epochs')
        if (self.P, self.metis_seed) != (128, 0):
            raise ValueError('Main and smoke use METIS P=128, seed=0')
        if (self.encoder_lr, self.as_lr, self.classifier_lr, self.discriminator_lr,
                self.weight_decay, self.gradient_clip_norm, self.warmup_epochs, self.minimum_lr_ratio,
                self.lambda_adv_max, self.lambda_sp) != (5e-4, 1e-3, 1e-3, 1e-3, 5e-4, 5., 5, .1, .1, 1e-4):
            raise ValueError('DA optimizer/loss defaults must follow the training plan')

    def resolved(self):
        return {**asdict(self), 'architecture': [6775, 256, 128], 'attention_heads': 1,
                'bga_blocks': 1, 'attention_scope': 'full_domain_full_batch',
                'selection': 'source_only', 'norm': 'post', 'dropout': .1,
                'threshold_initial_rho': 0., 'discriminator': [128, 64, 1],
                'domain_reduction': 'all_nodes_mean', 'grl_coefficient': 1.,
                'model_selection': 'source_validation_macro_f1_then_loss',
                'scheduler': 'linear_5_epochs_then_cosine_v1'}


def adversarial_weight(epoch, config):
    """One full-batch step per epoch; progress runs from 0 to 1 inclusive."""
    progress = (epoch - 1) / max(1, config.max_epochs - 1)
    if not 0 <= progress <= 1:
        raise ValueError('Invalid DA epoch/progress')
    return progress, config.lambda_adv_max * (2 / (1 + math.exp(-10 * progress)) - 1)
