from dataclasses import asdict, dataclass


STAGES = {'memory_warmup': (100, 20, 15, 'memory_best'),
          'source_self_training': (50, 10, 10, 'source_pl_best'),
          'dual_domain_finetuning': (100, 30, 20, 'full_best')}


@dataclass(frozen=True)
class MemoryConfig:
    stage: str = 'memory_warmup'
    max_epochs: int = 100
    min_epochs: int = 20
    patience: int = 15
    K: int = 128
    alpha: float = 2.
    beta: float = 1.
    temperature: float = .1
    gamma: float = .9
    q: float = .2
    refresh_interval: int = 10
    lambda_s: float = .5
    lambda_t_max: float = .5
    lambda_adv_max: float = .1
    lambda_sp: float = 1e-4
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 5.
    warmup_epochs: int = 5
    minimum_lr_ratio: float = .1
    unfreeze_epoch: int = 21
    target_ramp_epochs: int = 30
    smoke: bool = False

    def __post_init__(self):
        if self.stage not in STAGES or not 1 <= self.min_epochs <= self.max_epochs or self.patience < 1:
            raise ValueError('Invalid stage or stopping bounds')
        if self.unfreeze_epoch < 2 or self.target_ramp_epochs < 2:
            raise ValueError('Invalid transition schedule')
        if not self.smoke and (self.unfreeze_epoch, self.target_ramp_epochs) != (21, 30):
            raise ValueError('Formal transition schedule mismatch')
        if self.smoke:
            if self.max_epochs > 3:
                raise ValueError('Smoke limited to 1-3 epochs')
        elif (self.max_epochs, self.min_epochs, self.patience) != STAGES[self.stage][:3]:
            raise ValueError('Formal stage stopping schedule mismatch')
        if (self.alpha, self.beta, self.temperature, self.gamma, self.q, self.lambda_s,
            self.lambda_t_max, self.lambda_adv_max, self.lambda_sp, self.weight_decay,
            self.gradient_clip_norm, self.warmup_epochs, self.minimum_lr_ratio) != (
                2., 1., .1, .9, .2, .5, .5, .1, 1e-4, 5e-4, 5., 5, .1):
            raise ValueError('Unregistered change to plan defaults')
        if type(self.K) is not int or self.K < 1 or type(self.refresh_interval) is not int or self.refresh_interval < 1:
            raise ValueError('Invalid K/refresh interval')
        if not self.smoke and (self.K, self.refresh_interval) != (128, 10):
            raise ValueError('Formal anchor/refresh defaults mismatch')

    def resolved(self):
        return {**asdict(self), 'layernorm_affine': False, 'cap_denominator': 'all_predicted_class_candidates',
                'cap_rounding': 'floor', 'consistency_first_index': 1,
                'all_masked_read': 'zero', 'refresh_epochs': '1,1+R,...',
                'dual_source_rounds': 'continue_source_pl_best',
                'unfreeze_optimizer': 'rebuild_preserve_existing_moments_stage_clock',
                'lambda_t_schedule': f'(epoch-1)/{self.target_ramp_epochs - 1} capped at 1',
                'pre_unfreeze_objective': 'full_loss_with_frozen_representation',
                'scheduler': 'linear_5_epochs_then_cosine_v1'}
