import copy

import torch

from training.schedules.warmup_cosine import build_scheduler


def rates(stage, epoch, unfreeze_epoch=21):
    if stage == 'memory_warmup':
        return {'memory': 1e-3}
    if stage == 'source_self_training':
        return {'classifier': 1e-3}
    if stage != 'dual_domain_finetuning':
        raise ValueError('Unknown stage')
    if epoch < unfreeze_epoch:
        return {'classifier': 1e-3, 'discriminator': 1e-3}
    return {'shared_gcn': 1e-4, 'attribute_structure': 1e-4,
            'classifier': 5e-4, 'discriminator': 5e-4}


def configure(modules, config, epoch):
    active = rates(config.stage, epoch, config.unfreeze_epoch)
    for name, module in modules.items():
        module.requires_grad_(name in active)
        module.train(name in active)
        for parameter in module.parameters():
            parameter.grad = None
    return active


def build(modules, config, epoch=1, previous=None, completed_epochs=0):
    active = configure(modules, config, epoch)
    optimizer = torch.optim.AdamW([{'name': n, 'params': list(modules[n].parameters()), 'lr': lr}
                                  for n, lr in active.items()], weight_decay=config.weight_decay)
    if previous is not None:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in previous.state:
                    optimizer.state[p] = copy.deepcopy(previous.state[p])
    scheduler = build_scheduler(optimizer, config)
    if completed_epochs:
        scheduler.last_epoch = completed_epochs
        scheduler._step_count = completed_epochs + 1
        for base, group, fn in zip(scheduler.base_lrs, optimizer.param_groups, scheduler.lr_lambdas):
            group['lr'] = base * fn(completed_epochs)
        scheduler._last_lr = [g['lr'] for g in optimizer.param_groups]
    return optimizer, scheduler
