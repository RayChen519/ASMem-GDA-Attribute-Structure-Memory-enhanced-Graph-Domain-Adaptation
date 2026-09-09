import torch


def build_optimizer(modules, config):
    rates = {'shared_gcn': config.encoder_lr, 'attribute_structure': config.as_lr,
             'classifier': config.classifier_lr, 'discriminator': config.discriminator_lr}
    groups = [{'name': name, 'params': [p for p in module.parameters() if p.requires_grad],
               'lr': rates[name]} for name, module in modules.items()]
    ids = [id(p) for g in groups for p in g['params']]
    if len(ids) != len(set(ids)) or any(not g['params'] for g in groups):
        raise ValueError('Duplicate/empty optimizer parameter group')
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)
