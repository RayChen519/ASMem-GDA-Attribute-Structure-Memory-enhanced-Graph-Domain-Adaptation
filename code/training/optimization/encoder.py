from torch.optim import AdamW


def build_optimizer(encoder, classifier, config):
    groups = []
    seen = set()
    for name, module, lr in [('shared_gcn', encoder, config.encoder_lr),
                              ('classifier', classifier, config.classifier_lr)]:
        parameters = [p for p in module.parameters() if p.requires_grad]
        if not parameters or seen.intersection(map(id, parameters)):
            raise ValueError('Empty or duplicate optimizer parameter group')
        seen.update(map(id, parameters))
        groups.append({'name': name, 'params': parameters, 'lr': lr})
    return AdamW(groups, weight_decay=config.weight_decay)
