from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F

from contracts.data import SourceTrainView, SourceValidationView, TargetTrainView
from contracts.training.encoder import EarlyStopState, WarmupConfig
from data.cache.store import digest
from data.prepare import tensor_hash
from evaluation.metrics.classification import classification_metrics
from training.checkpoints.encoder import load_checkpoint, save_checkpoint, validate_metadata
from training.optimization.encoder import build_optimizer
from training.schedules.warmup_cosine import build_scheduler


@contextmanager
def evaluation_mode(*modules):
    previous = [(sub, sub.training) for module in modules for sub in module.modules()]
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            yield
    finally:
        for sub, training in previous:
            sub.training = training


def finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f'Nonfinite {name}')


def graph_inputs(graph, device):
    if graph.feature_dim != 6775 or graph.x.ndim != 2 or graph.x.shape[1] != 6775:
        raise ValueError('Graph requires 6775 features')
    if graph.x.dtype != torch.float32:
        raise ValueError('Dataset features must be float32')
    if not torch.equal(graph.node_id.cpu(), torch.arange(graph.x.shape[0])):
        raise ValueError('Graph node order differs from Dataset contiguous node IDs')
    finite(graph.x, 'features')
    adjacency = graph.normalized_adjacency
    finite(adjacency.values(), 'adjacency')
    return graph.x.to(device), adjacency.to(device)


def inspect_target(encoder, classifier, target, attribute_union_hash, dataset_version=None):
    """The only warm-up Target entry point: no supervision, metrics or selection."""
    if not isinstance(target, TargetTrainView):
        raise TypeError('Target inspection accepts only TargetTrainView')
    if target.attribute_union_hash != attribute_union_hash:
        raise ValueError('Source/Target attribute vocabulary mismatch')
    if dataset_version is not None and target.dataset_version != dataset_version:
        raise ValueError('Source/Target dataset version mismatch')
    with evaluation_mode(encoder, classifier):
        x, adjacency = graph_inputs(target, next(encoder.parameters()).device)
        output = encoder(x, adjacency)
        logits = classifier(output.z)
        n = x.shape[0]
        if output.h1.shape != (n, 256) or output.z.shape != (n, 128) or logits.shape != (n, classifier.num_classes):
            raise ValueError('Target output dimension mismatch')
        for name, value in [('H1', output.h1), ('Z', output.z), ('logits', logits)]:
            finite(value, name)
        return {'h1_shape': list(output.h1.shape), 'z_shape': list(output.z.shape),
                'logits_shape': list(logits.shape), 'finite': True}


class EncoderWarmup:
    def __init__(self, encoder, classifier, source, validation, metadata,
                 output_dir, config=None, device='cpu'):
        self.config = config or WarmupConfig()
        if not isinstance(source, SourceTrainView) or not isinstance(validation, SourceValidationView):
            raise TypeError('Use isolated Source train and validation views')
        validate_metadata(metadata, encoder, classifier)
        if metadata['resolved_config_hash'] != digest(self.config.resolved()):
            raise ValueError('Trainer config differs from checkpoint metadata')
        if source.split_hash != validation.split_hash or source.split_hash != metadata['split_hash']:
            raise ValueError('Source validation split mismatch')
        graph = source.graph
        if graph.attribute_union_hash != metadata['attribute_union_hash'] or graph.dataset_version != metadata['dataset_version']:
            raise ValueError('Source dataset identity mismatch')
        masks = [graph.source_train_mask, graph.source_val_mask, graph.source_unlabeled_mask]
        if any(m.dtype != torch.bool or m.shape != (graph.x.shape[0],) for m in masks):
            raise ValueError('Invalid Source masks')
        if not torch.all(sum(m.int() for m in masks) == 1):
            raise ValueError('Source masks must be disjoint and cover all nodes')
        for mask, ids, labels, key in [
                (masks[0], source.train_node_id, source.train_y, 'source_train_mask_hash'),
                (masks[1], validation.node_id, validation.validation_y, 'source_val_mask_hash')]:
            if (ids.dtype != torch.long or not torch.equal(ids.cpu(), torch.where(mask.cpu())[0])
                    or labels.dtype != torch.long or labels.shape != ids.shape or labels.numel() == 0
                    or labels.min() < 0 or labels.max() >= classifier.num_classes
                    or torch.unique(labels).numel() != classifier.num_classes):
                raise ValueError('Label indices/classes violate Source split contract')
            if tensor_hash({'mask': mask}) != metadata[key]:
                raise ValueError('Source mask hash mismatch')
        if not all(p.requires_grad for module in (encoder, classifier) for p in module.parameters()):
            raise ValueError('Stage 1 requires trainable GCN and classifier')
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.x, self.adjacency = graph_inputs(graph, device)
        self.train_ids, self.train_y = source.train_node_id.to(device), source.train_y.to(device)
        self.val_ids, self.val_y = validation.node_id.to(device), validation.validation_y.to(device)
        self.metadata = metadata
        self.output_dir = Path(output_dir)
        self.optimizer = build_optimizer(encoder, classifier, self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        self.early_stop = EarlyStopState()
        self.epoch = 0
        self.history = []
        self.target_diagnostics = None

    def train_step(self):
        self.encoder.train()
        self.classifier.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.encoder(self.x, self.adjacency)
        finite(output.h1, 'Source H1')
        finite(output.z, 'Source Z')
        logits = self.classifier(output.z[self.train_ids])
        loss = F.cross_entropy(logits, self.train_y)
        finite(loss, 'Source training loss')
        loss.backward()
        parameters = [p for module in (self.encoder, self.classifier) for p in module.parameters()]
        for p in parameters:
            if not p.requires_grad or p.grad is None:
                raise ValueError('Stage-1 trainable parameter has no gradient')
            finite(p.grad, 'Source gradient')
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip_norm,
                                                   error_if_nonfinite=True)
        self.optimizer.step()
        return loss.item(), grad_norm.item()

    def validate(self):
        with evaluation_mode(self.encoder, self.classifier):
            z = self.encoder(self.x, self.adjacency).z
            return classification_metrics(self.classifier(z[self.val_ids]), self.val_y,
                                          self.classifier.num_classes)

    def resume(self, path):
        if Path(path).resolve().parent != self.output_dir.resolve():
            raise ValueError('Resume in the original run directory to preserve encoder_best')
        if not (self.output_dir / 'encoder_best.pt').is_file():
            raise ValueError('Resume requires the original encoder_best artifact')
        state = load_checkpoint(path, self.encoder, self.classifier, self.metadata,
                                optimizer=self.optimizer, scheduler=self.scheduler, resume=True)
        self.epoch, self.early_stop, self.history = state['epoch'], state['early_stop'], state['history']

    def fit(self, target=None, until_epoch=None):
        """until_epoch interrupts at a completed epoch without changing the schedule.

        Live modules remain at the latest epoch. DA must explicitly load
        encoder_best.pt; latest is reserved for exact Stage-1 continuation.
        """
        if target is not None:
            self.target_diagnostics = inspect_target(
                self.encoder, self.classifier, target, self.metadata['attribute_union_hash'],
                self.metadata['dataset_version'])
        end = self.config.max_epochs if until_epoch is None else min(until_epoch, self.config.max_epochs)
        while self.epoch < end and not self.early_stop.should_stop(self.epoch, self.config):
            epoch = self.epoch + 1
            lr = [group['lr'] for group in self.optimizer.param_groups]
            train_loss, grad_norm = self.train_step()
            metrics = self.validate()
            improved = self.early_stop.update(epoch, metrics['macro_f1'], metrics['loss'])
            self.scheduler.step()
            self.epoch = epoch
            self.history.append({'epoch': epoch, 'train_loss': train_loss, 'grad_norm': grad_norm,
                                 'lr': lr, 'source_validation': metrics})
            args = (self.encoder, self.classifier, self.optimizer, self.scheduler,
                    self.early_stop, self.epoch, self.metadata, self.history)
            if improved:
                save_checkpoint(self.output_dir / 'encoder_best.pt', *args)
            save_checkpoint(self.output_dir / 'encoder_latest.pt', *args)
        return {'epoch': self.epoch, 'best_epoch': self.early_stop.best_epoch,
                'best_macro_f1': self.early_stop.best_macro_f1, 'best_loss': self.early_stop.best_loss,
                'stopped_early': self.early_stop.should_stop(self.epoch, self.config),
                'target_check': self.target_diagnostics,
                'encoder_best': str(self.output_dir / 'encoder_best.pt')}
