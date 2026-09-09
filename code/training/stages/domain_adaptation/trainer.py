"""Stage 2 only: full-domain A/S DA; no Memory or Target class labels."""
from pathlib import Path

import torch

from contracts.data import SourceTrainView, SourceValidationView, TargetTrainView
from contracts.training.domain_adaptation import DAConfig, adversarial_weight
from contracts.training.encoder import EarlyStopState
from data.cache.store import digest, write_json
from data.prepare import tensor_hash
from evaluation.metrics.classification import classification_metrics
from models.adversarial.domain import DomainDiscriminator
from models.fusion.attribute_structure import AttributeStructure
from training.checkpoints.encoder import load_checkpoint as load_encoder
from training.checkpoints.domain_adaptation import load_checkpoint, save_checkpoint
from training.losses.domain_adaptation import classification_loss, domain_loss, sparsity_loss
from training.optimization.domain_adaptation import build_optimizer
from training.schedules.warmup_cosine import build_scheduler
from training.stages.encoder_warmup.trainer import evaluation_mode, finite, graph_inputs


class DomainAdaptation:
    def __init__(self, encoder, classifier, source, target, validation, parent_path,
                 parent_metadata, source_partition, target_partition, output_dir,
                 config=None, device='cpu'):
        self.config = config or DAConfig()
        if (not isinstance(source, SourceTrainView) or not isinstance(target, TargetTrainView)
                or not isinstance(validation, SourceValidationView)):
            raise TypeError('DA requires isolated train/Source validation views')
        if Path(parent_path).stem != 'encoder_best':
            raise ValueError('DA must inherit encoder_best')
        if parent_metadata['resolved_config']['smoke'] and not self.config.smoke:
            raise ValueError('A smoke parent cannot initialize formal training')
        sg = source.graph
        for graph, domain, partition in [(sg, parent_metadata['source'], source_partition),
                                          (target, parent_metadata['target'], target_partition)]:
            if (graph.dataset_version != parent_metadata['dataset_version']
                    or graph.attribute_union_hash != parent_metadata['attribute_union_hash']):
                raise ValueError('DA dataset/vocabulary mismatch')
            partition.validate(len(graph.node_id))
            if partition.inputs != dict(dataset_version=graph.dataset_version, domain=domain,
                                         P=self.config.P, metis_seed=self.config.metis_seed):
                raise ValueError('DA METIS inputs mismatch')
        if not torch.all(target.target_unlabeled_mask):
            raise ValueError('All Target nodes must be unlabeled')
        if source.split_hash != validation.split_hash or source.split_hash != parent_metadata['split_hash']:
            raise ValueError('DA Source split mismatch')
        masks = [sg.source_train_mask, sg.source_val_mask, sg.source_unlabeled_mask]
        if any(m.dtype != torch.bool or m.shape != (len(sg.node_id),) for m in masks):
            raise ValueError('Invalid DA Source masks')
        if not torch.all(sum(m.int() for m in masks) == 1):
            raise ValueError('DA Source masks must be disjoint and exhaustive')
        for mask, ids, y, key in [(masks[0], source.train_node_id, source.train_y, 'source_train_mask_hash'),
                                  (masks[1], validation.node_id, validation.validation_y, 'source_val_mask_hash')]:
            if (ids.dtype != torch.long or not torch.equal(ids.cpu(), torch.where(mask.cpu())[0])
                    or y.dtype != torch.long or y.shape != ids.shape or y.numel() == 0
                    or y.min() < 0 or y.max() >= classifier.num_classes
                    or torch.unique(y).numel() != classifier.num_classes
                    or tensor_hash({'mask': mask}) != parent_metadata[key]):
                raise ValueError('DA Source label/mask contract mismatch')
        self.source_inputs, self.target_inputs = graph_inputs(sg, device), graph_inputs(target, device)
        inherited = load_encoder(parent_path, encoder, classifier, parent_metadata)
        if inherited['epoch'] != inherited['early_stop'].best_epoch:
            raise ValueError('Encoder parent must be the Source-validation selected best epoch')
        # Objects and Parameter identities are inherited, never replaced.
        self.encoder, self.classifier = encoder.to(device), classifier.to(device)
        self.attribute_structure, self.discriminator = AttributeStructure().to(device), DomainDiscriminator().to(device)
        self.modules = {'shared_gcn': self.encoder, 'attribute_structure': self.attribute_structure,
                        'classifier': self.classifier, 'discriminator': self.discriminator}
        for module in self.modules.values():
            module.requires_grad_(True)
        self.source_partition, self.target_partition = source_partition, target_partition
        self.train_ids, self.train_y = source.train_node_id.to(device), source.train_y.to(device)
        self.val_ids, self.val_y = validation.node_id.to(device), validation.validation_y.to(device)
        partitions = {domain: {'inputs': p.inputs, 'hash': p.fingerprint} for domain, p in
                      [(parent_metadata['source'], source_partition), (parent_metadata['target'], target_partition)]}
        self.metadata = {**parent_metadata, 'stage': 'domain_adaptation',
            'resolved_config': self.config.resolved(), 'resolved_config_hash': digest(self.config.resolved()),
            'parent_checkpoint': inherited['parent_checkpoint'], 'parent_metadata_hash': digest(parent_metadata),
            'partitions': partitions, 'metis_hash': digest(partitions), 'anchor_hash': None}
        self.optimizer = build_optimizer(self.modules, self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        self.early_stop, self.epoch, self.history = EarlyStopState(), 0, []
        self.output_dir = Path(output_dir)

    def representations(self, *, source):
        """H^A, H^LG, H^S, H^AS in input node_id order, each [N,128]."""
        inputs = self.source_inputs if source else self.target_inputs
        partition = self.source_partition if source else self.target_partition
        z = self.encoder(*inputs).z
        output = self.attribute_structure(z, partition, source=source)
        for name, tensor in zip(('H^A', 'H^LG', 'H^S', 'H^AS'), output[:4]):
            if tensor.shape != (len(z), 128):
                raise ValueError(f'DA shape mismatch: {name}')
            finite(tensor, name)
        return output

    def train_step(self, epoch):
        for module in self.modules.values():
            module.train()
        self.optimizer.zero_grad(set_to_none=True)
        source, target = self.representations(source=True), self.representations(source=False)
        cls = classification_loss(self.classifier, source, self.train_ids, self.train_y)
        dom = domain_loss(self.discriminator, source, target)
        sparse = sparsity_loss(source)
        progress, weight = adversarial_weight(epoch, self.config)
        total = cls + weight * dom + self.config.lambda_sp * sparse
        finite(total, 'DA total loss')
        total.backward()
        parameters = [p for module in self.modules.values() for p in module.parameters()]
        for p in parameters:
            if not p.requires_grad or p.grad is None:
                raise ValueError('DA trainable parameter has no gradient')
            finite(p.grad, 'DA gradient')
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip_norm, error_if_nonfinite=True)
        self.optimizer.step()
        return {'train_loss': total.item(), 'classification_loss': cls.item(), 'domain_loss': dom.item(),
                'sparsity_loss': sparse.item(), 'grad_norm': norm.item(), 'grl_progress': progress,
                'lambda_adv': weight}

    def validate(self):
        with evaluation_mode(*self.modules.values()):
            output = self.representations(source=True)
            return classification_metrics(self.classifier(output.h_as[self.val_ids]), self.val_y,
                                          self.classifier.num_classes)

    def resume(self, path):
        if Path(path).resolve().parent != self.output_dir.resolve():
            raise ValueError('Resume in the original run directory to preserve da_best')
        if not (self.output_dir / 'da_best.pt').is_file():
            raise ValueError('Resume requires da_best')
        candidate = load_checkpoint(path, self.modules, self.metadata, optimizer=self.optimizer,
                                    scheduler=self.scheduler, resume=True, validate_only=True)
        best = load_checkpoint(self.output_dir / 'da_best.pt', self.modules, self.metadata, validate_only=True)
        if (best['epoch'] != candidate['early_stop'].best_epoch
                or best['early_stop'].best_macro_f1 != candidate['early_stop'].best_macro_f1
                or best['early_stop'].best_loss != candidate['early_stop'].best_loss):
            raise ValueError('Resume da_best/early-stop state mismatch')
        state = load_checkpoint(path, self.modules, self.metadata, optimizer=self.optimizer,
                                scheduler=self.scheduler, resume=True)
        self.epoch, self.early_stop, self.history = state['epoch'], state['early_stop'], state['history']

    def fit(self, until_epoch=None):
        end = self.config.max_epochs if until_epoch is None else min(until_epoch, self.config.max_epochs)
        while self.epoch < end and not self.early_stop.should_stop(self.epoch, self.config):
            epoch = self.epoch + 1
            rates = [g['lr'] for g in self.optimizer.param_groups]
            try:
                step = self.train_step(epoch)
                metrics = self.validate()
            except (RuntimeError, MemoryError) as error:
                if isinstance(error, MemoryError) or any(s in str(error).lower() for s in ('out of memory', 'not enough memory', 'allocate memory')):
                    write_json(self.output_dir / 'oom_evidence.json', {'stage': 'domain_adaptation',
                        'epoch': epoch, 'error': repr(error), 'device': str(self.source_inputs[0].device),
                        'source_shape': list(self.source_inputs[0].shape), 'target_shape': list(self.target_inputs[0].shape),
                        'config': self.config.resolved(), 'automatic_fallback': False,
                        'required_separate_variant': 'attribute_partitioned_attention_oom'})
                raise
            improved = self.early_stop.update(epoch, metrics['macro_f1'], metrics['loss'])
            self.scheduler.step()
            self.epoch = epoch
            self.history.append({'epoch': epoch, **step, 'lr': rates, 'source_validation': metrics})
            args = (self.modules, self.optimizer, self.scheduler, self.early_stop, epoch, self.metadata, self.history)
            if improved:
                save_checkpoint(self.output_dir / 'da_best.pt', *args)
            save_checkpoint(self.output_dir / 'da_latest.pt', *args)
        return {'epoch': self.epoch, 'best_epoch': self.early_stop.best_epoch,
                'best_macro_f1': self.early_stop.best_macro_f1, 'best_loss': self.early_stop.best_loss,
                'stopped_early': self.early_stop.should_stop(self.epoch, self.config),
                'da_best': str(self.output_dir / 'da_best.pt'), 'integration_gate': 'NOT_RUN'}
