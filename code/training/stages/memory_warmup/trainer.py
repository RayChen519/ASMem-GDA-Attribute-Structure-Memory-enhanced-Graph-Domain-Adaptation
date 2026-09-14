"""Shared execution engine for P4 stages; upstream representations reuse Stage 2."""
import copy
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from contracts.training.encoder import EarlyStopState
from contracts.data import SourceTrainView
from contracts.training.memory import STAGES
from data.anchors.source import sample_anchors
from data.cache.store import digest, write_json
from data.prepare import tensor_hash
from evaluation.metrics.classification import classification_metrics
from models.memory.network import MemoryNetwork
from training.checkpoints.domain_adaptation import load_checkpoint as load_da
from training.checkpoints.memory import read_verified, reference, save, validate
from training.losses.domain_adaptation import domain_loss, sparsity_loss
from training.losses.pseudo_labels import confidence_weighted_loss
from training.optimization.memory import build, configure, rates
from training.pseudo_labels.selection import select
from training.stages.encoder_warmup.trainer import evaluation_mode, finite
from utils.randomness.state import restore_rng


class MemoryTraining:
    def __init__(self, da, source, da_best, output_dir, config, cache_root,
                 *, memory_best=None, source_pl_best=None):
        self.config, self.da = config, da
        self.output_dir = Path(output_dir)
        if (not isinstance(source, SourceTrainView) or source.split_hash != da.metadata['split_hash']
                or not torch.equal(source.train_node_id.to(da.train_ids.device), da.train_ids)
                or not torch.equal(source.train_y.to(da.train_y.device), da.train_y)
                or not torch.equal(source.graph.x.to(da.source_inputs[0].device), da.source_inputs[0])):
            raise ValueError('Source anchor view differs from the verified DA training view')
        masks = [source.graph.source_train_mask, source.graph.source_val_mask, source.graph.source_unlabeled_mask]
        if (any(m.dtype != torch.bool or m.shape != (len(source.graph.node_id),) for m in masks)
                or not torch.all(sum(m.int() for m in masks) == 1)
                or tensor_hash({'mask': masks[0]}) != da.metadata['source_train_mask_hash']
                or tensor_hash({'mask': masks[1]}) != da.metadata['source_val_mask_hash']
                or source.graph.dataset_version != da.metadata['dataset_version']
                or source.graph.attribute_union_hash != da.metadata['attribute_union_hash']):
            raise ValueError('Source anchor masks/data identity mismatch')
        if Path(da_best).stem != 'da_best':
            raise ValueError('Memory stages require da_best')
        parent = read_verified(da_best, best=True)
        if parent['metadata'] != da.metadata:
            raise ValueError('DA context does not match parent')
        if parent['metadata']['resolved_config']['smoke'] and not config.smoke:
            raise ValueError('Smoke checkpoint cannot initialize formal training')
        load_da(da_best, da.modules, da.metadata)
        self.device = da.source_inputs[0].device
        self.memory = MemoryNetwork(da.classifier.num_classes, config.temperature).to(self.device)
        self.modules = {**da.modules, 'memory': self.memory}
        self.source_ids = source.graph.node_id[source.graph.source_unlabeled_mask].to(self.device)
        self.target_ids = torch.arange(len(da.target_inputs[0]), device=self.device)
        self.anchor_state = sample_anchors(source, da.metadata, cache_root, K=config.K,
                                          alpha=config.alpha, beta=config.beta)
        self.pseudo_state, self.refresh_history = {}, []
        data_hash = tensor_hash({'source_x': da.source_inputs[0], 'target_x': da.target_inputs[0],
            'source_edges': da.source_inputs[1].coalesce().indices(),
            'source_weights': da.source_inputs[1].coalesce().values(),
            'target_edges': da.target_inputs[1].coalesce().indices(),
            'target_weights': da.target_inputs[1].coalesce().values(),
            'source_unlabeled_ids': self.source_ids})
        identity = {k: v for k, v in da.metadata.items() if k not in
                    {'stage', 'resolved_config', 'resolved_config_hash', 'parent_checkpoint',
                     'parent_metadata_hash', 'anchor_hash'}}
        identity.update(data_hash=data_hash, anchor_hash=self.anchor_state['hash'],
                        anchor_inputs=self.anchor_state['inputs'],
                        pseudo_node_ids={'source': self.source_ids.cpu().tolist(),
                                        'target': self.target_ids.cpu().tolist()})
        parents = {'da_best': reference(da_best)}
        if config.stage != 'memory_warmup':
            if memory_best is None or Path(memory_best).stem != 'memory_best':
                raise ValueError('PL stages require memory_best')
            inherited = read_verified(memory_best, best=True)
            self._check_inherited(inherited, identity, 'memory_warmup')
            if inherited['metadata']['parents'] != parents:
                raise ValueError('Memory belongs to another DA parent')
            self._load_models(inherited)
            self.anchor_state = copy.deepcopy(inherited['anchor_state'])
            parents['memory_best'] = reference(memory_best)
        if config.stage == 'dual_domain_finetuning':
            if source_pl_best is None or Path(source_pl_best).stem != 'source_pl_best':
                raise ValueError('Dual PL requires source_pl_best')
            inherited = read_verified(source_pl_best, best=True)
            self._check_inherited(inherited, identity, 'source_self_training')
            if inherited['metadata']['parents'] != parents:
                raise ValueError('Source PL belongs to another teacher/DA parent')
            latest = read_verified(Path(source_pl_best).with_name('source_pl_latest.pt'))
            validate(latest, self.modules, inherited['metadata'])
            completed = EarlyStopState(**latest['early_stop_state'])
            prior_config = type(config)(**{k: inherited['metadata']['resolved_config'][k]
                                          for k in type(config).__dataclass_fields__})
            if (latest['epoch'] < prior_config.max_epochs and not completed.should_stop(latest['epoch'], prior_config)):
                raise ValueError('Source self-training must finish before Target PL')
            if completed.best_epoch != inherited['epoch']:
                raise ValueError('Source latest/best mismatch')
            self._load_models(inherited)
            self.anchor_state = copy.deepcopy(inherited['anchor_state'])
            self.pseudo_state = copy.deepcopy(inherited['pseudo_label_state'])
            self.refresh_history = copy.deepcopy(inherited['refresh_history'])
            parents = {'source_pl_best': reference(source_pl_best), 'memory_best': reference(memory_best)}
        self.metadata = {**identity, 'stage': config.stage, 'parents': parents,
                         'resolved_config': config.resolved(), 'resolved_config_hash': digest(config.resolved())}
        self.optimizer, self.scheduler = build(self.modules, config)
        self.epoch, self.early_stop, self.history = 0, EarlyStopState(), []
        if config.stage == 'memory_warmup':
            with evaluation_mode(*self.modules.values()):
                self._update_anchors(self.da.representations(source=True))

    def _check_inherited(self, payload, identity, stage):
        meta = payload['metadata']
        if meta['stage'] != stage or any(meta.get(k) != v for k, v in identity.items()):
            raise ValueError('Inherited stage/data/split/METIS/anchor identity mismatch')
        for key in ('K', 'alpha', 'beta', 'temperature', 'gamma', 'q', 'smoke'):
            if meta['resolved_config'][key] != getattr(self.config, key):
                raise ValueError('Inherited teacher configuration mismatch')
        validate(payload, self.modules, meta)

    def _load_models(self, payload):
        for name, module in self.modules.items():
            module.load_state_dict(payload['model_state'][name], strict=True)

    @torch.no_grad()
    def _update_anchors(self, source):
        ids = self.anchor_state['ids'].to(self.device)
        h_s, h_as = source.h_s[ids].detach(), source.h_as[ids].detach()
        representations = {'h_s': h_s, 'h_as': h_as,
                           'key': F.normalize(self.memory.key(h_s), dim=-1),
                           'value': self.memory.value(h_as)}
        self.anchor_state['representations'] = {k: v.detach().cpu().clone() for k, v in representations.items()}
        self.anchor_state['representation_hash'] = tensor_hash(self.anchor_state['representations'])

    def memory_output(self, output, ids, *, source):
        bank = self.anchor_state['representations']
        return self.memory(output.h_s[ids], output.h_as[ids], bank['h_s'].to(self.device),
                           bank['h_as'].to(self.device), ids,
                           self.anchor_state['ids'].to(self.device), source=source)

    def refresh(self, epoch):
        if self.config.stage == 'memory_warmup':
            raise ValueError('No pseudo labels in Memory warm-up')
        with evaluation_mode(*self.modules.values()):
            source = self.da.representations(source=True)
            self._update_anchors(source)
            domains = [('source', source, self.source_ids)]
            if self.config.stage == 'dual_domain_finetuning':
                domains.append(('target', self.da.representations(source=False), self.target_ids))
            for domain, output, ids in domains:
                old = self.pseudo_state.get(domain)
                logits = self.da.classifier(output.h_as[ids]).detach()
                probability = self.memory_output(output, ids, source=domain == 'source').probability.detach()
                state = select(probability, logits, ids, gamma=self.config.gamma, q=self.config.q,
                               previous=None if old is None else old['labels'].to(self.device),
                               refresh_index=0 if old is None else old['refresh_index'] + 1)
                state.update(classifier_logits=logits, epoch=epoch, stage=self.config.stage,
                             anchor_representation_hash=self.anchor_state['representation_hash'])
                state = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
                self.pseudo_state[domain] = state
                self.refresh_history.append({'domain': domain, 'epoch': epoch, 'stage': self.config.stage,
                    'refresh_index': state['refresh_index'], 'coverage': state['coverage'],
                    'anchor_representation_hash': self.anchor_state['representation_hash']})

    def schedule(self, epoch):
        dual = self.config.stage == 'dual_domain_finetuning'
        progress = (epoch - 1) / max(1, self.config.max_epochs - 1) if dual else 0.
        return {'progress': progress, 'lambda_adv': self.config.lambda_adv_max *
                (2 / (1 + math.exp(-10 * progress)) - 1) if dual else 0.,
                'lambda_t': self.config.lambda_t_max * min(1., (epoch - 1) / 29) if dual else 0.,
                'coefficient': 1.}

    def train_step(self, epoch):
        if [g['name'] for g in self.optimizer.param_groups] != list(rates(self.config.stage, epoch)):
            self.optimizer, self.scheduler = build(self.modules, self.config, epoch,
                                                   self.optimizer, completed_epochs=epoch-1)
        configure(self.modules, self.config, epoch)
        if self.config.stage != 'memory_warmup' and (epoch - 1) % self.config.refresh_interval == 0:
            self.refresh(epoch)
        self.optimizer.zero_grad(set_to_none=True)
        source = self.da.representations(source=True)
        schedule = self.schedule(epoch)
        if self.config.stage == 'memory_warmup':
            logits = self.memory_output(source, self.da.train_ids, source=True).logits
            total = F.cross_entropy(logits, self.da.train_y)
            terms = {'memory_loss': total.item()}
        else:
            logits = self.da.classifier(source.h_as)
            supervised = F.cross_entropy(logits[self.da.train_ids], self.da.train_y)
            source_pl = confidence_weighted_loss(logits, self.pseudo_state['source'])
            total = supervised + self.config.lambda_s * source_pl
            terms = {'supervised_loss': supervised.item(), 'source_pl_loss': source_pl.item()}
            if self.config.stage == 'dual_domain_finetuning':
                target = self.da.representations(source=False)
                target_pl = confidence_weighted_loss(self.da.classifier(target.h_as), self.pseudo_state['target'])
                dom, sparse = domain_loss(self.da.discriminator, source, target), sparsity_loss(source)
                total = total + schedule['lambda_t'] * target_pl + schedule['lambda_adv'] * dom + self.config.lambda_sp * sparse
                terms.update(target_pl_loss=target_pl.item(), domain_loss=dom.item(), sparsity_loss=sparse.item())
        finite(total, 'Memory/PL loss')
        total.backward()
        parameters = []
        optimizer_ids = {id(p) for g in self.optimizer.param_groups for p in g['params']}
        for module in self.modules.values():
            for p in module.parameters():
                if p.requires_grad:
                    if p.grad is None or id(p) not in optimizer_ids:
                        raise ValueError('Missing trainable gradient/optimizer parameter')
                    finite(p.grad, 'Memory/PL gradient')
                    parameters.append(p)
                elif p.grad is not None or id(p) in optimizer_ids:
                    raise ValueError('Frozen parameter has gradient or optimizer membership')
        torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip_norm, error_if_nonfinite=True)
        self.optimizer.step()
        if self.config.stage == 'memory_warmup':
            # The trainable Key/Value projections changed; persist current projections.
            with evaluation_mode(*self.modules.values()):
                self._update_anchors(source)
        return {'train_loss': total.item(), **terms, 'schedule': schedule}

    def validate(self):
        with evaluation_mode(*self.modules.values()):
            source = self.da.representations(source=True)
            logits = (self.memory_output(source, self.da.val_ids, source=True).logits
                      if self.config.stage == 'memory_warmup' else self.da.classifier(source.h_as[self.da.val_ids]))
            return classification_metrics(logits, self.da.val_y, self.da.classifier.num_classes)

    def fit(self, until_epoch=None):
        if self.epoch == 0 and self.output_dir.exists() and any(self.output_dir.glob('*.pt')):
            raise ValueError('Existing checkpoints require explicit resume')
        end = min(self.config.max_epochs, until_epoch or self.config.max_epochs)
        best_name = STAGES[self.config.stage][3]
        while self.epoch < end and not self.early_stop.should_stop(self.epoch, self.config):
            epoch = self.epoch + 1
            step = self.train_step(epoch)
            metrics = self.validate()
            improved = self.early_stop.update(epoch, metrics['macro_f1'], metrics['loss'])
            self.scheduler.step()
            self.epoch = epoch
            self.history.append({'epoch': epoch, **step, 'source_validation': metrics})
            if improved:
                save(self.output_dir / (best_name + '.pt'), self)
            save(self.output_dir / (best_name.replace('_best', '_latest') + '.pt'), self)
        result = {'stage': self.config.stage, 'epoch': self.epoch, 'best_epoch': self.early_stop.best_epoch,
                  'checkpoint': str(self.output_dir / (best_name + '.pt')), 'integration_gate': 'NOT_RUN',
                  'device': str(self.device)}
        write_json(self.output_dir / 'stage_report.json', result)
        return result

    def resume(self, path):
        if Path(path).resolve().parent != self.output_dir.resolve():
            raise ValueError('Resume requires original run directory')
        payload = read_verified(path)
        early = validate(payload, self.modules, self.metadata)
        best = read_verified(self.output_dir / (STAGES[self.config.stage][3] + '.pt'), best=True)
        validate(best, self.modules, self.metadata)
        if (best['epoch'] != early.best_epoch or best['early_stop_state']['best_macro_f1'] != early.best_macro_f1
                or best['early_stop_state']['best_loss'] != early.best_loss):
            raise ValueError('Resume best checkpoint mismatch')
        if len(payload['rng_state']['cuda']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
            raise ValueError('Resume CUDA topology mismatch')
        self._load_models(payload)
        self.optimizer, self.scheduler = build(self.modules, self.config, payload['epoch'])
        self.optimizer.load_state_dict(payload['optimizer_state'])
        self.scheduler.load_state_dict(payload['scheduler_state'])
        self.epoch, self.early_stop, self.history = payload['epoch'], early, payload['history']
        self.anchor_state = payload['anchor_state']
        self.pseudo_state, self.refresh_history = payload['pseudo_label_state'], payload['refresh_history']
        restore_rng(payload['rng_state'])
