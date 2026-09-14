"""P6 real/synthetic short chain. All training uses the production trainers."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from contracts.training.encoder import WarmupConfig
from contracts.training.domain_adaptation import DAConfig
from contracts.training.memory import MemoryConfig
from data.cache.store import digest, file_hash, read_json, write_json
from data.partitions.metis import load_partition
from data.views.training import load_training_views
from evaluation.source_validation.data import load_source_validation
from evaluation.final_target.data import load_target_evaluation
from evaluation.metrics.classification import classification_metrics
from models.encoders.shared_gcn import SharedGCNEncoder
from models.classifiers.shared import SharedClassifier
from training.checkpoints.encoder import build_metadata, load_checkpoint as load_encoder
from training.checkpoints.domain_adaptation import load_checkpoint as load_da
from training.checkpoints.memory import read_verified, validate, reference
from training.stages.encoder_warmup.trainer import EncoderWarmup, evaluation_mode
from training.stages.domain_adaptation.trainer import DomainAdaptation
from training.stages.memory_warmup.trainer import MemoryTraining
from utils.randomness.state import seed_everything


def gradient_audit(modules, optimizer):
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids) == len(set(ids)), 'Duplicate optimizer parameters'
    assert set(ids) == {id(p) for m in modules.values() for p in m.parameters() if p.requires_grad}
    result = {}
    for name, module in modules.items():
        for p in module.parameters():
            assert (p.grad is not None and torch.isfinite(p.grad).all()) if p.requires_grad else p.grad is None
        result[name] = {'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad),
                        'frozen': sum(p.numel() for p in module.parameters() if not p.requires_grad)}
    return result


def run(manifest, output, config_path, *, source_name=None, target_name=None, device='cpu', threads=2):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty independent smoke directory')
    cfg = read_json(config_path)
    assert (cfg['label_rate'], cfg['seed']) == (.05, 0)
    assert all(1 <= cfg[k] <= 2 for k in ('encoder_epochs','da_epochs','memory_epochs','source_epochs','dual_epochs'))
    assert cfg['unfreeze_epoch'] == cfg['dual_epochs'] == 2 and cfg['refresh_interval'] == 1
    source_name, target_name = source_name or cfg['source'], target_name or cfg['target']
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    seed_everything(0)
    start = time.perf_counter()
    source, target = load_training_views(manifest, source_name, target_name, .05, 0)
    assert not hasattr(target, 'y')
    validation = load_source_validation(manifest, source_name, .05, 0)
    wc = WarmupConfig(max_epochs=cfg['encoder_epochs'], min_epochs=1, smoke=True)
    metadata = build_metadata(manifest, source_name, target_name, .05, 0, source, target, wc)
    warm = EncoderWarmup(SharedGCNEncoder(), SharedClassifier(metadata['num_classes']), source,
                         validation, metadata, output/'encoder', wc, device)
    checks, lineage, configs = [], [], {'encoder': wc.resolved()}
    def record(name, trainer, modules, path):
        checks.append({'stage': name, 'epochs': trainer.epoch,
                       'gradients': gradient_audit(modules, trainer.optimizer)})
        payload = read_verified(path, best=True)
        assert payload['optimizer_state']['state'] and payload['scheduler_state']['last_epoch'] == payload['epoch']
        lineage.append({**reference(path), 'metadata': payload['metadata'],
                        'best_epoch': payload['epoch'], 'completed_epochs': trainer.epoch,
                        'validated': True})
        print(json.dumps({'stage': name, 'status': 'PASS', 'epoch': trainer.epoch}), flush=True)
    warm.fit(target)
    ep = output/'encoder/encoder_best.pt'
    record('encoder', warm, {'shared_gcn':warm.encoder,'classifier':warm.classifier}, ep)
    warm.resume(output/'encoder/encoder_latest.pt')
    load_encoder(ep, warm.encoder, warm.classifier, metadata)
    parts = (load_partition(source.graph, source_name, output/'cache'),
             load_partition(target, target_name, output/'cache'))
    dc = DAConfig(max_epochs=cfg['da_epochs'], min_epochs=1, smoke=True)
    configs['da'] = dc.resolved()
    da = DomainAdaptation(warm.encoder, warm.classifier, source, target, validation, ep,
                          metadata, *parts, output/'da', dc, device)
    da.fit()
    dp = output/'da/da_best.pt'
    record('da', da, da.modules, dp)
    da.resume(output/'da/da_latest.pt')
    load_da(dp, da.modules, da.metadata)
    for stage, folder, epochs, best in [('memory_warmup','memory','memory_epochs','memory_best'),
            ('source_self_training','source','source_epochs','source_pl_best'),
            ('dual_domain_finetuning','dual','dual_epochs','full_best')]:
        mc = MemoryConfig(stage=stage, max_epochs=cfg[epochs], min_epochs=1, smoke=True,
                          K=cfg['K'], refresh_interval=cfg['refresh_interval'],
                          unfreeze_epoch=cfg['unfreeze_epoch'], target_ramp_epochs=cfg['target_ramp_epochs'])
        configs[folder] = mc.resolved()
        trainer = MemoryTraining(da, source, dp, output/folder, mc, output/'cache',
                    memory_best=output/'memory/memory_best.pt', source_pl_best=output/'source/source_pl_best.pt')
        # Capture both sides of the real smoke transition, before resume clears gradients.
        trainer.fit(until_epoch=1)
        first = gradient_audit(trainer.modules, trainer.optimizer)
        trainer.fit()
        record(folder, trainer, trainer.modules, output/folder/(best+'.pt'))
        checks[-1]['epoch1_gradients'] = first
        if folder == 'dual':
            assert first['shared_gcn']['trainable'] == 0
            assert checks[-1]['gradients']['shared_gcn']['trainable'] > 0
            assert len([r for r in trainer.refresh_history if r['domain']=='target']) == 2
            before = trainer.anchor_state['representation_hash']
            ids = trainer.anchor_state['ids'].clone()
            # Next refresh after the unfrozen optimizer step: representation changes, IDs do not.
            trainer.refresh(3)
            assert before != trainer.anchor_state['representation_hash']
            assert torch.equal(ids, trainer.anchor_state['ids'])
            checks[-1]['post_unfreeze_refresh'] = True
        trainer.resume(output/folder/(best.replace('_best','_latest')+'.pt'))
        payload = read_verified(output/folder/(best+'.pt'), best=True)
        validate(payload, trainer.modules, trainer.metadata)
        trainer._load_models(payload)
    cp = output/'dual/full_best.pt'
    run_manifest = {'run_id':f'smoke__B6__{source_name}-to-{target_name}__rate-0.05__seed-0',
        'tier':'smoke','variant':'B6','source':source_name,'target':target_name,'label_rate':.05,'split_seed':0,
        'dataset_version':metadata['dataset_version'],'dataset_manifest_sha256':file_hash(manifest),
        'split_hash':metadata['split_hash'],'resolved_configs':configs,'config_hash':digest(configs),
        'anchor_hash':trainer.metadata['anchor_hash'],'checkpoint_lineage':lineage}
    write_json(output/'run_manifest.json', run_manifest)
    lock = {'status':'locked','run_manifest_sha256':file_hash(output/'run_manifest.json'),
            'final_checkpoint':{'name':'full_best','path':'dual/full_best.pt','sha256':file_hash(cp)}}
    write_json(output/'run_lock.json', {**lock,'lock_hash':digest(lock)})
    evaluation = load_target_evaluation(manifest, output/'run_manifest.json', output/'run_lock.json')
    assert torch.equal(evaluation.node_id, target.node_id)
    with evaluation_mode(*trainer.modules.values()):
        logits = da.classifier(da.representations(source=False).h_as).cpu()
    metrics = classification_metrics(logits, evaluation.y, metadata['num_classes'])
    report = {'status':'PASS','device':device,'torch':str(torch.__version__),'threads':threads,
              'elapsed_seconds':time.perf_counter()-start,'dimensions':{'source':[len(source.graph.x),6775],
              'target':[len(target.x),6775],'C':metadata['num_classes'],'K':cfg['K']},
              'checks':checks,'checkpoint_lineage':lineage,'config_hash':digest(configs),
              'run_manifest':str(output/'run_manifest.json'),'lock':str(output/'run_lock.json'),
              'final_evaluation':{'checkpoint_hash':evaluation.checkpoint_hash, **metrics}}
    write_json(output/'chain_report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/overrides/smoke/integration.json'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    run(a.manifest,a.output,a.config,device=a.device,threads=a.threads)
