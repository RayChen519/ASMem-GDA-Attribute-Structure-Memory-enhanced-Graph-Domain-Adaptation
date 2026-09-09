import argparse
from dataclasses import fields
from pathlib import Path
import time

import torch

from contracts.training.domain_adaptation import DAConfig
from contracts.training.encoder import WarmupConfig
from data.cache.store import file_hash, io_path, read_json, write_json
from data.partitions.metis import load_partition
from data.views.training import load_training_views
from evaluation.source_validation.data import load_source_validation
from models.classifiers.shared import SharedClassifier
from models.encoders.shared_gcn import SharedGCNEncoder
from training.checkpoints.encoder import build_metadata
from training.checkpoints.domain_adaptation import load_checkpoint
from training.stages.domain_adaptation.trainer import DomainAdaptation
from utils.randomness.state import seed_everything


def run(args):
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        raise ValueError('Output is nonempty; use a new run or explicit resume')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    seed_everything(args.seed)
    config_root = Path(__file__).resolve().parents[3] / 'configs'
    base = read_json(config_root / 'base/domain_adaptation.json')
    if args.smoke:
        base.update(read_json(config_root / 'overrides/smoke/domain_adaptation.json'))
        base['max_epochs'] = args.smoke_epochs
    config = DAConfig(**base)
    source, target = load_training_views(args.manifest, args.source, args.target, args.label_rate, args.seed)
    validation = load_source_validation(args.manifest, args.source, args.label_rate, args.seed)
    record = read_json(args.encoder_best.with_suffix('.json'))
    if file_hash(args.encoder_best) != record['sha256']:
        raise ValueError('Encoder parent file hash mismatch')
    parent = torch.load(io_path(args.encoder_best), map_location='cpu', weights_only=True)
    if parent['epoch'] != parent['early_stop_state']['best_epoch']:
        raise ValueError('Encoder parent is not its selected best epoch')
    parent_config = WarmupConfig(**{f.name: parent['metadata']['resolved_config'][f.name] for f in fields(WarmupConfig)})
    expected = build_metadata(args.manifest, args.source, args.target, args.label_rate, args.seed,
                              source, target, parent_config)
    sp = load_partition(source.graph, args.source, args.cache)
    tp = load_partition(target, args.target, args.cache)
    trainer = DomainAdaptation(SharedGCNEncoder(), SharedClassifier(expected['num_classes']), source, target,
        validation, args.encoder_best, expected, sp, tp, args.output, config, args.device)
    if args.resume:
        trainer.resume(args.resume)
    write_json(args.output / 'resolved_config.json', config.resolved())
    start = time.perf_counter()
    result = trainer.fit()
    elapsed = time.perf_counter() - start
    load_checkpoint(args.output / 'da_best.pt', trainer.modules, trainer.metadata)
    from training.stages.encoder_warmup.trainer import evaluation_mode
    with evaluation_mode(*trainer.modules.values()):
        shapes = {name: [list(t.shape) for t in trainer.representations(source=is_source)[:4]]
                  for name, is_source in [('source', True), ('target', False)]}
    report = {**result, 'scope': 'P3_stage2_only', 'smoke': config.smoke, 'device': args.device,
        'torch': torch.__version__, 'threads': args.threads, 'elapsed_seconds': elapsed,
        'target_labels_accessed': False, 'parent_checkpoint': trainer.metadata['parent_checkpoint'],
        'partitions': trainer.metadata['partitions'], 'partition_cache_hits': [sp.cache_hit, tp.cache_hit],
        'representation_shapes': shapes, 'not_run': ['Memory', 'Stages 3-5', 'final_target_evaluation', 'full_integration_gate']}
    write_json(args.output / ('resume_report.json' if args.resume else 'stage_report.json'), report)
    write_json(args.output / 'run_manifest.json', {'status': 'unlocked', 'stage': 'domain_adaptation',
        'smoke': config.smoke, 'metadata': trainer.metadata, 'checkpoint': read_json(args.output / 'da_best.json')})
    print(report)
    return report


def main():
    p = argparse.ArgumentParser(description='P3 full-domain Stage 2 DA')
    p.add_argument('--manifest', type=Path, default=Path('artifacts/datasets/manifests/dataset_manifest.json'))
    p.add_argument('--source', default='ACMv9')
    p.add_argument('--target', default='Citationv1')
    p.add_argument('--label-rate', type=float, default=.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--encoder-best', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path('artifacts/cache'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-epochs', type=int, choices=(1, 2, 3), default=2)
    p.add_argument('--resume', type=Path)
    run(p.parse_args())


if __name__ == '__main__':
    main()
