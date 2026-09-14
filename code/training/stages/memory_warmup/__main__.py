"""CLI sharing the verified Dataset, encoder, DA and METIS contracts."""
import argparse
from pathlib import Path

import torch

from contracts.training.encoder import WarmupConfig
from contracts.training.domain_adaptation import DAConfig
from contracts.training.memory import MemoryConfig, STAGES
from data.cache.store import read_json, write_json
from data.partitions.metis import load_partition
from data.views.training import load_training_views
from evaluation.source_validation.data import load_source_validation
from models.encoders.shared_gcn import SharedGCNEncoder
from models.classifiers.shared import SharedClassifier
from training.checkpoints.encoder import build_metadata
from training.checkpoints.memory import read_verified
from training.stages.domain_adaptation.trainer import DomainAdaptation
from training.stages.memory_warmup.trainer import MemoryTraining
from utils.randomness.state import seed_everything


def context(manifest, da_best, cache, device='cpu'):
    parent = read_verified(da_best, best=True)
    meta = parent['metadata']
    source, target = load_training_views(manifest, meta['source'], meta['target'], meta['label_rate'], meta['seed'])
    validation = load_source_validation(manifest, meta['source'], meta['label_rate'], meta['seed'])
    encoder_path = Path(meta['parent_checkpoint']['path'])
    encoder_payload = read_verified(encoder_path, best=True)
    wc = WarmupConfig(**{k: encoder_payload['metadata']['resolved_config'][k] for k in WarmupConfig.__dataclass_fields__})
    expected = build_metadata(manifest, meta['source'], meta['target'], meta['label_rate'], meta['seed'], source, target, wc)
    dc = DAConfig(**{k: meta['resolved_config'][k] for k in DAConfig.__dataclass_fields__})
    parts = (load_partition(source.graph, meta['source'], cache), load_partition(target, meta['target'], cache))
    da = DomainAdaptation(SharedGCNEncoder(), SharedClassifier(meta['num_classes']), source, target,
                          validation, encoder_path, expected, *parts, Path(da_best).parent, dc, device)
    if da.metadata != meta:
        raise ValueError('Live dataset/config/METIS differs from da_best')
    return da, source


def main(stage='memory_warmup'):
    parser = argparse.ArgumentParser(description=stage)
    parser.add_argument('--manifest', type=Path, default=Path('artifacts/datasets/manifests/dataset_manifest.json'))
    parser.add_argument('--da-best', type=Path, required=True)
    parser.add_argument('--memory-best', type=Path)
    parser.add_argument('--source-pl-best', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('artifacts/cache'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--integration-report', type=Path)
    parser.add_argument('--smoke-epochs', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    from experiments.scheduling.admission import check_cli
    check_cli(args)
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        raise ValueError('Use a new output directory or explicitly resume')
    config_root = Path(__file__).resolve().parents[3] / 'configs'
    values = read_json(config_root / 'base' / (stage + '.json'))
    if args.smoke:
        values.update(read_json(config_root / 'overrides/smoke/memory.json'))
        values['max_epochs'] = args.smoke_epochs
    config = MemoryConfig(**values)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    da, source = context(args.manifest, args.da_best, args.cache, args.device)
    seed_everything(da.metadata['seed'])
    trainer = MemoryTraining(da, source, args.da_best, args.output, config, args.cache,
                             memory_best=args.memory_best, source_pl_best=args.source_pl_best)
    if args.resume:
        trainer.resume(args.resume)
    write_json(args.output / 'resolved_config.json', config.resolved())
    result = trainer.fit()
    write_json(args.output / 'run_manifest.json', {'status': 'unlocked', 'smoke': config.smoke,
                'metadata': trainer.metadata, 'checkpoint': read_json(args.output / (STAGES[stage][3] + '.json'))})
    print(result)


if __name__ == '__main__':
    main()
