"""Run from code/: python -m training.stages.encoder_warmup --help."""
import argparse
from pathlib import Path

import torch

from contracts.training.encoder import WarmupConfig
from data.cache.store import file_hash, read_json, write_json
from data.views.training import load_training_views
from evaluation.source_validation.data import load_source_validation
from models.classifiers.shared import SharedClassifier
from models.encoders.shared_gcn import SharedGCNEncoder
from training.checkpoints.encoder import build_metadata, load_checkpoint
from training.stages.encoder_warmup.trainer import EncoderWarmup
from utils.randomness.state import seed_everything


def main():
    parser = argparse.ArgumentParser(description='Source-only shared GCN warm-up')
    parser.add_argument('--manifest', type=Path, default=Path('artifacts/datasets/manifests/dataset_manifest.json'))
    parser.add_argument('--source', default='ACMv9')
    parser.add_argument('--target', default='Citationv1')
    parser.add_argument('--label-rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        raise ValueError('Output directory is nonempty; choose a new run or explicitly resume')
    if args.resume is not None and args.resume.resolve().parent != args.output.resolve():
        raise ValueError('Resume in the original run directory to preserve encoder_best')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    seed_everything(args.seed)
    config = WarmupConfig(max_epochs=2, min_epochs=1, patience=15, smoke=True) if args.smoke else WarmupConfig()
    source, target = load_training_views(args.manifest, args.source, args.target, args.label_rate, args.seed)
    validation = load_source_validation(args.manifest, args.source, args.label_rate, args.seed)
    metadata = build_metadata(args.manifest, args.source, args.target, args.label_rate,
                              args.seed, source, target, config)
    encoder, classifier = SharedGCNEncoder(), SharedClassifier(metadata['num_classes'])
    trainer = EncoderWarmup(encoder, classifier, source, validation, metadata, args.output, config, args.device)
    if args.resume:
        trainer.resume(args.resume)
    result = trainer.fit(target)
    # Verify that the selected artifact can initialize the existing DA modules.
    inherited = load_checkpoint(args.output / 'encoder_best.pt', encoder, classifier, metadata)
    write_json(args.output / 'resolved_config.json', config.resolved())
    write_json(args.output / 'stage_report.json', {**result, 'scope': 'encoder_warmup_only',
               'integration_gate': 'NOT_RUN', 'smoke': args.smoke,
               'parent_for_da': inherited['parent_checkpoint'], 'target_labels_accessed': False})
    write_json(args.output / 'run_manifest.json', {
        'variant': 'B0', 'status': 'unlocked', 'stage': 'encoder_warmup',
        'source': args.source, 'target': args.target, 'label_rate': args.label_rate,
        'split_seed': args.seed, 'split_hash': source.split_hash,
        'dataset_version': metadata['dataset_version'],
        'dataset_manifest_sha256': file_hash(args.manifest),
        'resolved_config_hash': metadata['resolved_config_hash'],
        'final_checkpoint': read_json(args.output / 'encoder_best.json'), 'smoke': args.smoke})
    print(result)


if __name__ == '__main__':
    main()
