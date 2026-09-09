"""Persist a reproducible Stage-1 smoke, including Dataset views and DA handoff."""
import argparse
from pathlib import Path

import torch

from contracts.training.encoder import WarmupConfig
from data.cache.store import write_json
from models.classifiers.shared import SharedClassifier
from models.encoders.shared_gcn import SharedGCNEncoder
from training.checkpoints.encoder import build_metadata, load_checkpoint
from training.stages.encoder_warmup.trainer import EncoderWarmup
from utils.randomness.state import seed_everything
from verification.smoke.synthetic.encoder_fixture import synthetic_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new synthetic smoke output directory')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_everything(0)
    path, source, target, validation, _ = synthetic_dataset(args.output / 'dataset')
    config = WarmupConfig(max_epochs=2, min_epochs=1, smoke=True)
    metadata = build_metadata(path, 'A', 'B', .05, 0, source, target, config)
    encoder, classifier = SharedGCNEncoder(), SharedClassifier(3)
    trainer = EncoderWarmup(encoder, classifier, source, validation, metadata, args.output, config)
    result = trainer.fit(target)
    inherited = load_checkpoint(args.output / 'encoder_best.pt', encoder, classifier, metadata)
    report = {**result, 'scope': 'encoder_warmup_only', 'integration_gate': 'NOT_RUN',
              'source_nodes': 9, 'target_nodes': 7, 'feature_dim': 6775,
              'includes_isolates_and_zero_features': True,
              'parent_for_da': inherited['parent_checkpoint']}
    write_json(args.output / 'stage_report.json', report)
    print(report)


if __name__ == '__main__':
    main()
