"""Standalone 2-epoch synthetic P3 smoke with P=128 (N >= P)."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from contracts.training.encoder import WarmupConfig
from models.classifiers.shared import SharedClassifier
from models.encoders.shared_gcn import SharedGCNEncoder
from training.checkpoints.encoder import build_metadata
from training.stages.encoder_warmup.trainer import EncoderWarmup
from training.stages.domain_adaptation.__main__ import run
from utils.randomness.state import seed_everything
from verification.smoke.synthetic.encoder_fixture import synthetic_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new output directory')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    seed_everything(0)
    path, source, target, validation, _ = synthetic_dataset(args.output / 'data', sizes=(256, 257))
    wc = WarmupConfig(max_epochs=1, min_epochs=1, smoke=True)
    metadata = build_metadata(path, 'A', 'B', .05, 0, source, target, wc)
    warm = EncoderWarmup(SharedGCNEncoder(), SharedClassifier(3), source, validation,
        metadata, args.output / 'encoder', wc)
    warm.fit(target)
    run(SimpleNamespace(manifest=path, source='A', target='B', label_rate=.05, seed=0,
        encoder_best=args.output / 'encoder/encoder_best.pt', output=args.output / 'da',
        cache=args.output / 'cache', device='cpu', threads=2, smoke=True, smoke_epochs=2, resume=None))


if __name__ == '__main__':
    main()
