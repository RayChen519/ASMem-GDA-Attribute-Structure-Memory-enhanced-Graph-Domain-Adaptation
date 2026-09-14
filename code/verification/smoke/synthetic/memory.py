"""Isolated synthetic encoder -> DA -> Memory -> Source PL -> Dual PL smoke."""
import argparse
from pathlib import Path

import torch

from contracts.training.encoder import WarmupConfig
from contracts.training.domain_adaptation import DAConfig
from contracts.training.memory import MemoryConfig
from data.partitions.metis import load_partition
from models.encoders.shared_gcn import SharedGCNEncoder
from models.classifiers.shared import SharedClassifier
from training.checkpoints.encoder import build_metadata
from training.stages.encoder_warmup.trainer import EncoderWarmup
from training.stages.domain_adaptation.trainer import DomainAdaptation
from training.stages.memory_warmup.trainer import MemoryTraining
from utils.randomness.state import seed_everything
from verification.smoke.synthetic.encoder_fixture import synthetic_dataset


def run(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new smoke output directory')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    seed_everything(0)
    manifest, source, target, validation, _ = synthetic_dataset(output / 'data', sizes=(256, 257))
    wc = WarmupConfig(max_epochs=1, min_epochs=1, smoke=True)
    metadata = build_metadata(manifest, 'A', 'B', .05, 0, source, target, wc)
    warm = EncoderWarmup(SharedGCNEncoder(), SharedClassifier(3), source, validation,
                         metadata, output / 'encoder', wc)
    warm.fit(target)
    parts = (load_partition(source.graph, 'A', output / 'cache'), load_partition(target, 'B', output / 'cache'))
    da = DomainAdaptation(warm.encoder, warm.classifier, source, target, validation,
                          output / 'encoder/encoder_best.pt', metadata, *parts, output / 'da',
                          DAConfig(max_epochs=1, min_epochs=1, smoke=True))
    da.fit()
    results = []
    for stage, folder in [('memory_warmup', 'memory'), ('source_self_training', 'source'),
                          ('dual_domain_finetuning', 'dual')]:
        config = MemoryConfig(stage=stage, max_epochs=3, min_epochs=1, smoke=True, K=8, refresh_interval=1)
        trainer = MemoryTraining(da, source, output / 'da/da_best.pt', output / folder, config,
                                 output / 'cache', memory_best=output / 'memory/memory_best.pt',
                                 source_pl_best=output / 'source/source_pl_best.pt')
        results.append(trainer.fit())
        trainer.resume(output / folder / {'memory': 'memory_latest.pt', 'source': 'source_pl_latest.pt',
                                         'dual': 'full_latest.pt'}[folder])
    print(results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
