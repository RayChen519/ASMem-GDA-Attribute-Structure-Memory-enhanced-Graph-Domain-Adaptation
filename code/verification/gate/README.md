# P6 integration acceptance

From `code/`, use the project interpreter:

```powershell
../.venv/Scripts/python.exe -m verification.gate.run --output artifacts/reports/integration/NEW_ID --runs artifacts/runs/smoke/NEW_ID
```

On a Linux server, create the repository-root `.venv`, install the project
requirements (and a PyTorch build compatible with the server's CUDA stack),
acquire/prepare the datasets, then run from `code/` with
`../.venv/bin/python -m verification.gate.run` and the same arguments.
Generated datasets, checkpoints and reports are intentionally excluded from Git;
the server must create its own integration report before non-smoke training.

Each output directory must be new. The entry point runs static config, dimensions,
stage gradients (including real epoch 20/21 steps), GRL, Target sentinel, refresh/resume,
synthetic chain and ACMv9 → Citationv1 (5%, seed 0), in that order. Existing production
trainers and checkpoint validators are used throughout. No formal matrix is launched.

`configs/overrides/smoke/integration.json` limits stages to 1/2/1/2/2 epochs, K=8,
refresh interval=1, Dual unfreeze epoch=2 and Target weight ramp=2. Model dimensions,
full-domain attention and METIS P=128/seed=0 are unchanged. The next representation
refresh after the unfrozen step is checked explicitly, then persisted latest state is
restored before loading selected best for final evaluation.

Reports contain per-test JUnit identities, commands, logs, runtime/dependencies,
code/config/data hashes, checkpoint lineage, gradient audits and locked final metrics.
Failure, interruption, missing tests and pytest skips never produce PASS. Code changes
during a run invalidate it. `experiments.scheduling.admission.require_integration`
rechecks the report, current snapshot and evidence hashes before formal admission;
P8 development config, registry and run-manifest locks remain separate prerequisites.

Target sentinel tests alter only isolated evaluation files; label-file access is trapped
during training, including pseudo labels and checkpoint selection. CPU deterministic
runs compare complete tensors exactly. The synthetic graph includes empty/nonempty
pseudo-label tests in the ordered regression suite; confidence injection is confined
to that targeted test and is never applied to real-data training.

The registered_experiments group additionally checks all 18 variants, M5 untrained
artifacts, fail-closed formal admission, orchestration resume and independent
evaluation. See the repository README for the complete Linux server workflow.
