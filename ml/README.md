# ML

Phase 1: binary supervised classifier — suspicious / non_suspicious (PyTorch + HAM10000).
Phase 2: full 7-class HAM10000 classifier.
Phase 3: multi-dataset expansion (ISIC + others).
Phase 4: confidence thresholding & UX polish (rule-based threshold, visual indicators, model_version).
Phase 5 (deferred): RL for image-quality scoring or adaptive decision thresholds.

See [docs/phases.md](../docs/phases.md) for the full build plan.

## Layout
- `data/` — dataset loaders
- `models/` — model architectures + checkpoints
- `training/` — train/eval scripts
- `inference/` — load + predict helpers used by backend
