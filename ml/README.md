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

## Model artifacts

Checkpoint files such as `ml/models/lesion_classifier_binary.pt` are local artifacts and are intentionally not committed. The public repo should show the training code, inference code, dataset split, and recorded metrics without distributing model weights by default.

For a demo on another machine, either copy the checkpoint out-of-band or retrain it with:

```bash
python -m ml.training.train
```
