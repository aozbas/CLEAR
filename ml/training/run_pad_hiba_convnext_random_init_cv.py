"""Run the preregistered rights-clean ConvNeXt-Tiny initialization feasibility test.

This controlled PAD-UFES plus HIBA development experiment changes only the initialization from the
completed source-balanced reference: the model starts from random weights and never downloads or
loads pretrained model weights. It does not evaluate MILK10k, change the demo model, or establish
medical or deployment readiness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ml.training.prepare_pad_ufes import project_relative
from ml.training.prepare_pad_ufes_cv import DEFAULT_FOLDS
from ml.training.run_pad_hiba_convnext_cv import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIBA_SPLITS_DIR,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PAD_SPLITS_DIR,
    DEFAULT_SEED,
    DEFAULT_TORCH_CACHE_DIR,
    DEFAULT_WEIGHT_DECAY,
    load_multi_source_manifests,
    summarize_reports,
    train_fold,
    validate_report,
)
from ml.training.train import get_device, resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = (
    PROJECT_ROOT / "ml" / "runs" / "training" / "pad_hiba_convnext_tiny-random-init-cv-seed42"
)
DEFAULT_CHECKPOINTS_DIR = (
    PROJECT_ROOT / "ml" / "models" / "pad_hiba_convnext_tiny_random_init_cv_seed42"
)
RIGHTS_CLEAN_EPOCHS = DEFAULT_EPOCHS
RIGHTS_CLEAN_BATCH_SIZE = DEFAULT_BATCH_SIZE
RIGHTS_CLEAN_LEARNING_RATE = DEFAULT_LEARNING_RATE
RIGHTS_CLEAN_WEIGHT_DECAY = DEFAULT_WEIGHT_DECAY
RIGHTS_CLEAN_SEED = DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked PAD/HIBA random-initialized ConvNeXt-Tiny feasibility CV."
    )
    parser.add_argument("--pad-splits-dir", type=Path, default=DEFAULT_PAD_SPLITS_DIR)
    parser.add_argument("--hiba-splits-dir", type=Path, default=DEFAULT_HIBA_SPLITS_DIR)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--torch-cache-dir", type=Path, default=DEFAULT_TORCH_CACHE_DIR)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    device = get_device(args.device)
    print(f"Using device: {device}")
    manifests = load_multi_source_manifests(args.pad_splits_dir, args.hiba_splits_dir)
    print(f"Validated {DEFAULT_FOLDS} PAD/HIBA rotating folds; fingerprint={manifests.fingerprint}")

    torch_cache_dir = resolve_project_path(args.torch_cache_dir)
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_cache_dir))
    runs_root = resolve_project_path(args.runs_root)
    checkpoints_dir = resolve_project_path(args.checkpoints_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for fold_index in range(DEFAULT_FOLDS):
        run_dir = runs_root / f"fold_{fold_index}"
        report_path = run_dir / "report.json"
        if args.resume and report_path.is_file():
            print(f"fold_{fold_index}: report exists; skipping")
            continue
        print(f"fold_{fold_index}: starting random-initialized ConvNeXt-Tiny")
        report = train_fold(
            manifests.folds[fold_index],
            fold_index=fold_index,
            pad_summary=manifests.pad_fold_summaries[fold_index],
            hiba_summary=manifests.hiba_fold_summaries[fold_index],
            manifest_fingerprint=manifests.fingerprint,
            source_total_raw_image_counts=manifests.source_total_raw_image_counts,
            source_total_effective_unit_counts=manifests.source_total_effective_unit_counts,
            checkpoint_path=checkpoints_dir / f"fold_{fold_index}.pt",
            run_dir=run_dir,
            device=device,
            num_workers=args.num_workers,
            epochs=RIGHTS_CLEAN_EPOCHS,
            batch_size=RIGHTS_CLEAN_BATCH_SIZE,
            learning_rate=RIGHTS_CLEAN_LEARNING_RATE,
            weight_decay=RIGHTS_CLEAN_WEIGHT_DECAY,
            seed=RIGHTS_CLEAN_SEED,
            pretrained=False,
        )
        validate_report(
            report,
            fold_index=fold_index,
            epochs=RIGHTS_CLEAN_EPOCHS,
            batch_size=RIGHTS_CLEAN_BATCH_SIZE,
            learning_rate=RIGHTS_CLEAN_LEARNING_RATE,
            weight_decay=RIGHTS_CLEAN_WEIGHT_DECAY,
            seed=RIGHTS_CLEAN_SEED,
            pretrained=False,
        )
        print(
            f"fold_{fold_index}: best_epoch={report['best_epoch']} "
            "outer-test report held for the pooled five-fold summary"
        )

    summary = summarize_reports(
        runs_root,
        runs_root / "summary.json",
        epochs=RIGHTS_CLEAN_EPOCHS,
        batch_size=RIGHTS_CLEAN_BATCH_SIZE,
        learning_rate=RIGHTS_CLEAN_LEARNING_RATE,
        weight_decay=RIGHTS_CLEAN_WEIGHT_DECAY,
        seed=RIGHTS_CLEAN_SEED,
        pretrained=False,
    )
    print(
        "Random-initialized PAD/HIBA feasibility test complete: "
        f"source_mean={summary['pooled_primary_source_mean_macro_f1']:.4f} "
        f"worst_source={summary['pooled_primary_worst_source_macro_f1']:.4f} "
        f"all_rules_pass={summary['decision_rules']['all_pass']} "
        f"summary={project_relative(runs_root / 'summary.json')}"
    )
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
