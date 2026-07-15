"""Run the locked PAD-UFES grouped cross-validation baseline sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ml.preprocessing import PAD_UFES_AUGMENTATION_PROFILES
from ml.training.prepare_pad_ufes import project_relative
from ml.training.prepare_pad_ufes_cv import DEFAULT_FOLDS
from ml.training.summarize_pad_ufes_cv import summarize_reports
from ml.training.train import resolve_project_path
from ml.training.train_pad_ufes import ARCHITECTURES, IMBALANCE_STRATEGIES, LR_SCHEDULES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "external_splits" / "pad_ufes_native_cv_224"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "ml" / "runs" / "training" / "pad_ufes_resnet18-cv-seed42"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "ml" / "models" / "pad_ufes_resnet18_cv_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked PAD-UFES-native grouped CV baseline."
    )
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="resnet18")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--augmentation-profile",
        choices=PAD_UFES_AUGMENTATION_PROFILES,
        default="baseline",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="none")
    parser.add_argument(
        "--imbalance-strategy",
        choices=IMBALANCE_STRATEGIES,
        default="inverse_frequency_loss",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_training_command(
    *,
    split_csv: Path,
    checkpoint: Path,
    run_dir: Path,
    architecture: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    augmentation_profile: str,
    label_smoothing: float,
    lr_schedule: str,
    imbalance_strategy: str,
    num_workers: int,
    seed: int,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "ml.training.train_pad_ufes",
        "--split-csv",
        str(split_csv),
        "--checkpoint",
        str(checkpoint),
        "--run-dir",
        str(run_dir),
        "--architecture",
        architecture,
        "--weights",
        "imagenet",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--augmentation-profile",
        augmentation_profile,
        "--label-smoothing",
        str(label_smoothing),
        "--lr-schedule",
        lr_schedule,
        "--imbalance-strategy",
        imbalance_strategy,
        "--num-workers",
        str(num_workers),
        "--seed",
        str(seed),
        "--device",
        device,
    ]


def run_cross_validation(args: argparse.Namespace) -> None:
    if args.folds < 3:
        raise ValueError("folds must be at least 3.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in the range [0, 1).")

    splits_dir = resolve_project_path(args.splits_dir)
    runs_root = resolve_project_path(args.runs_root)
    checkpoints_dir = resolve_project_path(args.checkpoints_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for fold_index in range(args.folds):
        split_csv = splits_dir / f"fold_{fold_index}.csv"
        split_summary = split_csv.with_suffix(".summary.json")
        if not split_csv.exists() or not split_summary.exists():
            raise FileNotFoundError(f"Missing materialized fold inputs for fold_{fold_index}.")

        run_dir = runs_root / f"fold_{fold_index}"
        report_path = run_dir / "report.json"
        if args.resume and report_path.exists():
            print(f"fold_{fold_index}: report exists; skipping")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoints_dir / f"fold_{fold_index}.pt"
        command = build_training_command(
            split_csv=split_csv,
            checkpoint=checkpoint,
            run_dir=run_dir,
            architecture=args.architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            augmentation_profile=args.augmentation_profile,
            label_smoothing=args.label_smoothing,
            lr_schedule=args.lr_schedule,
            imbalance_strategy=args.imbalance_strategy,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
        )
        print(f"fold_{fold_index}: starting")
        log_path = run_dir / "train.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(f"fold_{fold_index}: {line}", end="")
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        print(f"fold_{fold_index}: completed")

    summary = summarize_reports(
        runs_root,
        runs_root / "summary.json",
        num_folds=args.folds,
        seed=args.seed,
        architecture=args.architecture,
        augmentation_profile=args.augmentation_profile,
        label_smoothing=args.label_smoothing,
        lr_schedule=args.lr_schedule,
        imbalance_strategy=args.imbalance_strategy,
        weight_decay=args.weight_decay,
    )
    macro_f1 = summary["fold_metrics"]["macro_f1"]
    print(
        f"Cross-validation complete: macro_f1_mean={macro_f1['mean']:.4f} "
        f"population_std={macro_f1['population_std']:.4f} "
        f"summary={project_relative(runs_root / 'summary.json')}"
    )


def main() -> None:
    run_cross_validation(parse_args())


if __name__ == "__main__":
    main()
