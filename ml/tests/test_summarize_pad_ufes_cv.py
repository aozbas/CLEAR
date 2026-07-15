import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.summarize_pad_ufes_cv import summarize_reports


def fake_report(
    fold_index: int,
    *,
    architecture: str = "resnet18",
    protocol: str = "patient_grouped_rotating_cv",
    augmentation_profile: str = "baseline",
    label_smoothing: float = 0.0,
    lr_schedule: str = "none",
    weight_decay: float = 1e-4,
    imbalance_strategy: str = "inverse_frequency_loss",
) -> dict:
    size = len(PAD_UFES_NATIVE_LABELS)
    confusion = [[0 for _ in range(size)] for _ in range(size)]
    per_class = {}
    for index, label in enumerate(PAD_UFES_NATIVE_LABELS):
        confusion[index][index] = 1
        per_class[label] = {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "support": 1,
        }
    return {
        "architecture": architecture,
        "input_mode": "image_only",
        "pretrained_weights": "imagenet",
        "selection_metric": "val_macro_f1",
        "seed": 42,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "best_epoch": 2,
        "best_val_macro_f1": 0.8,
        "hyperparameters": {
            "epochs": 15,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "augmentation_profile": augmentation_profile,
            "label_smoothing": label_smoothing,
            "lr_schedule": lr_schedule,
            "weight_decay": weight_decay,
            "imbalance_strategy": imbalance_strategy,
        },
        "split_summary": {
            "protocol": protocol,
            "num_folds": 5,
            "fold_index": fold_index,
            "test_outer_fold": fold_index,
            "validation_outer_fold": (fold_index + 1) % 5,
            "cv_total_image_count": 30,
        },
        "test": {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "per_class": per_class,
            "confusion_matrix": confusion,
        },
    }


def write_reports(root: Path, **configuration: object) -> None:
    for fold_index in range(5):
        fold_dir = root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "report.json").write_text(
            json.dumps(fake_report(fold_index, **configuration)),
            encoding="utf-8",
        )


class SummarizePadUfesCrossValidationTests(unittest.TestCase):
    def test_aggregates_fold_distribution_and_pooled_confusion(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_reports(root)
            out_path = root / "summary.json"

            summary = summarize_reports(root, out_path, num_folds=5, seed=42)

        self.assertEqual(summary["fold_metrics"]["macro_f1"]["mean"], 1.0)
        self.assertEqual(summary["fold_metrics"]["macro_f1"]["population_std"], 0.0)
        self.assertEqual(summary["pooled_test"]["total_support"], 30)
        self.assertEqual(summary["pooled_test"]["macro_f1"], 1.0)

    def test_rejects_missing_fold_report(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_reports(root)
            (root / "fold_4" / "report.json").unlink()

            with self.assertRaises(FileNotFoundError):
                summarize_reports(root, root / "summary.json", num_folds=5, seed=42)

    def test_rejects_protocol_mismatch(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_reports(root)
            report_path = root / "fold_2" / "report.json"
            report_path.write_text(
                json.dumps(fake_report(2, protocol="different_protocol")),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "violates the locked protocol"):
                summarize_reports(root, root / "summary.json", num_folds=5, seed=42)

    def test_accepts_completed_legacy_baseline_reports_without_new_fields(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_reports(root)
            for fold_index in range(5):
                report_path = root / f"fold_{fold_index}" / "report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report.pop("hyperparameters")
                report_path.write_text(json.dumps(report), encoding="utf-8")

            summary = summarize_reports(root, root / "summary.json", num_folds=5, seed=42)

        self.assertEqual(summary["augmentation_profile"], "baseline")
        self.assertEqual(summary["imbalance_strategy"], "inverse_frequency_loss")

    def test_verifies_imbalance_strategy(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_reports(root, imbalance_strategy="balanced_sampler")

            summary = summarize_reports(
                root,
                root / "summary.json",
                num_folds=5,
                seed=42,
                imbalance_strategy="balanced_sampler",
            )

        self.assertEqual(summary["imbalance_strategy"], "balanced_sampler")

    def test_verifies_architecture(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_reports(root, architecture="efficientnet_b0")

            summary = summarize_reports(
                root,
                root / "summary.json",
                num_folds=5,
                seed=42,
                architecture="efficientnet_b0",
            )

        self.assertEqual(summary["architecture"], "efficientnet_b0")

    def test_verifies_regularized_training_configuration(self) -> None:
        configuration = {
            "augmentation_profile": "regularized_v2",
            "label_smoothing": 0.1,
            "lr_schedule": "cosine",
            "weight_decay": 1e-3,
        }
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_reports(root, **configuration)

            summary = summarize_reports(
                root,
                root / "summary.json",
                num_folds=5,
                seed=42,
                **configuration,
            )

        self.assertEqual(summary["augmentation_profile"], "regularized_v2")
        self.assertEqual(summary["label_smoothing"], 0.1)

    def test_rejects_regularized_configuration_mismatch(self) -> None:
        configuration = {
            "augmentation_profile": "regularized_v2",
            "label_smoothing": 0.1,
            "lr_schedule": "cosine",
            "weight_decay": 1e-3,
        }
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_reports(root, **configuration)
            report_path = root / "fold_2" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["hyperparameters"]["label_smoothing"] = 0.0
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "violates the locked protocol"):
                summarize_reports(
                    root,
                    root / "summary.json",
                    num_folds=5,
                    seed=42,
                    **configuration,
                )

    def test_rejects_mixed_core_training_hyperparameters(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_reports(root)
            report_path = root / "fold_2" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["hyperparameters"]["learning_rate"] = 2e-4
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "violates the locked protocol"):
                summarize_reports(root, root / "summary.json", num_folds=5, seed=42)


if __name__ == "__main__":
    unittest.main()
