import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.summarize_pad_ufes_cv import summarize_reports


def fake_report(fold_index: int, *, protocol: str = "patient_grouped_rotating_cv") -> dict:
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
        "architecture": "resnet18",
        "input_mode": "image_only",
        "pretrained_weights": "imagenet",
        "selection_metric": "val_macro_f1",
        "seed": 42,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "best_epoch": 2,
        "best_val_macro_f1": 0.8,
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


def write_reports(root: Path) -> None:
    for fold_index in range(5):
        fold_dir = root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "report.json").write_text(
            json.dumps(fake_report(fold_index)),
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


if __name__ == "__main__":
    unittest.main()
