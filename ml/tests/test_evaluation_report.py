import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.evaluation.report import write_report
from ml.evaluation.schema import HAM10000_LABELS, EvaluationExample, ModelMetadata, ModelPrediction


class EvaluationReportTests(unittest.TestCase):
    def test_write_report_creates_parseable_outputs_without_copying_images(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image-bytes")
            output_dir = root / "report"
            metadata = ModelMetadata(
                name="baseline",
                source="local",
                adapter="baseline",
                revision=None,
                license=None,
                labels=list(HAM10000_LABELS),
                notes=["test"],
            )
            examples = [
                EvaluationExample(image_path=image_path, label="nevus", split="test"),
            ]
            predictions = [
                ModelPrediction(label="nevus", confidence=0.91, latency_ms=12.5),
            ]
            metrics = {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "per_class": {},
            }
            dataset_metadata = {
                "key": "ph2_holdout",
                "name": "PH2 external holdout",
                "split_type": "external_holdout",
                "partial_label_set": True,
                "labels": ["melanoma", "nevus"],
                "contamination_notes": [],
            }

            write_report(
                output_dir,
                model_metadata=metadata,
                examples=examples,
                predictions=predictions,
                metrics=metrics,
                dataset_metadata=dataset_metadata,
            )

            metrics_json = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            metadata_json = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            with (output_dir / "predictions.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            summary = (output_dir / "summary.md").read_text(encoding="utf-8")
            output_names = sorted(path.name for path in output_dir.iterdir())

        self.assertEqual(metrics_json["accuracy"], 1.0)
        self.assertEqual(metadata_json["model"]["name"], "baseline")
        self.assertEqual(metadata_json["dataset"]["key"], "ph2_holdout")
        self.assertEqual(
            rows[0].keys(),
            {"image_path", "truth", "prediction", "confidence", "latency_ms"},
        )
        self.assertIn("experimental classification", summary)
        self.assertIn("PH2 external holdout", summary)
        self.assertEqual(
            output_names,
            ["metadata.json", "metrics.json", "predictions.csv", "summary.md"],
        )

    def test_summary_includes_covered_metrics_and_prediction_distribution(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image-bytes")
            output_dir = root / "report"
            metadata = ModelMetadata(
                name="baseline",
                source="local",
                adapter="baseline",
                revision=None,
                license=None,
                labels=list(HAM10000_LABELS),
                notes=[],
            )
            examples = [
                EvaluationExample(image_path=image_path, label="melanoma", split="test"),
                EvaluationExample(image_path=image_path, label="nevus", split="test"),
            ]
            predictions = [
                ModelPrediction(label="melanoma", confidence=0.91, latency_ms=12.5),
                ModelPrediction(label="basal_cell_carcinoma", confidence=0.61, latency_ms=12.7),
            ]
            metrics = {
                "accuracy": 0.5,
                "balanced_accuracy": 1 / 7,
                "macro_f1": 1 / 7,
                "covered_labels": ["melanoma", "nevus"],
                "covered_label_macro_precision": 0.5,
                "covered_label_macro_recall": 0.5,
                "covered_label_balanced_accuracy": 0.5,
                "covered_label_macro_f1": 0.5,
                "prediction_distribution": {
                    "melanoma": {"count": 1, "fraction": 0.5},
                    "nevus": {"count": 0, "fraction": 0.0},
                    "basal_cell_carcinoma": {"count": 1, "fraction": 0.5},
                    "actinic_keratosis": {"count": 0, "fraction": 0.0},
                    "benign_keratosis": {"count": 0, "fraction": 0.0},
                    "dermatofibroma": {"count": 0, "fraction": 0.0},
                    "vascular_lesion": {"count": 0, "fraction": 0.0},
                },
            }
            dataset_metadata = {
                "key": "ph2_holdout",
                "name": "PH2 external holdout",
                "split_type": "external_holdout",
                "partial_label_set": True,
                "labels": ["melanoma", "nevus"],
                "contamination_notes": [],
            }

            write_report(
                output_dir,
                model_metadata=metadata,
                examples=examples,
                predictions=predictions,
                metrics=metrics,
                dataset_metadata=dataset_metadata,
            )

            summary = (output_dir / "summary.md").read_text(encoding="utf-8")

        self.assertIn("Covered labels: melanoma, nevus", summary)
        self.assertIn("Covered-label metrics", summary)
        self.assertIn("| covered_label_macro_f1 | 0.5 |", summary)
        self.assertIn("Prediction distribution", summary)
        self.assertIn("| melanoma | 1 | 0.5 |", summary)
        self.assertIn("| basal_cell_carcinoma | 1 | 0.5 |", summary)

    def test_summary_includes_phone_stress_metrics(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image-bytes")
            output_dir = root / "report"
            metadata = ModelMetadata(
                name="baseline",
                source="local",
                adapter="baseline",
                revision=None,
                license=None,
                labels=list(HAM10000_LABELS),
                notes=[],
            )
            examples = [
                EvaluationExample(image_path=image_path, label="melanoma", split="test"),
            ]
            predictions = [
                ModelPrediction(label="melanoma", confidence=0.91, latency_ms=12.5),
            ]
            metrics = {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "phone_stress": {
                    "aggregate": {
                        "sample_count": 2,
                        "accuracy": 0.5,
                        "covered_label_macro_f1": 0.5,
                        "latency_p95_ms": 15.0,
                    },
                    "variants": {
                        "blur": {
                            "sample_count": 1,
                            "accuracy": 1.0,
                            "covered_label_macro_f1": 1.0,
                            "per_class": {"melanoma": {"recall": 1.0}},
                        },
                        "rotation": {
                            "sample_count": 1,
                            "accuracy": 0.0,
                            "covered_label_macro_f1": 0.0,
                            "per_class": {"melanoma": {"recall": 0.0}},
                        },
                    },
                },
            }

            write_report(
                output_dir,
                model_metadata=metadata,
                examples=examples,
                predictions=predictions,
                metrics=metrics,
            )

            summary = (output_dir / "summary.md").read_text(encoding="utf-8")

        self.assertIn("Phone-photo stress tests", summary)
        self.assertIn("| aggregate | 2 | 0.5 | 0.5 |  | 15 |", summary)
        self.assertIn("| blur | 1 | 1 | 1 | 1 |  |", summary)
        self.assertIn("| rotation | 1 | 0 | 0 | 0 |  |", summary)


if __name__ == "__main__":
    unittest.main()
