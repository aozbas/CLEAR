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

            write_report(
                output_dir,
                model_metadata=metadata,
                examples=examples,
                predictions=predictions,
                metrics=metrics,
            )

            metrics_json = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            metadata_json = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            with (output_dir / "predictions.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            summary = (output_dir / "summary.md").read_text(encoding="utf-8")
            output_names = sorted(path.name for path in output_dir.iterdir())

        self.assertEqual(metrics_json["accuracy"], 1.0)
        self.assertEqual(metadata_json["model"]["name"], "baseline")
        self.assertEqual(
            rows[0].keys(),
            {"image_path", "truth", "prediction", "confidence", "latency_ms"},
        )
        self.assertIn("experimental classification", summary)
        self.assertEqual(
            output_names,
            ["metadata.json", "metrics.json", "predictions.csv", "summary.md"],
        )


if __name__ == "__main__":
    unittest.main()
