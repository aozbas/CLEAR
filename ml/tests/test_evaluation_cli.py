import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ml.evaluation.cli import main
from ml.evaluation.schema import EvaluationExample, HAM10000_LABELS, ModelMetadata, ModelPrediction


class FakeAdapter:
    metadata = ModelMetadata(
        name="baseline",
        source="test",
        adapter="baseline",
        revision=None,
        license=None,
        labels=list(HAM10000_LABELS),
        notes=[],
    )

    def predict_image(self, image_path: Path) -> ModelPrediction:
        return ModelPrediction(label="nevus", confidence=0.8, latency_ms=1.0)


class EvaluationCliTests(unittest.TestCase):
    def test_baseline_run_uses_loader_adapter_metrics_and_report(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_csv = root / "split.csv"
            output_dir = root / "out"
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image")
            examples = [
                EvaluationExample(image_path=image_path, label="nevus", split="test"),
            ]

            with (
                patch("ml.evaluation.cli.load_examples", return_value=examples) as load_examples,
                patch("ml.evaluation.cli.BaselineAdapter", return_value=FakeAdapter()) as baseline,
                patch(
                    "ml.evaluation.cli.summarize_metrics",
                    return_value={"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0},
                ) as summarize,
                patch("ml.evaluation.cli.write_report") as write_report,
            ):
                exit_code = main(
                    [
                        "--split-csv",
                        str(split_csv),
                        "--split",
                        "test",
                        "--model",
                        "baseline",
                        "--out",
                        str(output_dir),
                        "--max-samples",
                        "2",
                    ]
                )

        self.assertEqual(exit_code, 0)
        load_examples.assert_called_once_with(
            split_csv,
            "test",
            max_samples=2,
            samples_per_label=None,
        )
        baseline.assert_called_once_with(model_path=None)
        summarize.assert_called_once_with(["nevus"], ["nevus"])
        write_report.assert_called_once()

    def test_unknown_model_fails_clearly(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "--split-csv",
                        str(Path(tmp_dir) / "split.csv"),
                        "--split",
                        "test",
                        "--model",
                        "unknown",
                        "--out",
                        str(Path(tmp_dir) / "out"),
                    ]
                )

        self.assertNotEqual(exit_code, 0)
        self.assertIn("Unsupported model", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
