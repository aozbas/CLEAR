import contextlib
import io
import json
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

    def test_huggingface_candidate_uses_registry_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image")
            examples = [
                EvaluationExample(image_path=image_path, label="melanoma", split="test"),
            ]

            with (
                patch("ml.evaluation.cli.load_examples", return_value=examples),
                patch(
                    "ml.evaluation.cli.HuggingFaceImageClassifierAdapter",
                    return_value=FakeAdapter(),
                ) as hf_adapter,
                patch(
                    "ml.evaluation.cli.summarize_metrics",
                    return_value={"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0},
                ),
                patch("ml.evaluation.cli.write_report"),
            ):
                exit_code = main(
                    [
                        "--split-csv",
                        str(root / "split.csv"),
                        "--split",
                        "test",
                        "--model",
                        "gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer",
                        "--out",
                        str(root / "out"),
                    ]
                )

        self.assertEqual(exit_code, 0)
        _, kwargs = hf_adapter.call_args
        self.assertEqual(
            kwargs["model_id"],
            "gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer",
        )
        self.assertEqual(kwargs["revision"], "3b408dc64c66e7a39c86b87d2283146821a8be28")
        self.assertEqual(kwargs["label_map"]["Melanoma"], "melanoma")
        self.assertEqual(kwargs["cache_dir"], Path("ml/model_cache/huggingface"))

    def test_inspect_model_writes_local_metadata_report(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "inspect"
            with patch("ml.evaluation.cli.HfApi", return_value=FakeHfApi()):
                exit_code = main(
                    [
                        "--inspect-model",
                        "example/model",
                        "--out",
                        str(output_dir),
                    ]
                )

            inspection = json.loads((output_dir / "inspection.json").read_text(encoding="utf-8"))
            summary = (output_dir / "summary.md").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(inspection["model_id"], "example/model")
        self.assertEqual(inspection["revision"], "abc123")
        self.assertEqual(inspection["license"], "mit")
        self.assertEqual(inspection["config_labels"], {"0": "melanoma"})
        self.assertIn("experimental classification", summary)


class FakeCardData:
    license = "mit"
    license_name = None
    license_link = None
    datasets = ["HAM10000"]


class FakeSibling:
    def __init__(self, filename: str, size: int) -> None:
        self.rfilename = filename
        self.size = size


class FakeModelInfo:
    id = "example/model"
    sha = "abc123"
    pipeline_tag = "image-classification"
    library_name = "transformers"
    tags = ["license:mit"]
    card_data = FakeCardData()
    config = {"id2label": {"0": "melanoma"}}
    siblings = [FakeSibling("config.json", 100)]


class FakeHfApi:
    def model_info(self, repo_id: str, *, files_metadata: bool) -> FakeModelInfo:
        self.repo_id = repo_id
        self.files_metadata = files_metadata
        return FakeModelInfo()


if __name__ == "__main__":
    unittest.main()
