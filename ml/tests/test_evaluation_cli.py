import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ml.evaluation.cli import main
from ml.evaluation.schema import HAM10000_LABELS, EvaluationExample, ModelMetadata, ModelPrediction


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
        _, report_kwargs = write_report.call_args
        self.assertEqual(report_kwargs["dataset_metadata"]["key"], "ham10000_internal")
        self.assertEqual(report_kwargs["dataset_metadata"]["contamination_notes"], [])

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

    def test_keras_candidate_uses_registry_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image")
            examples = [
                EvaluationExample(image_path=image_path, label="melanoma", split="test"),
            ]

            with (
                patch("ml.evaluation.cli.load_examples", return_value=examples),
                patch("ml.evaluation.cli.KerasH5Adapter", return_value=FakeAdapter()) as adapter,
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
                        "Miguel764/efficientnetv2s-skin-cancer-classifier",
                        "--out",
                        str(root / "out"),
                    ]
                )

        self.assertEqual(exit_code, 0)
        _, kwargs = adapter.call_args
        self.assertEqual(kwargs["model_id"], "Miguel764/efficientnetv2s-skin-cancer-classifier")
        self.assertEqual(kwargs["revision"], "97a25e6b71c4b426c259b747a6c49d235c2dade7")
        self.assertEqual(kwargs["artifact_filename"], "efficientnetv2s.h5")
        self.assertEqual(kwargs["label_map"]["mel"], "melanoma")
        self.assertEqual(kwargs["cache_dir"], Path("ml/model_cache/huggingface"))

    def test_open_clip_candidate_uses_registry_metadata(self) -> None:
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
                    "ml.evaluation.cli.OpenClipZeroShotAdapter",
                    return_value=FakeAdapter(),
                ) as adapter,
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
                        "redlessone/DermLIP_ViT-B-16",
                        "--out",
                        str(root / "out"),
                    ]
                )

        self.assertEqual(exit_code, 0)
        _, kwargs = adapter.call_args
        self.assertEqual(kwargs["model_id"], "redlessone/DermLIP_ViT-B-16")
        self.assertEqual(kwargs["revision"], "main")
        self.assertEqual(kwargs["cache_dir"], Path("ml/model_cache/huggingface"))

    def test_transformers_zero_shot_candidate_uses_registry_metadata(self) -> None:
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
                    "ml.evaluation.cli.TransformersZeroShotAdapter",
                    return_value=FakeAdapter(),
                ) as adapter,
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
                        "google/medsiglip-448",
                        "--out",
                        str(root / "out"),
                    ]
                )

        self.assertEqual(exit_code, 0)
        _, kwargs = adapter.call_args
        self.assertEqual(kwargs["model_id"], "google/medsiglip-448")
        self.assertEqual(kwargs["revision"], "9cea28a1a1195f665105faa6e8544c112fd960a4")
        self.assertEqual(kwargs["cache_dir"], Path("ml/model_cache/huggingface"))

    def test_embedding_probe_candidate_requires_probe_workflow(self) -> None:
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
                        "google/derm-foundation",
                        "--out",
                        str(Path(tmp_dir) / "out"),
                    ]
                )

        self.assertNotEqual(exit_code, 0)
        self.assertIn("embedding-probe workflow", stderr.getvalue())

    def test_dataset_source_argument_is_written_to_report_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image")
            examples = [
                EvaluationExample(image_path=image_path, label="nevus", split="test"),
            ]

            with (
                patch("ml.evaluation.cli.load_examples", return_value=examples),
                patch("ml.evaluation.cli.BaselineAdapter", return_value=FakeAdapter()),
                patch(
                    "ml.evaluation.cli.summarize_metrics",
                    return_value={"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0},
                ),
                patch("ml.evaluation.cli.write_report") as write_report,
            ):
                exit_code = main(
                    [
                        "--split-csv",
                        str(root / "split.csv"),
                        "--split",
                        "test",
                        "--dataset-source",
                        "ph2_holdout",
                        "--model",
                        "baseline",
                        "--out",
                        str(root / "out"),
                    ]
                )

        self.assertEqual(exit_code, 0)
        _, kwargs = write_report.call_args
        self.assertEqual(kwargs["dataset_metadata"]["key"], "ph2_holdout")
        self.assertEqual(kwargs["dataset_metadata"]["split_type"], "external_holdout")
        self.assertTrue(kwargs["dataset_metadata"]["partial_label_set"])

    def test_phone_stress_flag_adds_stress_metrics_to_report(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image")
            output_dir = root / "out"
            examples = [
                EvaluationExample(image_path=image_path, label="melanoma", split="test"),
            ]
            stress_metrics = {
                "aggregate": {"sample_count": 1, "accuracy": 1.0},
                "variants": {"blur": {"sample_count": 1, "accuracy": 1.0}},
            }

            with (
                patch("ml.evaluation.cli.load_examples", return_value=examples),
                patch("ml.evaluation.cli.BaselineAdapter", return_value=FakeAdapter()),
                patch(
                    "ml.evaluation.cli.summarize_metrics",
                    return_value={"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0},
                ),
                patch(
                    "ml.evaluation.cli.evaluate_phone_stress",
                    return_value=stress_metrics,
                ) as evaluate_phone_stress,
                patch("ml.evaluation.cli.write_report") as write_report,
            ):
                exit_code = main(
                    [
                        "--split-csv",
                        str(root / "split.csv"),
                        "--split",
                        "test",
                        "--model",
                        "baseline",
                        "--out",
                        str(output_dir),
                        "--phone-stress",
                    ]
                )

        self.assertEqual(exit_code, 0)
        evaluate_phone_stress.assert_called_once()
        _, stress_kwargs = evaluate_phone_stress.call_args
        self.assertEqual(stress_kwargs["output_dir"], output_dir / "phone_stress_images")
        _, report_kwargs = write_report.call_args
        self.assertEqual(report_kwargs["metrics"]["phone_stress"], stress_metrics)

    def test_ham10000_candidate_on_ham10000_source_records_contamination_note(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image")
            examples = [
                EvaluationExample(image_path=image_path, label="melanoma", split="test"),
            ]

            with (
                patch("ml.evaluation.cli.load_examples", return_value=examples),
                patch("ml.evaluation.cli.KerasH5Adapter", return_value=FakeAdapter()),
                patch(
                    "ml.evaluation.cli.summarize_metrics",
                    return_value={"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0},
                ),
                patch("ml.evaluation.cli.write_report") as write_report,
            ):
                exit_code = main(
                    [
                        "--split-csv",
                        str(root / "split.csv"),
                        "--split",
                        "test",
                        "--model",
                        "Miguel764/efficientnetv2s-skin-cancer-classifier",
                        "--out",
                        str(root / "out"),
                    ]
                )

        self.assertEqual(exit_code, 0)
        _, kwargs = write_report.call_args
        notes = kwargs["dataset_metadata"]["contamination_notes"]
        self.assertTrue(any("possible train/test overlap" in note for note in notes))

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

    def test_inspect_model_reads_config_file_when_api_omits_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "inspect"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({"id2label": {"0": "nevus"}}),
                encoding="utf-8",
            )

            with (
                patch("ml.evaluation.cli.HfApi", return_value=FakeHfApiNoLabels()),
                patch("ml.evaluation.cli.hf_hub_download", return_value=str(config_path)),
            ):
                exit_code = main(
                    [
                        "--inspect-model",
                        "example/model",
                        "--out",
                        str(output_dir),
                    ]
                )

            inspection = json.loads((output_dir / "inspection.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(inspection["config_labels"], {"0": "nevus"})


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


class FakeModelInfoNoLabels(FakeModelInfo):
    config = {}


class FakeHfApi:
    def model_info(self, repo_id: str, *, files_metadata: bool) -> FakeModelInfo:
        self.repo_id = repo_id
        self.files_metadata = files_metadata
        return FakeModelInfo()


class FakeHfApiNoLabels:
    def model_info(self, repo_id: str, *, files_metadata: bool) -> FakeModelInfoNoLabels:
        self.repo_id = repo_id
        self.files_metadata = files_metadata
        return FakeModelInfoNoLabels()


if __name__ == "__main__":
    unittest.main()
