import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from ml.evaluation.schema import (
    HAM10000_LABELS,
    PAD_UFES_NATIVE_LABELS,
    EvaluationExample,
    ModelMetadata,
    ModelPrediction,
)
from ml.evaluation.stress import (
    PHONE_STRESS_VARIANTS,
    build_phone_stress_examples,
    evaluate_phone_stress,
)


class RecordingAdapter:
    metadata = ModelMetadata(
        name="fake",
        source="test",
        adapter="fake",
        revision=None,
        license=None,
        labels=list(HAM10000_LABELS),
        notes=[],
    )

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def predict_image(self, image_path: Path) -> ModelPrediction:
        self.paths.append(Path(image_path))
        label = "melanoma" if "blur" in image_path.name else "nevus"
        return ModelPrediction(label=label, confidence=0.8, latency_ms=2.0)


class NativeRecordingAdapter:
    metadata = ModelMetadata(
        name="native-fake",
        source="test",
        adapter="fake",
        revision=None,
        license=None,
        labels=list(PAD_UFES_NATIVE_LABELS),
        notes=[],
    )

    def predict_image(self, image_path: Path) -> ModelPrediction:
        return ModelPrediction(
            label="squamous_cell_carcinoma",
            confidence=0.8,
            latency_ms=2.0,
            labels=PAD_UFES_NATIVE_LABELS,
        )


class EvaluationStressTests(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (96, 80), color=(120, 40, 200)).save(path)

    def test_build_phone_stress_examples_creates_expected_variants(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "source.jpg"
            self._write_image(image_path)
            original_bytes = image_path.read_bytes()
            output_dir = root / "stress"
            examples = [
                EvaluationExample(image_path=image_path, label="melanoma", split="test"),
            ]

            stress_examples = build_phone_stress_examples(examples, output_dir)

            variant_keys = {example.variant_key for example in stress_examples}
            stress_paths = [example.example.image_path for example in stress_examples]

            self.assertEqual(variant_keys, set(PHONE_STRESS_VARIANTS))
            self.assertEqual(len(stress_paths), len(PHONE_STRESS_VARIANTS))
            self.assertEqual(image_path.read_bytes(), original_bytes)
            self.assertTrue(all(path.exists() for path in stress_paths))
            self.assertTrue(all(output_dir in path.parents for path in stress_paths))

    def test_evaluate_phone_stress_summarizes_each_variant_and_aggregate(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "source.jpg"
            self._write_image(image_path)
            examples = [
                EvaluationExample(image_path=image_path, label="melanoma", split="test"),
            ]
            adapter = RecordingAdapter()

            result = evaluate_phone_stress(adapter, examples, root / "stress")

        self.assertEqual(set(result["variants"]), set(PHONE_STRESS_VARIANTS))
        self.assertEqual(
            result["variants"]["blur"]["per_class"]["melanoma"]["recall"],
            1.0,
        )
        self.assertEqual(
            result["variants"]["rotation"]["per_class"]["melanoma"]["recall"],
            0.0,
        )
        self.assertEqual(result["aggregate"]["sample_count"], len(PHONE_STRESS_VARIANTS))
        self.assertEqual(result["aggregate"]["latency_p95_ms"], 2.0)
        self.assertEqual(len(adapter.paths), len(PHONE_STRESS_VARIANTS))

    def test_evaluate_phone_stress_preserves_pad_ufes_native_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "source.jpg"
            self._write_image(image_path)
            examples = [
                EvaluationExample(
                    image_path=image_path,
                    label="squamous_cell_carcinoma",
                    split="test",
                    labels=PAD_UFES_NATIVE_LABELS,
                ),
            ]

            result = evaluate_phone_stress(NativeRecordingAdapter(), examples, root / "stress")

        self.assertEqual(result["aggregate"]["labels"], list(PAD_UFES_NATIVE_LABELS))
        self.assertEqual(
            result["aggregate"]["per_class"]["squamous_cell_carcinoma"]["recall"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
