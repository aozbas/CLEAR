import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_pad_ufes_cv import PROTOCOL
from ml.training.run_pad_ufes_medsiglip_probe_cv import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    EmbeddingDataset,
    build_cache_payload,
    extract_embeddings,
    load_cv_manifests,
    load_embedding_cache,
    summarize_probe_reports,
    train_probe_fold,
)


def write_cv_manifests(root: Path) -> list[Path]:
    paths = [f"image-{index}.png" for index in range(30)]
    labels = [PAD_UFES_NATIVE_LABELS[index // 5] for index in range(30)]
    fold_paths = []
    for fold_index in range(5):
        validation_fold = (fold_index + 1) % 5
        rows = []
        for index, (path, label) in enumerate(zip(paths, labels, strict=True)):
            outer_fold = index % 5
            if outer_fold == fold_index:
                split = "test"
            elif outer_fold == validation_fold:
                split = "val"
            else:
                split = "train"
            rows.append({"split": split, "image_path": path, "label": label})

        fold_path = root / f"fold_{fold_index}.csv"
        pd.DataFrame(rows).to_csv(fold_path, index=False)
        summary = {
            "dataset": "pad_ufes",
            "label_mode": "native",
            "split_strategy": "patient",
            "protocol": PROTOCOL,
            "num_folds": 5,
            "fold_index": fold_index,
            "test_outer_fold": fold_index,
            "validation_outer_fold": validation_fold,
            "group_key": "patient_id",
            "patient_overlap_count": 0,
            "patient_lesion_overlap_count": 0,
            "image_count": 30,
            "cv_total_image_count": 30,
            "images_by_split": {"train": 18, "val": 6, "test": 6},
        }
        fold_path.with_suffix(".summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        fold_paths.append(fold_path)
    return fold_paths


def perfect_report(fold_index: int) -> dict[str, object]:
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
    metrics = {
        "loss": 0.1,
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "macro_f1": 1.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
    return {
        "architecture": "medsiglip_frozen_linear_probe",
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "encoder_frozen": True,
        "known_pretraining_overlap": True,
        "embedding_normalization": "l2",
        "preprocessing": "medsiglip_auto_image_processor_slow_native_448",
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "seed": 42,
        "selection_metric": "val_macro_f1",
        "split_summary": {
            "protocol": PROTOCOL,
            "num_folds": 5,
            "fold_index": fold_index,
            "test_outer_fold": fold_index,
            "validation_outer_fold": (fold_index + 1) % 5,
            "cv_total_image_count": 30,
        },
        "hyperparameters": {
            "epochs": 100,
            "batch_size": 128,
            "learning_rate": 1e-2,
            "weight_decay": 1e-2,
            "imbalance_strategy": "inverse_frequency_loss",
        },
        "best_epoch": 1,
        "best_val_macro_f1": 1.0,
        "selected_train": metrics,
        "selected_val": metrics,
        "test": metrics,
    }


class CvManifestTests(unittest.TestCase):
    def test_accepts_consistent_rotating_patient_grouped_manifests(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            write_cv_manifests(Path(tmp_dir))
            manifests = load_cv_manifests(Path(tmp_dir), num_folds=5)

        self.assertEqual(len(manifests.fold_rows), 5)
        self.assertEqual(len(manifests.unique_rows), 30)
        self.assertEqual(len(manifests.fingerprint), 64)

    def test_rejects_label_drift_between_folds(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fold_paths = write_cv_manifests(root)
            rows = pd.read_csv(fold_paths[2])
            rows.loc[rows["image_path"] == "image-0.png", "label"] = "melanoma"
            rows.to_csv(fold_paths[2], index=False)

            with self.assertRaisesRegex(ValueError, "label mapping differs"):
                load_cv_manifests(root, num_folds=5)

    def test_rejects_image_not_used_as_test_exactly_once(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fold_paths = write_cv_manifests(root)
            rows = pd.read_csv(fold_paths[0])
            rows.loc[rows["image_path"] == "image-0.png", "split"] = "train"
            rows.loc[rows["image_path"] == "image-2.png", "split"] = "test"
            rows.to_csv(fold_paths[0], index=False)
            summary_path = fold_paths[0].with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["images_by_split"] = {"train": 18, "val": 6, "test": 6}
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "test data exactly once"):
                load_cv_manifests(root, num_folds=5)


class FakeProcessor:
    image_processor = SimpleNamespace(size={"height": 448, "width": 448})

    def __call__(
        self, *, images: list[Image.Image], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        self.return_tensors = return_tensors
        values = [float(image.getpixel((0, 0))[0]) for image in images]
        return {"pixel_values": torch.tensor(values).reshape(-1, 1)}


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def get_image_features(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        values = pixel_values * self.scale
        return SimpleNamespace(pooler_output=torch.cat((values, values + 1.0), dim=1))


class EmbeddingCacheTests(unittest.TestCase):
    def test_extracts_normalized_embeddings_with_frozen_encoder(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_paths = []
            for index in range(2):
                image_path = root / f"image-{index}.png"
                Image.new("RGB", (2, 2), color=(index + 1, 0, 0)).save(image_path)
                image_paths.append(str(image_path))
            rows = pd.DataFrame(
                {
                    "image_path": image_paths,
                    "label": PAD_UFES_NATIVE_LABELS[:2],
                }
            )
            encoder = FakeEncoder()
            features, processor_metadata = extract_embeddings(
                rows,
                model_id=DEFAULT_MODEL_ID,
                revision=DEFAULT_MODEL_REVISION,
                cache_dir=root / "cache",
                device=torch.device("cpu"),
                batch_size=2,
                num_workers=0,
                processor_loader=lambda *_args, **_kwargs: FakeProcessor(),
                model_loader=lambda *_args, **_kwargs: encoder,
            )

        self.assertEqual(tuple(features.shape), (2, 2))
        self.assertTrue(torch.allclose(features.norm(dim=1), torch.ones(2)))
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        self.assertEqual(processor_metadata["size"], {"height": 448, "width": 448})

    def test_cache_rejects_revision_mismatch(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "embeddings.pt"
            rows = pd.DataFrame({"image_path": ["image.png"], "label": [PAD_UFES_NATIVE_LABELS[0]]})
            payload = build_cache_payload(
                rows=rows,
                features=torch.tensor([[1.0, 0.0]]),
                model_id=DEFAULT_MODEL_ID,
                revision="different-revision",
                manifest_fingerprint="a" * 64,
                processor_metadata={"class": "FakeProcessor"},
            )
            torch.save(payload, cache_path)

            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                load_embedding_cache(
                    cache_path,
                    rows=rows,
                    model_id=DEFAULT_MODEL_ID,
                    revision=DEFAULT_MODEL_REVISION,
                    manifest_fingerprint="a" * 64,
                )


class ProbeTrainingTests(unittest.TestCase):
    def test_trains_only_a_linear_head_and_returns_aggregate_metrics(self) -> None:
        features = torch.eye(6).repeat(3, 1)
        targets = torch.tensor(list(range(6)) * 3)
        paths = [f"image-{index}.png" for index in range(18)]
        cache = {
            "features": features,
            "targets": targets,
            "image_paths": paths,
            "image_labels": list(PAD_UFES_NATIVE_LABELS) * 3,
            "labels": list(PAD_UFES_NATIVE_LABELS),
            "feature_dim": 6,
            "manifest_fingerprint": "a" * 64,
        }
        rows = pd.DataFrame(
            {
                "image_path": paths,
                "label": list(PAD_UFES_NATIVE_LABELS) * 3,
                "split": ["train"] * 6 + ["val"] * 6 + ["test"] * 6,
            }
        )

        with TemporaryDirectory() as tmp_dir:
            report = train_probe_fold(
                rows,
                split_summary={
                    "protocol": PROTOCOL,
                    "num_folds": 5,
                    "fold_index": 0,
                    "test_outer_fold": 0,
                    "validation_outer_fold": 1,
                    "cv_total_image_count": 18,
                },
                cache=cache,
                checkpoint_path=Path(tmp_dir) / "head.pt",
                run_dir=Path(tmp_dir) / "run",
                model_id=DEFAULT_MODEL_ID,
                revision=DEFAULT_MODEL_REVISION,
                device=torch.device("cpu"),
                epochs=2,
                batch_size=6,
                learning_rate=0.1,
                weight_decay=0.01,
                seed=42,
            )

        self.assertTrue(report["encoder_frozen"])
        self.assertEqual(report["trainable_parameter_count"], 42)
        self.assertIn("macro_f1", report["test"])
        self.assertIn("selected_train", report)

    def test_embedding_dataset_rejects_label_mismatch(self) -> None:
        cache = {
            "features": torch.ones((1, 2)),
            "targets": torch.tensor([0]),
            "image_paths": ["image.png"],
            "image_labels": [PAD_UFES_NATIVE_LABELS[0]],
            "labels": list(PAD_UFES_NATIVE_LABELS),
        }
        rows = pd.DataFrame(
            {
                "image_path": ["image.png"],
                "label": [PAD_UFES_NATIVE_LABELS[1]],
            }
        )

        with self.assertRaisesRegex(ValueError, "label differs"):
            EmbeddingDataset(rows, cache)


class ProbeSummaryTests(unittest.TestCase):
    def test_summarizes_all_fold_reports_and_selected_gap(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for fold_index in range(5):
                fold_dir = root / f"fold_{fold_index}"
                fold_dir.mkdir()
                (fold_dir / "report.json").write_text(
                    json.dumps(perfect_report(fold_index)),
                    encoding="utf-8",
                )

            summary = summarize_probe_reports(root, root / "summary.json", num_folds=5)

        self.assertEqual(summary["fold_metrics"]["macro_f1"]["mean"], 1.0)
        self.assertEqual(summary["pooled_test"]["total_support"], 30)
        self.assertEqual(summary["selected_train_val_macro_f1_gap"]["mean"], 0.0)
        self.assertTrue(summary["known_pretraining_overlap"])


if __name__ == "__main__":
    unittest.main()
