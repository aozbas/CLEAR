import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.preprocessing import get_pad_ufes_transforms
from ml.training.train_pad_ufes import (
    PadUfesDataset,
    add_macro_metrics,
    build_lr_scheduler,
    build_transfer_model,
    load_training_split,
    save_checkpoint,
)


def write_split(root: Path) -> Path:
    split_path = root / "pad_ufes_native_training.csv"
    rows = []
    for split in ("train", "val", "test"):
        for label in PAD_UFES_NATIVE_LABELS:
            rows.append(
                {
                    "split": split,
                    "image_path": f"{split}-{label}.png",
                    "label": label,
                }
            )
    pd.DataFrame(rows).to_csv(split_path, index=False)
    summary = {
        "dataset": "pad_ufes",
        "label_mode": "native",
        "split_strategy": "patient",
        "group_key": "patient_id",
        "patient_overlap_count": 0,
        "patient_lesion_overlap_count": 0,
        "image_count": len(rows),
        "images_by_split": {"train": 6, "val": 6, "test": 6},
    }
    split_path.with_suffix(".summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return split_path


class LoadTrainingSplitTests(unittest.TestCase):
    def test_accepts_verified_patient_grouped_native_split(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            split_path = write_split(Path(tmp_dir))
            rows, summary = load_training_split(split_path)

        self.assertEqual(len(rows), 18)
        self.assertEqual(summary["split_strategy"], "patient")

    def test_rejects_non_patient_summary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            split_path = write_split(Path(tmp_dir))
            summary_path = split_path.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["split_strategy"] = "all-test"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "verified patient-grouped"):
                load_training_split(split_path)

    def test_rejects_missing_label_coverage(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            split_path = write_split(Path(tmp_dir))
            rows = pd.read_csv(split_path)
            rows = rows[~((rows["split"] == "test") & (rows["label"] == "squamous_cell_carcinoma"))]
            rows.to_csv(split_path, index=False)

            with self.assertRaisesRegex(ValueError, "missing label coverage"):
                load_training_split(split_path)


class PadUfesDatasetTests(unittest.TestCase):
    def test_capped_rows_keep_every_native_label(self) -> None:
        rows = []
        for label in PAD_UFES_NATIVE_LABELS:
            for index in range(2):
                rows.append(
                    {
                        "split": "train",
                        "image_path": f"{label}-{index}.png",
                        "label": label,
                    }
                )
        dataset = PadUfesDataset(
            pd.DataFrame(rows),
            "train",
            max_samples=len(PAD_UFES_NATIVE_LABELS),
            seed=42,
        )

        self.assertEqual(len(dataset), len(PAD_UFES_NATIVE_LABELS))
        self.assertEqual(sorted(dataset.labels()), list(range(len(PAD_UFES_NATIVE_LABELS))))


class TransferModelTests(unittest.TestCase):
    def test_builds_six_class_resnet_without_downloading_test_weights(self) -> None:
        model = build_transfer_model(weights="none")

        self.assertEqual(model.fc.out_features, len(PAD_UFES_NATIVE_LABELS))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_add_macro_metrics_averages_all_native_classes(self) -> None:
        per_class = {
            label: {"precision": 0.5, "recall": 0.25, "f1": 1.0 / 3.0, "support": 1}
            for label in PAD_UFES_NATIVE_LABELS
        }

        metrics = add_macro_metrics(
            {
                "loss": 1.0,
                "accuracy": 0.25,
                "per_class": per_class,
                "confusion_matrix": [],
            }
        )

        self.assertEqual(metrics["balanced_accuracy"], 0.25)
        self.assertEqual(metrics["macro_precision"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0 / 3.0)

    def test_regularized_profile_augments_train_but_not_validation(self) -> None:
        train_transform = get_pad_ufes_transforms(
            "train",
            augmentation_profile="regularized_v2",
        )
        val_transform = get_pad_ufes_transforms(
            "val",
            augmentation_profile="regularized_v2",
        )

        self.assertTrue(
            any(
                isinstance(transform, transforms.RandomVerticalFlip)
                for transform in train_transform.transforms
            )
        )
        self.assertTrue(
            any(
                isinstance(transform, transforms.RandomErasing)
                for transform in train_transform.transforms
            )
        )
        self.assertFalse(
            any(
                isinstance(transform, transforms.RandomVerticalFlip)
                for transform in val_transform.transforms
            )
        )

    def test_cosine_scheduler_decays_learning_rate(self) -> None:
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = build_lr_scheduler(optimizer, schedule="cosine", epochs=4)

        self.assertIsNotNone(scheduler)
        optimizer.step()
        scheduler.step()

        self.assertLess(optimizer.param_groups[0]["lr"], 1e-4)

    def test_checkpoint_records_image_only_native_configuration(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "model.pt"
            save_checkpoint(
                checkpoint,
                nn.Linear(2, 2),
                epoch=3,
                val_metrics={"macro_f1": 0.4},
                weights="none",
                seed=42,
                hyperparameters={
                    "learning_rate": 0.001,
                    "augmentation_profile": "regularized_v2",
                    "label_smoothing": 0.1,
                    "lr_schedule": "cosine",
                },
                augmentation_profile="regularized_v2",
            )
            saved = torch.load(checkpoint, map_location="cpu")

        self.assertEqual(saved["input_mode"], "image_only")
        self.assertEqual(saved["label_set"], "pad_ufes_native")
        self.assertEqual(saved["selection_metric"], "val_macro_f1")
        self.assertEqual(saved["labels"], list(PAD_UFES_NATIVE_LABELS))
        self.assertEqual(saved["augmentation_profile"], "regularized_v2")
        self.assertEqual(saved["hyperparameters"]["label_smoothing"], 0.1)


if __name__ == "__main__":
    unittest.main()
