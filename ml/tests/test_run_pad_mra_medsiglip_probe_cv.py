import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_mra_midas_cv import (
    PROTOCOL as MRA_PROTOCOL,
)
from ml.training.prepare_mra_midas_cv import (
    assign_record_folds,
)
from ml.training.prepare_pad_ufes_cv import PROTOCOL as PAD_PROTOCOL
from ml.training.run_pad_mra_medsiglip_probe_cv import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    MRA_AGGREGATION,
    SELECTION_METRIC,
    SOURCE_ORDER,
    WEIGHTING,
    MultiSourceEmbeddingDataset,
    aggregate_unit_embeddings,
    build_image_embedding_cache,
    load_image_embedding_cache,
    load_multi_source_manifests,
    source_class_weights,
    summarize_probe_reports,
    train_probe_fold,
)
from ml.training.run_pad_ufes_medsiglip_probe_cv import KNOWN_PRETRAINING_DATASETS


def write_pad_manifests(root: Path) -> None:
    paths = [f"pad-image-{index}.png" for index in range(30)]
    labels = [PAD_UFES_NATIVE_LABELS[index // 5] for index in range(30)]
    for fold_index in range(5):
        validation_fold = (fold_index + 1) % 5
        rows = []
        for index, (path, label) in enumerate(zip(paths, labels, strict=True)):
            outer_fold = index % 5
            split = (
                "test"
                if outer_fold == fold_index
                else "val"
                if outer_fold == validation_fold
                else "train"
            )
            rows.append({"split": split, "image_path": path, "label": label})
        fold_path = root / f"fold_{fold_index}.csv"
        pd.DataFrame(rows).to_csv(fold_path, index=False)
        summary = {
            "dataset": "pad_ufes",
            "label_mode": "native",
            "split_strategy": "patient",
            "protocol": PAD_PROTOCOL,
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
        fold_path.with_suffix(".summary.json").write_text(json.dumps(summary), encoding="utf-8")


def write_mra_manifests(root: Path) -> list[Path]:
    fold_paths = []
    units = []
    for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
        for outer_fold in range(5):
            unit_id = f"mra-unit-{label_index}-{outer_fold}"
            record_id = f"mra-record-{label_index}-{outer_fold}"
            for distance in ("6in", "1ft"):
                units.append(
                    {
                        "source": "mra_midas",
                        "unit_id": unit_id,
                        "record_id": record_id,
                        "distance": distance,
                        "image_path": f"{unit_id}-{distance}.jpg",
                        "label": label,
                        "outer_fold": outer_fold,
                    }
                )
    for fold_index in range(5):
        validation_fold = (fold_index + 1) % 5
        rows = []
        for unit in units:
            split = (
                "test"
                if unit["outer_fold"] == fold_index
                else "val"
                if unit["outer_fold"] == validation_fold
                else "train"
            )
            rows.append(
                {
                    "split": split,
                    **{key: value for key, value in unit.items() if key != "outer_fold"},
                }
            )
        fold_path = root / f"fold_{fold_index}.csv"
        pd.DataFrame(rows).to_csv(fold_path, index=False)
        summary = {
            "dataset": "mra_midas",
            "role": "authorized_multisource_development",
            "protocol": MRA_PROTOCOL,
            "num_folds": 5,
            "fold_index": fold_index,
            "test_outer_fold": fold_index,
            "validation_outer_fold": validation_fold,
            "group_key": "midas_record_id",
            "feature_unit": "paired_distance_lesion_embedding",
            "distance_aggregation": MRA_AGGREGATION,
            "record_overlap_count": 0,
            "lesion_overlap_count": 0,
            "image_count": 60,
            "unit_count": 30,
            "record_count": 30,
        }
        fold_path.with_suffix(".summary.json").write_text(json.dumps(summary), encoding="utf-8")
        fold_paths.append(fold_path)
    return fold_paths


def perfect_metrics(support_per_class: int = 1) -> dict[str, object]:
    size = len(PAD_UFES_NATIVE_LABELS)
    confusion = [[0 for _ in range(size)] for _ in range(size)]
    per_class = {}
    for index, label in enumerate(PAD_UFES_NATIVE_LABELS):
        confusion[index][index] = support_per_class
        per_class[label] = {
            "true_positive": support_per_class,
            "false_positive": 0,
            "false_negative": 0,
            "support": support_per_class,
            "predicted": support_per_class,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    return {
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "macro_f1": 1.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "total_support": support_per_class * size,
    }


def perfect_report(fold_index: int) -> dict[str, object]:
    source_metrics = {source: perfect_metrics() for source in SOURCE_ORDER}
    combined = perfect_metrics(support_per_class=2)
    selected = {
        "loss": 0.1,
        **combined,
        "by_source": source_metrics,
        "source_mean_macro_f1": 1.0,
        "worst_source_macro_f1": 1.0,
    }
    return {
        "architecture": "medsiglip_frozen_multisource_linear_probe",
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "encoder_frozen": True,
        "encoder_trainable_parameter_count": 0,
        "encoder_parameter_count": 100,
        "feature_dim": 6,
        "trainable_parameter_count": 42,
        "manifest_fingerprint": "a" * 64,
        "known_pad_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "mra_role": "authorized_multisource_development",
        "embedding_normalization": "l2",
        "preprocessing": "medsiglip_auto_image_processor_slow_native_448",
        "mra_aggregation": MRA_AGGREGATION,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "seed": 42,
        "selection_metric": SELECTION_METRIC,
        "pad_split_summary": {
            "protocol": PAD_PROTOCOL,
            "num_folds": 5,
            "fold_index": fold_index,
            "test_outer_fold": fold_index,
            "validation_outer_fold": (fold_index + 1) % 5,
        },
        "mra_split_summary": {
            "protocol": MRA_PROTOCOL,
            "num_folds": 5,
            "fold_index": fold_index,
            "test_outer_fold": fold_index,
            "validation_outer_fold": (fold_index + 1) % 5,
        },
        "source_unit_counts": {source: 30 for source in SOURCE_ORDER},
        "hyperparameters": {
            "epochs": 100,
            "batch_size": 128,
            "learning_rate": 1e-2,
            "weight_decay": 1e-2,
            "source_class_weighting": WEIGHTING,
            "optimizer": "AdamW",
            "schedule": "none",
        },
        "best_epoch": 1,
        "best_val_source_mean_macro_f1": 1.0,
        "selected_train": selected,
        "selected_val": selected,
        "test": selected,
    }


class MraFoldTests(unittest.TestCase):
    def test_assigns_each_record_to_one_stratified_outer_fold(self) -> None:
        rows = []
        for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
            for record_index in range(5):
                rows.append({"record_id": f"record-{label_index}-{record_index}", "label": label})
        assignments = assign_record_folds(pd.DataFrame(rows), num_folds=5, seed=42)

        self.assertEqual(len(assignments), 30)
        for label_index in range(6):
            observed = {
                assignments[f"record-{label_index}-{record_index}"] for record_index in range(5)
            }
            self.assertEqual(observed, set(range(5)))

    def test_loads_consistent_pad_and_mra_rotating_manifests(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pad_root = root / "pad"
            mra_root = root / "mra"
            pad_root.mkdir()
            mra_root.mkdir()
            write_pad_manifests(pad_root)
            write_mra_manifests(mra_root)
            manifests = load_multi_source_manifests(pad_root, mra_root, num_folds=5)

        self.assertEqual(len(manifests.pad.unique_rows), 30)
        self.assertEqual(len(manifests.image_rows), 90)
        self.assertEqual(len(manifests.fingerprint), 64)

    def test_rejects_mra_record_split_across_roles(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pad_root = root / "pad"
            mra_root = root / "mra"
            pad_root.mkdir()
            mra_root.mkdir()
            write_pad_manifests(pad_root)
            fold_paths = write_mra_manifests(mra_root)
            rows = pd.read_csv(fold_paths[0])
            rows.loc[0, "split"] = "train"
            rows.to_csv(fold_paths[0], index=False)

            with self.assertRaisesRegex(ValueError, "splits a lesion|splits a record"):
                load_multi_source_manifests(pad_root, mra_root, num_folds=5)


class EmbeddingTests(unittest.TestCase):
    def test_aggregates_mra_views_with_equal_distance_weight_and_l2_normalization(self) -> None:
        image_rows = pd.DataFrame(
            [
                {
                    "source": "pad_ufes",
                    "unit_id": "pad-unit",
                    "distance": "single",
                    "image_path": "pad.jpg",
                    "label": PAD_UFES_NATIVE_LABELS[0],
                },
                {
                    "source": "mra_midas",
                    "unit_id": "mra-unit",
                    "distance": "6in",
                    "image_path": "mra-6-a.jpg",
                    "label": PAD_UFES_NATIVE_LABELS[1],
                },
                {
                    "source": "mra_midas",
                    "unit_id": "mra-unit",
                    "distance": "6in",
                    "image_path": "mra-6-b.jpg",
                    "label": PAD_UFES_NATIVE_LABELS[1],
                },
                {
                    "source": "mra_midas",
                    "unit_id": "mra-unit",
                    "distance": "1ft",
                    "image_path": "mra-1.jpg",
                    "label": PAD_UFES_NATIVE_LABELS[1],
                },
            ]
        )
        features = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
        )
        cache = {"image_paths": image_rows["image_path"].tolist(), "features": features}
        units = aggregate_unit_embeddings(image_rows, cache)

        mra_index = list(zip(units["source"], units["unit_id"], strict=True)).index(
            ("mra_midas", "mra-unit")
        )
        expected = torch.tensor([2**-0.5, 2**-0.5])
        self.assertTrue(torch.allclose(units["features"][mra_index], expected))
        self.assertTrue(torch.allclose(units["features"].norm(dim=1), torch.ones(2)))

    def test_embedding_cache_rejects_source_or_manifest_drift(self) -> None:
        rows = pd.DataFrame(
            {
                "source": ["pad_ufes"],
                "unit_id": ["pad-unit"],
                "distance": ["single"],
                "image_path": ["pad.jpg"],
                "label": [PAD_UFES_NATIVE_LABELS[0]],
            }
        )
        payload = build_image_embedding_cache(
            rows,
            features=torch.tensor([[1.0, 0.0]]),
            model_id=DEFAULT_MODEL_ID,
            revision=DEFAULT_MODEL_REVISION,
            manifest_fingerprint="a" * 64,
            processor_metadata={
                "class": "FakeProcessor",
                "encoder_trainable_parameter_count": 0,
            },
        )
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cache.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                load_image_embedding_cache(
                    path,
                    image_rows=rows,
                    model_id=DEFAULT_MODEL_ID,
                    revision=DEFAULT_MODEL_REVISION,
                    manifest_fingerprint="b" * 64,
                )


class WeightingAndTrainingTests(unittest.TestCase):
    def test_source_class_weights_give_each_cell_equal_total_weight(self) -> None:
        rows = []
        for source in SOURCE_ORDER:
            for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
                repeats = label_index + (2 if source == "pad_ufes" else 1)
                rows.extend(
                    {"source": source, "label": label, "unit_id": f"{source}-{label}-{index}"}
                    for index in range(repeats)
                )
        frame = pd.DataFrame(rows)
        weights = source_class_weights(frame)
        frame["weight"] = weights.tolist()
        totals = frame.groupby(["source", "label"])["weight"].sum()

        self.assertTrue(all(abs(value - totals.iloc[0]) < 1e-5 for value in totals))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_trains_only_linear_head_and_records_source_aware_selection(self) -> None:
        rows = []
        feature_rows = []
        sources = []
        unit_ids = []
        labels = []
        for split in ("train", "val", "test"):
            for source in SOURCE_ORDER:
                for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
                    unit_id = f"{split}-{source}-{label}"
                    rows.append(
                        {"split": split, "source": source, "unit_id": unit_id, "label": label}
                    )
                    feature_rows.append(torch.eye(6)[label_index])
                    sources.append(source)
                    unit_ids.append(unit_id)
                    labels.append(label)
        frame = pd.DataFrame(rows)
        unit_cache = {
            "features": torch.stack(feature_rows),
            "feature_dim": 6,
            "source": sources,
            "unit_id": unit_ids,
            "label": labels,
        }
        image_cache = {
            "manifest_fingerprint": "a" * 64,
            "processor": {
                "encoder_parameter_count": 100,
                "encoder_trainable_parameter_count": 0,
            },
        }
        pad_summary = {
            "protocol": PAD_PROTOCOL,
            "num_folds": 5,
            "fold_index": 0,
            "test_outer_fold": 0,
            "validation_outer_fold": 1,
        }
        mra_summary = {**pad_summary, "protocol": MRA_PROTOCOL}
        with TemporaryDirectory() as tmp_dir:
            report = train_probe_fold(
                frame,
                pad_summary=pad_summary,
                mra_summary=mra_summary,
                unit_cache=unit_cache,
                image_cache=image_cache,
                checkpoint_path=Path(tmp_dir) / "head.pt",
                run_dir=Path(tmp_dir) / "run",
                model_id=DEFAULT_MODEL_ID,
                revision=DEFAULT_MODEL_REVISION,
                device=torch.device("cpu"),
                epochs=2,
                batch_size=12,
                learning_rate=0.1,
                weight_decay=0.01,
                seed=42,
            )

        self.assertEqual(report["trainable_parameter_count"], 42)
        self.assertEqual(report["encoder_trainable_parameter_count"], 0)
        self.assertEqual(report["selection_metric"], SELECTION_METRIC)
        self.assertIn("source_mean_macro_f1", report["test"])

    def test_embedding_dataset_rejects_unit_label_drift(self) -> None:
        rows = pd.DataFrame(
            {
                "source": ["pad_ufes"],
                "unit_id": ["unit"],
                "label": [PAD_UFES_NATIVE_LABELS[1]],
            }
        )
        cache = {
            "features": torch.ones((1, 2)),
            "source": ["pad_ufes"],
            "unit_id": ["unit"],
            "label": [PAD_UFES_NATIVE_LABELS[0]],
        }
        with self.assertRaisesRegex(ValueError, "label differs"):
            MultiSourceEmbeddingDataset(rows, cache)


class SummaryTests(unittest.TestCase):
    def test_summarizes_source_metrics_and_preregistered_rules(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for fold_index in range(5):
                fold_dir = root / f"fold_{fold_index}"
                fold_dir.mkdir()
                (fold_dir / "report.json").write_text(
                    json.dumps(perfect_report(fold_index)), encoding="utf-8"
                )
            summary = summarize_probe_reports(root, root / "summary.json", num_folds=5)

        self.assertEqual(summary["pooled_by_source"]["pad_ufes"]["macro_f1"], 1.0)
        self.assertEqual(summary["pooled_by_source"]["mra_midas"]["macro_f1"], 1.0)
        self.assertEqual(summary["pooled_source_mean_macro_f1"], 1.0)
        self.assertTrue(summary["decision_rules"]["all_pass"])


if __name__ == "__main__":
    unittest.main()
