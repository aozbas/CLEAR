from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from ml.evaluation.mra_midas import CLINICAL_DISTANCES
from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.preprocessing import get_pad_ufes_transforms
from ml.training.prepare_mra_midas_cv import PROTOCOL as MRA_PROTOCOL
from ml.training.prepare_pad_ufes_cv import PROTOCOL as PAD_PROTOCOL
from ml.training.run_pad_mra_convnext_distillation_cv import (
    ARCHITECTURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DISTILLATION_WEIGHT,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    SELECTION_METRIC,
    SOURCE_CLASS_WEIGHTING,
    STUDENT_ARCHITECTURE,
    STUDENT_FEATURE_DIM,
    STUDENT_IMAGE_LOADING,
    STUDENT_PREPROCESSING,
    STUDENT_WEIGHTS_ID,
    VIEW_WEIGHTING,
    _metrics_from_confusion,
    _validate_teacher_cache_for_rows,
    add_view_masses,
    build_student_model,
    cached_student_tensor,
    checkpoint_provenance,
    metrics_from_image_logits,
    preload_student_image_tensors,
    source_class_image_weights,
    summarize_distillation_reports,
    validate_checkpoint_payload,
    validate_distillation_report,
)
from ml.training.run_pad_ufes_medsiglip_probe_cv import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    KNOWN_PRETRAINING_DATASETS,
)
from ml.training.run_pad_ufes_medsiglip_probe_cv import (
    PREPROCESSING as TEACHER_PREPROCESSING,
)


def _row(
    *,
    source: str,
    unit_id: str,
    distance: str,
    image_path: str,
    label: str,
) -> dict[str, str]:
    return {
        "split": "train",
        "source": source,
        "unit_id": unit_id,
        "record_id": f"record-{unit_id}",
        "distance": distance,
        "image_path": image_path,
        "label": label,
    }


def _perfect_confusion(per_class_support: int) -> list[list[int]]:
    size = len(PAD_UFES_NATIVE_LABELS)
    return [
        [per_class_support if row == column else 0 for column in range(size)] for row in range(size)
    ]


def _split_summary(protocol: str, fold_index: int) -> dict[str, object]:
    return {
        "protocol": protocol,
        "num_folds": 5,
        "fold_index": fold_index,
        "test_outer_fold": fold_index,
        "validation_outer_fold": (fold_index + 1) % 5,
    }


def _metric_bundle(
    *,
    pad_confusion: list[list[int]],
    mra_raw_confusion: list[list[int]],
    mra_distance_confusion: list[list[int]],
    mra_paired_confusion: list[list[int]],
) -> dict[str, object]:
    pad = _metrics_from_confusion(pad_confusion)
    mra = _metrics_from_confusion(mra_raw_confusion)
    return {
        "loss": 0.5,
        "supervised_loss": 0.4,
        "distillation_loss": 0.4,
        "mean_teacher_cosine_similarity": 0.6,
        "by_source": {"pad_ufes": pad, "mra_midas": mra},
        "source_mean_macro_f1": (float(pad["macro_f1"]) + float(mra["macro_f1"])) / 2,
        "worst_source_macro_f1": min(float(pad["macro_f1"]), float(mra["macro_f1"])),
        "combined_primary_secondary": _metrics_from_confusion(
            [
                [
                    int(pad_confusion[row][column]) + int(mra_raw_confusion[row][column])
                    for column in range(len(PAD_UFES_NATIVE_LABELS))
                ]
                for row in range(len(PAD_UFES_NATIVE_LABELS))
            ]
        ),
        "mra_distance_unit": _metrics_from_confusion(mra_distance_confusion),
        "mra_paired_lesion": _metrics_from_confusion(mra_paired_confusion),
    }


def _report(fold_index: int) -> dict[str, object]:
    inference_count = 28_000_000
    projection_count = (STUDENT_FEATURE_DIM + 1) * 1152
    train_metrics = _metric_bundle(
        pad_confusion=_perfect_confusion(3),
        mra_raw_confusion=_perfect_confusion(6),
        mra_distance_confusion=_perfect_confusion(6),
        mra_paired_confusion=_perfect_confusion(3),
    )
    val_metrics = copy.deepcopy(train_metrics)
    train_metrics["source_mean_macro_f1"] = 0.90
    val_metrics["source_mean_macro_f1"] = 0.85
    test_metrics = _metric_bundle(
        pad_confusion=_perfect_confusion(2),
        mra_raw_confusion=_perfect_confusion(4),
        mra_distance_confusion=_perfect_confusion(4),
        mra_paired_confusion=_perfect_confusion(2),
    )
    return {
        "context": "synthetic experimental classification report",
        "architecture": ARCHITECTURE,
        "student_architecture": STUDENT_ARCHITECTURE,
        "student_weights": STUDENT_WEIGHTS_ID,
        "student_preprocessing": STUDENT_PREPROCESSING,
        "student_image_loading": STUDENT_IMAGE_LOADING,
        "teacher_model_id": DEFAULT_MODEL_ID,
        "teacher_model_revision": DEFAULT_MODEL_REVISION,
        "teacher_preprocessing": TEACHER_PREPROCESSING,
        "teacher_encoder_frozen": True,
        "teacher_encoder_trainable_parameter_count": 0,
        "teacher_encoder_parameter_count": 878_300_338,
        "teacher_feature_dim": 1152,
        "known_pad_pretraining_overlap": True,
        "known_pretraining_datasets": list(KNOWN_PRETRAINING_DATASETS),
        "mra_role": "authorized_multisource_development",
        "mra_training_input": "original_clinical_images",
        "mra_primary_metric_unit": "raw_image",
        "mra_secondary_aggregation": "mean_within_distance_then_equal_distance_mean_then_l2",
        "manifest_fingerprint": "a" * 64,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": ["pad_ufes", "mra_midas"],
        "seed": DEFAULT_SEED,
        "pad_split_summary": _split_summary(PAD_PROTOCOL, fold_index),
        "mra_split_summary": _split_summary(MRA_PROTOCOL, fold_index),
        "source_total_raw_image_counts": {"pad_ufes": 60, "mra_midas": 120},
        "source_total_effective_unit_counts": {"pad_ufes": 60.0, "mra_midas": 60.0},
        "dataset_raw_image_counts": {},
        "dataset_effective_unit_counts": {},
        "inference_parameter_count": inference_count,
        "projection_parameter_count": projection_count,
        "training_parameter_count": inference_count + projection_count,
        "checkpoint_bytes": 120_000_000 + fold_index,
        "hyperparameters": {
            "epochs": DEFAULT_EPOCHS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "supervised_loss_weight": 1.0,
            "distillation_loss_weight": DEFAULT_DISTILLATION_WEIGHT,
            "optimizer": "AdamW",
            "schedule": "none",
            "augmentation_profile": "baseline",
            "view_weighting": VIEW_WEIGHTING,
            "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        },
        "best_epoch": 2,
        "selection_metric": SELECTION_METRIC,
        "best_val_source_mean_macro_f1": 0.85,
        "history": [],
        "selected_train": train_metrics,
        "selected_val": val_metrics,
        "test": test_metrics,
        "caveat": "synthetic",
    }


class StudentImageCacheTests(unittest.TestCase):
    def test_cached_pipeline_is_bit_identical_to_original_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "asymmetric.png"
            image = Image.new("RGB", (19, 11))
            for x in range(image.width):
                for y in range(image.height):
                    image.putpixel(
                        (x, y),
                        ((x * 13) % 256, (y * 23) % 256, ((x + y) * 17) % 256),
                    )
            image.save(image_path)
            rows = pd.DataFrame({"image_path": [str(image_path), str(image_path)]})

            cache = preload_student_image_tensors(rows)
            self.assertEqual(list(cache), [str(image_path)])
            with Image.open(image_path) as source_image:
                expected_val = get_pad_ufes_transforms("val", augmentation_profile="baseline")(
                    source_image.convert("RGB")
                )
            self.assertTrue(
                torch.equal(
                    expected_val,
                    cached_student_tensor(cache[str(image_path)], training=False),
                )
            )

            original_train_transform = get_pad_ufes_transforms(
                "train", augmentation_profile="baseline"
            )
            cached_results: list[torch.Tensor] = []
            for seed in (0, 1):
                with Image.open(image_path) as source_image:
                    torch.manual_seed(seed)
                    expected_train = original_train_transform(source_image.convert("RGB"))
                torch.manual_seed(seed)
                actual_train = cached_student_tensor(
                    cache[str(image_path)],
                    training=True,
                )
                self.assertTrue(torch.equal(expected_train, actual_train))
                cached_results.append(actual_train)
            self.assertFalse(torch.equal(cached_results[0], cached_results[1]))


class ViewWeightTests(unittest.TestCase):
    def test_repeated_mra_views_share_distance_and_lesion_mass(self) -> None:
        label = PAD_UFES_NATIVE_LABELS[0]
        distance_a, distance_b = CLINICAL_DISTANCES
        rows = pd.DataFrame(
            [
                _row(
                    source="mra_midas",
                    unit_id="u1",
                    distance=distance_a,
                    image_path="a1.jpg",
                    label=label,
                ),
                _row(
                    source="mra_midas",
                    unit_id="u1",
                    distance=distance_a,
                    image_path="a2.jpg",
                    label=label,
                ),
                _row(
                    source="mra_midas",
                    unit_id="u1",
                    distance=distance_b,
                    image_path="b1.jpg",
                    label=label,
                ),
            ]
        )
        weighted = add_view_masses(rows)
        self.assertEqual(weighted["view_mass"].tolist(), [0.25, 0.25, 0.5])
        self.assertAlmostEqual(float(weighted["view_mass"].sum()), 1.0)

    def test_incomplete_mra_pair_is_rejected(self) -> None:
        rows = pd.DataFrame(
            [
                _row(
                    source="mra_midas",
                    unit_id="u1",
                    distance=CLINICAL_DISTANCES[0],
                    image_path="a.jpg",
                    label=PAD_UFES_NATIVE_LABELS[0],
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "missing a locked distance"):
            add_view_masses(rows)

    def test_source_class_weights_equalize_all_twelve_cells(self) -> None:
        distance_a, distance_b = CLINICAL_DISTANCES
        rows: list[dict[str, str]] = []
        for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
            rows.append(
                _row(
                    source="pad_ufes",
                    unit_id=f"pad-{label_index}",
                    distance="single",
                    image_path=f"pad-{label_index}.jpg",
                    label=label,
                )
            )
            rows.extend(
                [
                    _row(
                        source="mra_midas",
                        unit_id=f"mra-{label_index}",
                        distance=distance_a,
                        image_path=f"mra-{label_index}-a1.jpg",
                        label=label,
                    ),
                    _row(
                        source="mra_midas",
                        unit_id=f"mra-{label_index}",
                        distance=distance_a,
                        image_path=f"mra-{label_index}-a2.jpg",
                        label=label,
                    ),
                    _row(
                        source="mra_midas",
                        unit_id=f"mra-{label_index}",
                        distance=distance_b,
                        image_path=f"mra-{label_index}-b.jpg",
                        label=label,
                    ),
                ]
            )
        frame = add_view_masses(pd.DataFrame(rows)).reset_index(drop=True)
        weights = source_class_image_weights(frame)
        cell_totals = {}
        for source in ("pad_ufes", "mra_midas"):
            for label in PAD_UFES_NATIVE_LABELS:
                mask = torch.tensor(
                    ((frame["source"] == source) & (frame["label"] == label)).tolist(),
                    dtype=torch.bool,
                )
                cell_totals[(source, label)] = float(weights[mask].sum())
        self.assertEqual(len({round(total, 6) for total in cell_totals.values()}), 1)
        self.assertAlmostEqual(float(weights.sum()), 12.0)


class TeacherAndStudentTests(unittest.TestCase):
    def test_teacher_cache_rejects_label_drift(self) -> None:
        rows = pd.DataFrame(
            [
                _row(
                    source="pad_ufes",
                    unit_id="p1",
                    distance="single",
                    image_path="p1.jpg",
                    label=PAD_UFES_NATIVE_LABELS[0],
                )
            ]
        )
        cache = {
            "image_paths": ["p1.jpg"],
            "image_sources": ["pad_ufes"],
            "image_labels": [PAD_UFES_NATIVE_LABELS[1]],
            "features": torch.tensor([[1.0, 0.0]]),
        }
        with self.assertRaisesRegex(ValueError, "source or label differs"):
            _validate_teacher_cache_for_rows(rows, cache)

    def test_student_parameter_accounting_excludes_projection_from_inference(self) -> None:
        model = build_student_model(teacher_feature_dim=8, pretrained=False)
        self.assertEqual(model.student_feature_dim, STUDENT_FEATURE_DIM)
        self.assertEqual(model.projection_parameter_count(), (STUDENT_FEATURE_DIM + 1) * 8)
        self.assertEqual(
            model.training_parameter_count(),
            model.inference_parameter_count() + model.projection_parameter_count(),
        )
        self.assertLessEqual(model.inference_parameter_count(), 30_000_000)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_checkpoint_provenance_rejects_teacher_revision_drift(self) -> None:
        provenance = checkpoint_provenance(
            fold_index=0,
            manifest_fingerprint="b" * 64,
            teacher_feature_dim=1152,
            seed=DEFAULT_SEED,
        )
        payload = {
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "provenance": provenance,
            "labels": list(PAD_UFES_NATIVE_LABELS),
            "sources": ["pad_ufes", "mra_midas"],
            "epoch": 1,
        }
        validate_checkpoint_payload(payload, expected_provenance=provenance)
        drifted = copy.deepcopy(payload)
        drifted["provenance"]["teacher_model_revision"] = "changed"
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            validate_checkpoint_payload(drifted, expected_provenance=provenance)


class MetricTests(unittest.TestCase):
    def test_raw_image_and_paired_lesion_metrics_remain_separate(self) -> None:
        rows: list[dict[str, str]] = []
        logits: list[torch.Tensor] = []
        targets: list[int] = []
        distance_a, distance_b = CLINICAL_DISTANCES
        size = len(PAD_UFES_NATIVE_LABELS)
        for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
            pad_logits = torch.zeros(size)
            pad_logits[label_index] = 6.0
            rows.append(
                _row(
                    source="pad_ufes",
                    unit_id=f"pad-{label_index}",
                    distance="single",
                    image_path=f"pad-{label_index}.jpg",
                    label=label,
                )
            )
            logits.append(pad_logits)
            targets.append(label_index)
            correct = torch.zeros(size)
            correct[label_index] = 6.0
            wrong = torch.zeros(size)
            wrong[(label_index + 1) % size] = 4.0
            wrong[label_index] = 3.0
            for distance, image_logits in ((distance_a, correct), (distance_b, wrong)):
                rows.append(
                    _row(
                        source="mra_midas",
                        unit_id=f"mra-{label_index}",
                        distance=distance,
                        image_path=f"mra-{label_index}-{distance}.jpg",
                        label=label,
                    )
                )
                logits.append(image_logits)
                targets.append(label_index)
        metrics = metrics_from_image_logits(
            pd.DataFrame(rows),
            torch.stack(logits),
            torch.tensor(targets),
        )
        self.assertEqual(metrics["by_source"]["pad_ufes"]["macro_f1"], 1.0)
        self.assertLess(metrics["by_source"]["mra_midas"]["macro_f1"], 1.0)
        self.assertEqual(metrics["mra_paired_lesion"]["macro_f1"], 1.0)
        self.assertEqual(metrics["mra_paired_lesion"]["total_support"], 6)
        self.assertEqual(metrics["mra_distance_unit"]["total_support"], 12)


class ReportTests(unittest.TestCase):
    def test_report_rejects_locked_configuration_drift(self) -> None:
        report = _report(0)
        validate_distillation_report(report, fold_index=0)
        report["student_weights"] = "changed"
        with self.assertRaisesRegex(ValueError, "locked protocol"):
            validate_distillation_report(report, fold_index=0)

    def test_summary_pools_outer_tests_and_passes_locked_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for fold_index in range(5):
                fold = root / f"fold_{fold_index}"
                fold.mkdir()
                (fold / "report.json").write_text(
                    json.dumps(_report(fold_index)),
                    encoding="utf-8",
                )
            summary = summarize_distillation_reports(root, root / "summary.json")
        self.assertTrue(summary["decision_rules"]["all_pass"])
        self.assertEqual(summary["pooled_primary_by_source"]["pad_ufes"]["total_support"], 60)
        self.assertEqual(
            summary["pooled_primary_by_source"]["mra_midas"]["total_support"],
            120,
        )
        self.assertEqual(summary["pooled_mra_distance_unit"]["total_support"], 120)
        self.assertEqual(summary["pooled_mra_paired_lesion"]["total_support"], 60)

    def test_summary_records_failed_rule_without_changing_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for fold_index in range(5):
                report = _report(fold_index)
                bad = [[0 for _ in PAD_UFES_NATIVE_LABELS] for _ in PAD_UFES_NATIVE_LABELS]
                for row in range(len(PAD_UFES_NATIVE_LABELS)):
                    bad[row][0] = 4
                report["test"]["by_source"]["mra_midas"] = _metrics_from_confusion(bad)
                fold = root / f"fold_{fold_index}"
                fold.mkdir()
                (fold / "report.json").write_text(json.dumps(report), encoding="utf-8")
            summary = summarize_distillation_reports(root, root / "summary.json")
        self.assertFalse(summary["decision_rules"]["all_pass"])
        self.assertFalse(summary["decision_rules"]["pooled_mra_raw_image_macro_f1_gte_0_4000"])


if __name__ == "__main__":
    unittest.main()
