from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.prepare_hiba_cv import PROTOCOL as HIBA_PROTOCOL
from ml.training.prepare_pad_ufes_cv import PROTOCOL as PAD_PROTOCOL
from ml.training.run_pad_hiba_convnext_cv import (
    ARCHITECTURE,
    AUGMENTATION_PROFILE,
    HIBA_VIEW_WEIGHTING,
    PREPROCESSING,
    PRETRAINED_WEIGHTS,
    SELECTION_METRIC,
    SOURCE_CLASS_WEIGHTING,
    SOURCE_ORDER,
    _metrics_from_confusion,
    load_multi_source_manifests,
    metrics_from_probabilities,
    source_class_image_weights,
    summarize_reports,
)
from ml.training.train_pad_ufes import pretrained_weights_id

PAD_SUPPORT = (257, 845, 52, 244, 192, 708)
HIBA_LESION_SUPPORT = (17, 112, 58, 53, 47, 21)
HIBA_IMAGE_SUPPORT = (17, 112, 59, 53, 47, 21)


def _role(outer_fold: int, fold_index: int) -> str:
    if outer_fold == fold_index:
        return "test"
    if outer_fold == (fold_index + 1) % 5:
        return "val"
    return "train"


def _write_manifest_fixture(root: Path) -> tuple[Path, Path]:
    pad_dir = root / "pad"
    hiba_dir = root / "hiba"
    pad_dir.mkdir()
    hiba_dir.mkdir()
    for fold_index in range(5):
        pad_rows = []
        hiba_rows = []
        image_number = 0
        for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
            for outer_fold in range(5):
                image_number += 1
                pad_rows.append(
                    {
                        "split": _role(outer_fold, fold_index),
                        "image_path": f"/private/pad-{label_index}-{outer_fold}.png",
                        "label": label,
                    }
                )
                hiba_rows.append(
                    {
                        "split": _role(outer_fold, fold_index),
                        "source": "hiba",
                        "image_path": f"/private/hiba-{label_index}-{outer_fold}.jpg",
                        "label": label,
                        "patient_id": f"private-patient-{label_index}-{outer_fold}",
                        "lesion_id": f"private-lesion-{label_index}-{outer_fold}",
                        "isic_id": f"ISIC_{image_number:07d}",
                    }
                )
        hiba_rows.append(
            {
                **hiba_rows[10],
                "image_path": "/private/hiba-repeated-view.jpg",
                "isic_id": "ISIC_9999999",
            }
        )
        pd.DataFrame(pad_rows).to_csv(pad_dir / f"fold_{fold_index}.csv", index=False)
        pd.DataFrame(hiba_rows).to_csv(hiba_dir / f"fold_{fold_index}.csv", index=False)
        for directory, protocol, extra in (
            (pad_dir, PAD_PROTOCOL, {}),
            (hiba_dir, HIBA_PROTOCOL, {"lesion_overlap_count": 0}),
        ):
            summary = {
                "protocol": protocol,
                "num_folds": 5,
                "fold_index": fold_index,
                "test_outer_fold": fold_index,
                "validation_outer_fold": (fold_index + 1) % 5,
                "patient_overlap_count": 0,
                **extra,
            }
            (directory / f"fold_{fold_index}.summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
    return pad_dir, hiba_dir


def _identity_confusion(counts: tuple[int, ...]) -> list[list[int]]:
    size = len(counts)
    return [[counts[row] if row == column else 0 for column in range(size)] for row in range(size)]


def _fold_confusion(counts: tuple[int, ...], fold_index: int) -> list[list[int]]:
    fold_counts = tuple(count // 5 + (1 if fold_index < count % 5 else 0) for count in counts)
    return _identity_confusion(fold_counts)


def _metric_bundle(
    pad_confusion: list[list[int]],
    hiba_image_confusion: list[list[int]],
    hiba_lesion_confusion: list[list[int]],
) -> dict[str, object]:
    pad = _metrics_from_confusion(pad_confusion)
    hiba_image = _metrics_from_confusion(hiba_image_confusion)
    hiba_lesion = _metrics_from_confusion(hiba_lesion_confusion)
    source_mean = (float(pad["macro_f1"]) + float(hiba_lesion["macro_f1"])) / 2
    return {
        "loss": 0.1,
        "by_source": {"pad_ufes": pad, "hiba": hiba_image},
        "hiba_lesion": hiba_lesion,
        "primary_source_mean_macro_f1": source_mean,
        "primary_worst_source_macro_f1": min(
            float(pad["macro_f1"]), float(hiba_lesion["macro_f1"])
        ),
        "primary_units": {"pad_ufes": "image", "hiba": "lesion"},
    }


def _write_report_fixture(
    root: Path,
    *,
    fail_hiba_scc: bool = False,
    pretrained: bool = True,
) -> None:
    for fold_index in range(5):
        pad_confusion = _fold_confusion(PAD_SUPPORT, fold_index)
        hiba_image_confusion = _fold_confusion(HIBA_IMAGE_SUPPORT, fold_index)
        hiba_lesion_confusion = _fold_confusion(HIBA_LESION_SUPPORT, fold_index)
        if fail_hiba_scc:
            scc_index = PAD_UFES_NATIVE_LABELS.index("squamous_cell_carcinoma")
            ack_index = PAD_UFES_NATIVE_LABELS.index("actinic_keratosis")
            hiba_lesion_confusion[scc_index][ack_index] = hiba_lesion_confusion[scc_index][
                scc_index
            ]
            hiba_lesion_confusion[scc_index][scc_index] = 0
        test = _metric_bundle(pad_confusion, hiba_image_confusion, hiba_lesion_confusion)
        selected_train = _metric_bundle(pad_confusion, hiba_image_confusion, hiba_lesion_confusion)
        selected_train["primary_source_mean_macro_f1"] = 0.80
        selected_val = _metric_bundle(pad_confusion, hiba_image_confusion, hiba_lesion_confusion)
        selected_val["primary_source_mean_macro_f1"] = 0.70
        summary_base = {
            "num_folds": 5,
            "fold_index": fold_index,
            "test_outer_fold": fold_index,
            "validation_outer_fold": (fold_index + 1) % 5,
        }
        report = {
            "architecture": ARCHITECTURE,
            "input_mode": "image_only",
            "pretrained_weights": PRETRAINED_WEIGHTS if pretrained else "none",
            "pretrained_weights_id": (
                pretrained_weights_id(ARCHITECTURE, PRETRAINED_WEIGHTS) if pretrained else None
            ),
            "preprocessing": PREPROCESSING,
            "augmentation_profile": AUGMENTATION_PROFILE,
            "labels": list(PAD_UFES_NATIVE_LABELS),
            "sources": list(SOURCE_ORDER),
            "seed": 42,
            "fold_index": fold_index,
            "pad_protocol": PAD_PROTOCOL,
            "hiba_protocol": HIBA_PROTOCOL,
            "hiba_role": "multisource_development",
            "manifest_fingerprint": "a" * 64,
            "selection_metric": SELECTION_METRIC,
            "source_class_weighting": SOURCE_CLASS_WEIGHTING,
            "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
            "primary_units": {"pad_ufes": "image", "hiba": "lesion"},
            "hyperparameters": {
                "epochs": 15,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "optimizer": "AdamW",
                "schedule": "none",
                "augmentation_profile": AUGMENTATION_PROFILE,
                "label_smoothing": 0.0,
                "sampling": "random_shuffle_without_replacement",
                "source_class_weighting": SOURCE_CLASS_WEIGHTING,
                "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
            },
            "inference_parameter_count": 28_589_414,
            "source_total_raw_image_counts": {"pad_ufes": 2_298, "hiba": 309},
            "source_total_effective_unit_counts": {"pad_ufes": 2_298.0, "hiba": 308.0},
            "pad_split_summary": {"protocol": PAD_PROTOCOL, **summary_base},
            "hiba_split_summary": {"protocol": HIBA_PROTOCOL, **summary_base},
            "best_epoch": 7,
            "best_val_primary_source_mean_macro_f1": 0.70,
            "selected_train": selected_train,
            "selected_val": selected_val,
            "test": test,
            "checkpoint_sha256": f"{fold_index}" * 64,
            "checkpoint_bytes": 114_000_000,
        }
        fold_dir = root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")


class PadHibaConvnextCrossValidationTests(unittest.TestCase):
    def test_loads_rotating_manifests_and_preserves_lesion_mass(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            pad_dir, hiba_dir = _write_manifest_fixture(Path(tmp_dir))
            manifests = load_multi_source_manifests(
                pad_dir,
                hiba_dir,
                expected_raw_counts={"pad_ufes": 30, "hiba": 31},
                expected_effective_counts={"pad_ufes": 30.0, "hiba": 30.0},
            )

        self.assertEqual(len(manifests.folds), 5)
        self.assertEqual(len(manifests.fingerprint), 64)
        self.assertEqual(manifests.source_total_raw_image_counts, {"pad_ufes": 30, "hiba": 31})
        self.assertEqual(
            manifests.source_total_effective_unit_counts,
            {"pad_ufes": 30.0, "hiba": 30.0},
        )
        for rows in manifests.folds:
            lesion_mass = rows.loc[rows["source"] == "hiba"].groupby("unit_id")["view_mass"].sum()
            self.assertTrue((lesion_mass == 1.0).all())

    def test_weights_equalize_all_source_class_cells(self) -> None:
        rows = []
        for source in SOURCE_ORDER:
            for label in PAD_UFES_NATIVE_LABELS:
                rows.append({"source": source, "label": label, "view_mass": 1.0})
        rows.append(
            {
                "source": "hiba",
                "label": PAD_UFES_NATIVE_LABELS[0],
                "view_mass": 1.0,
            }
        )
        rows[6]["view_mass"] = 0.5
        rows[-1]["view_mass"] = 0.5
        frame = pd.DataFrame(rows)

        weights = source_class_image_weights(frame)

        self.assertAlmostEqual(float(weights.sum()), 12.0, places=5)
        frame["weight"] = weights.tolist()
        cell_totals = frame.groupby(["source", "label"])["weight"].sum()
        self.assertTrue(all(abs(total - 1.0) < 1e-6 for total in cell_totals))

    def test_hiba_primary_metrics_average_repeated_lesion_views(self) -> None:
        rows = []
        targets = []
        probabilities = []
        for source in SOURCE_ORDER:
            for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
                rows.append(
                    {
                        "source": source,
                        "label": label,
                        "unit_id": f"{source}-{label_index}",
                    }
                )
                targets.append(label_index)
                probability = [0.0] * len(PAD_UFES_NATIVE_LABELS)
                probability[label_index] = 1.0
                probabilities.append(probability)
        probabilities[6] = [0.1, 0.9, 0.0, 0.0, 0.0, 0.0]
        rows.append(
            {
                "source": "hiba",
                "label": PAD_UFES_NATIVE_LABELS[0],
                "unit_id": "hiba-0",
            }
        )
        targets.append(0)
        probabilities.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        metrics = metrics_from_probabilities(
            pd.DataFrame(rows),
            torch.tensor(probabilities),
            torch.tensor(targets),
        )

        self.assertLess(float(metrics["by_source"]["hiba"]["macro_f1"]), 1.0)
        self.assertEqual(float(metrics["hiba_lesion"]["macro_f1"]), 1.0)

    def test_summary_passes_only_when_every_locked_gate_passes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            passing_root = root / "passing"
            failing_root = root / "failing"
            _write_report_fixture(passing_root)
            _write_report_fixture(failing_root, fail_hiba_scc=True)

            passing = summarize_reports(passing_root, root / "passing-summary.json")
            failing = summarize_reports(failing_root, root / "failing-summary.json")
            passing_text = (root / "passing-summary.json").read_text(encoding="utf-8")

        self.assertTrue(passing["decision_rules"]["all_pass"])
        self.assertFalse(failing["decision_rules"]["pooled_hiba_scc_f1_gte_0_3000"])
        self.assertFalse(failing["decision_rules"]["all_pass"])
        self.assertNotIn("private-patient", passing_text)
        self.assertNotIn("ISIC_", passing_text)

    def test_summary_accepts_random_initialization_without_pretrained_identity(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            reports_root = root / "random-init"
            _write_report_fixture(reports_root, pretrained=False)

            summary = summarize_reports(
                reports_root,
                root / "random-init-summary.json",
                pretrained=False,
            )

        self.assertEqual(summary["pretrained_weights"], "none")
        self.assertIsNone(summary["pretrained_weights_id"])


if __name__ == "__main__":
    unittest.main()
