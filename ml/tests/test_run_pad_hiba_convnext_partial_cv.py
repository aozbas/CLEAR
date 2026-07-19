from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch.nn as nn

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
)
from ml.training.run_pad_hiba_convnext_partial_cv import (
    DEVELOPMENT_PROTOCOL,
    FORMER_TEST_ROLE_USE,
    TRAINABLE_SCOPE,
    build_trainable_optimizer,
    configure_partial_freeze,
    development_rows_for_fold,
    summarize_reports,
    validate_report,
)
from ml.training.train_pad_ufes import build_transfer_model, pretrained_weights_id

PAD_SUPPORT = (730, 845, 52, 244, 192, 235)
HIBA_LESION_SUPPORT = (17, 112, 58, 53, 47, 21)
HIBA_IMAGE_SUPPORT = (17, 112, 59, 53, 47, 21)


class _TinyConvnextShape(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Sequential(nn.Linear(4, 4)),
            nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)),
        )
        self.classifier = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 6))


def _role_fixture() -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        for source in SOURCE_ORDER:
            for label_index, label in enumerate(PAD_UFES_NATIVE_LABELS):
                rows.append(
                    {
                        "split": split,
                        "source": source,
                        "label": label,
                        "image_path": f"/private/{split}-{source}-{label_index}.jpg",
                        "unit_id": f"{split}-{source}-{label_index}",
                        "lesion_id": f"{split}-{source}-{label_index}",
                        "view_mass": 1.0,
                    }
                )
    return pd.DataFrame(rows)


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


def _report(fold_index: int, *, fail_hiba_scc: bool = False) -> dict[str, object]:
    pad_confusion = _fold_confusion(PAD_SUPPORT, fold_index)
    hiba_image_confusion = _fold_confusion(HIBA_IMAGE_SUPPORT, fold_index)
    hiba_lesion_confusion = _fold_confusion(HIBA_LESION_SUPPORT, fold_index)
    if fail_hiba_scc:
        scc_index = PAD_UFES_NATIVE_LABELS.index("squamous_cell_carcinoma")
        ack_index = PAD_UFES_NATIVE_LABELS.index("actinic_keratosis")
        hiba_lesion_confusion[scc_index][ack_index] = hiba_lesion_confusion[scc_index][scc_index]
        hiba_lesion_confusion[scc_index][scc_index] = 0
    selected_val = _metric_bundle(pad_confusion, hiba_image_confusion, hiba_lesion_confusion)
    selected_train = copy.deepcopy(selected_val)
    selected_train["primary_source_mean_macro_f1"] = 0.90
    return {
        "development_protocol": DEVELOPMENT_PROTOCOL,
        "outer_test_scored": False,
        "former_test_role_use": FORMER_TEST_ROLE_USE,
        "architecture": ARCHITECTURE,
        "input_mode": "image_only",
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "pretrained_weights_id": pretrained_weights_id(ARCHITECTURE, PRETRAINED_WEIGHTS),
        "preprocessing": PREPROCESSING,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "seed": 42,
        "fold_index": fold_index,
        "validation_outer_fold": (fold_index + 1) % 5,
        "excluded_outer_fold": fold_index,
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
            "optimizer": "AdamW_trainable_parameters_only",
            "schedule": "none",
            "augmentation_profile": AUGMENTATION_PROFILE,
            "label_smoothing": 0.0,
            "sampling": "random_shuffle_without_replacement",
            "source_class_weighting": SOURCE_CLASS_WEIGHTING,
            "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
            "trainable_scope": TRAINABLE_SCOPE,
            "native_training_mode_behavior": True,
        },
        "profile": TRAINABLE_SCOPE,
        "trainable_parameter_count": 4_800_000,
        "inference_parameter_count": 27_824_742,
        "trainable_parameter_fraction": 4_800_000 / 27_824_742,
        "trainable_parameter_tensor_count": 12,
        "trainable_parameter_names_sha256": "b" * 64,
        "source_total_raw_image_counts": {"pad_ufes": 2_298, "hiba": 309},
        "source_total_effective_unit_counts": {"pad_ufes": 2_298.0, "hiba": 308.0},
        "best_epoch": 5,
        "best_val_primary_source_mean_macro_f1": float(
            selected_val["primary_source_mean_macro_f1"]
        ),
        "selected_train": selected_train,
        "selected_val": selected_val,
        "checkpoint_sha256": f"{fold_index}" * 64,
        "checkpoint_bytes": 114_000_000,
    }


def _write_reports(root: Path, *, fail_hiba_scc: bool = False) -> None:
    for fold_index in range(5):
        fold_dir = root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "report.json").write_text(
            json.dumps(_report(fold_index, fail_hiba_scc=fail_hiba_scc)),
            encoding="utf-8",
        )


class PadHibaPartialFreezeTests(unittest.TestCase):
    def test_development_rows_exclude_every_former_test_row(self) -> None:
        rows = _role_fixture()

        development = development_rows_for_fold(rows)

        self.assertEqual(set(development["split"]), {"train", "val"})
        self.assertEqual(len(development), 24)
        self.assertTrue(
            set(rows.loc[rows["split"] == "test", "image_path"]).isdisjoint(
                development["image_path"]
            )
        )

    def test_only_final_block_and_classifier_enter_optimizer(self) -> None:
        model = _TinyConvnextShape()

        scope = configure_partial_freeze(model)
        optimizer = build_trainable_optimizer(model)

        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(
            all(name.startswith(("features.1.1.", "classifier.")) for name in trainable_names)
        )
        self.assertTrue(any(name.startswith("features.1.1.") for name in trainable_names))
        self.assertTrue(any(name.startswith("classifier.") for name in trainable_names))
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        self.assertEqual(
            optimizer_ids,
            {id(parameter) for parameter in model.parameters() if parameter.requires_grad},
        )
        self.assertEqual(scope["profile"], TRAINABLE_SCOPE)
        self.assertEqual(len(str(scope["trainable_parameter_names_sha256"])), 64)

    def test_real_convnext_scope_stays_below_locked_capacity(self) -> None:
        model = build_transfer_model(architecture="convnext_tiny", weights="none")

        scope = configure_partial_freeze(model)

        self.assertLessEqual(int(scope["trainable_parameter_count"]), 5_000_000)
        self.assertLessEqual(float(scope["trainable_parameter_fraction"]), 0.2)
        self.assertLessEqual(int(scope["inference_parameter_count"]), 30_000_000)
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(
            all(name.startswith(("features.7.2.", "classifier.")) for name in trainable_names)
        )

    def test_report_rejects_any_test_metrics_object(self) -> None:
        report = _report(0)
        report["test"] = {}

        with self.assertRaisesRegex(ValueError, "forbidden test metrics object"):
            validate_report(report, fold_index=0)

    def test_summary_applies_all_validation_only_gates(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            passing_root = root / "passing"
            failing_root = root / "failing"
            _write_reports(passing_root)
            _write_reports(failing_root, fail_hiba_scc=True)

            passing = summarize_reports(passing_root, root / "passing-summary.json")
            failing = summarize_reports(failing_root, root / "failing-summary.json")
            passing_text = (root / "passing-summary.json").read_text(encoding="utf-8")

        self.assertTrue(passing["decision_rules"]["all_pass"])
        self.assertFalse(failing["decision_rules"]["pooled_validation_hiba_scc_f1_gte_0_3000"])
        self.assertFalse(failing["decision_rules"]["all_pass"])
        self.assertFalse(passing["outer_test_scored"])
        self.assertNotIn('"test"', passing_text)
        self.assertNotIn("ISIC_", passing_text)
        self.assertNotIn("/private/", passing_text)


if __name__ == "__main__":
    unittest.main()
