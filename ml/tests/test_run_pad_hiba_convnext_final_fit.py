from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ml.evaluation.schema import PAD_UFES_NATIVE_LABELS
from ml.training.run_pad_hiba_convnext_cv import (
    ARCHITECTURE,
    AUGMENTATION_PROFILE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    HIBA_VIEW_WEIGHTING,
    PREPROCESSING,
    PRETRAINED_WEIGHTS,
    SOURCE_CLASS_WEIGHTING,
    SOURCE_ORDER,
    MultiSourceManifests,
)
from ml.training.run_pad_hiba_convnext_final_fit import (
    EXPECTED_MANIFEST_FINGERPRINT,
    EXPECTED_MANIFEST_IDENTITY_FINGERPRINT,
    EXPECTED_SELECTED_EPOCHS,
    FAILED_GATE_KEYS,
    FINAL_EPOCHS,
    build_final_fit_rows,
    load_locked_cv_summary,
    manifest_identity_fingerprint,
    sha256_file,
)


def _locked_summary() -> dict[str, object]:
    decision_rules = {key: False for key in FAILED_GATE_KEYS}
    decision_rules["all_pass"] = False
    return {
        "architecture": ARCHITECTURE,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "pretrained_weights_id": "IMAGENET1K_V1",
        "preprocessing": PREPROCESSING,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "labels": list(PAD_UFES_NATIVE_LABELS),
        "sources": list(SOURCE_ORDER),
        "seed": DEFAULT_SEED,
        "epochs": 15,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "source_class_weighting": SOURCE_CLASS_WEIGHTING,
        "hiba_view_weighting": HIBA_VIEW_WEIGHTING,
        "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
        "folds": [
            {"fold_index": index, "best_epoch": epoch}
            for index, epoch in enumerate(EXPECTED_SELECTED_EPOCHS)
        ],
        "decision_rules": decision_rules,
    }


def _manifests() -> MultiSourceManifests:
    rows = pd.DataFrame(
        [
            {
                "split": "test",
                "source": "pad_ufes",
                "image_path": "/private/pad.jpg",
                "label": "melanoma",
                "unit_id": "pad-unit",
                "lesion_id": "pad-unit",
                "view_mass": 1.0,
            },
            {
                "split": "val",
                "source": "hiba",
                "image_path": "/private/hiba.jpg",
                "label": "melanoma",
                "unit_id": "hiba-unit",
                "lesion_id": "hiba-lesion",
                "view_mass": 1.0,
            },
        ]
    )
    return MultiSourceManifests(
        folds=(rows,),
        pad_fold_summaries=({},),
        hiba_fold_summaries=({},),
        fingerprint=EXPECTED_MANIFEST_FINGERPRINT,
        source_total_raw_image_counts={"pad_ufes": 1, "hiba": 1},
        source_total_effective_unit_counts={"pad_ufes": 1.0, "hiba": 1.0},
    )


class PadHibaConvnextFinalFitTests(unittest.TestCase):
    def test_accepts_exact_locked_summary_and_fixed_median_epoch(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "summary.json"
            path.write_text(json.dumps(_locked_summary(), sort_keys=True), encoding="utf-8")

            summary = load_locked_cv_summary(path, expected_sha256=sha256_file(path))

        self.assertFalse(summary["decision_rules"]["all_pass"])
        self.assertEqual(FINAL_EPOCHS, 11)

    def test_rejects_changed_selected_epoch(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "summary.json"
            summary = _locked_summary()
            summary["folds"][0]["best_epoch"] = 14
            path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_locked_cv_summary(path, expected_sha256=sha256_file(path))

    def test_final_fit_uses_each_approved_image_once_and_removes_split_roles(self) -> None:
        manifests = _manifests()
        rows = build_final_fit_rows(
            manifests,
            expected_manifest_identity_fingerprint=manifest_identity_fingerprint(manifests.folds),
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows["split"]), {"train"})
        self.assertEqual(rows["image_path"].nunique(), 2)

    def test_final_fit_rejects_manifest_drift(self) -> None:
        with self.assertRaises(ValueError):
            build_final_fit_rows(
                _manifests(),
                expected_manifest_identity_fingerprint="0" * 64,
            )

    def test_manifest_identity_fingerprint_ignores_machine_specific_parent_paths(self) -> None:
        manifests = _manifests()
        moved = manifests.folds[0].copy()
        moved["image_path"] = moved["image_path"].map(
            lambda value: f"D:\\cloud-staging\\{Path(value).name}"
        )

        self.assertEqual(
            manifest_identity_fingerprint(manifests.folds),
            manifest_identity_fingerprint((moved,)),
        )
        self.assertEqual(len(EXPECTED_MANIFEST_IDENTITY_FINGERPRINT), 64)


if __name__ == "__main__":
    unittest.main()
